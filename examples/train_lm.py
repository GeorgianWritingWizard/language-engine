import evaluate
from datasets import load_dataset
import datasets
from transformers import AutoModelForMaskedLM, AutoTokenizer, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling, AlbertConfig, HfArgumentParser

parser = HfArgumentParser(TrainingArguments)

training_args = parser.parse_args_into_dataclasses()
training_args = training_args[0]

# TODO: this is dumb temporary
cf = {
  "architectures": [
    "AlbertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0,
  "bos_token_id": 2,
  "classifier_dropout_prob": 0.1,
  "down_scale_factor": 1,
  "embedding_size": 128,
  "eos_token_id": 3,
  "gap_size": 0,
  "hidden_act": "gelu_new",
  "hidden_dropout_prob": 0,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "inner_group_num": 1,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "albert",
  "net_structure_type": 0,
  "num_attention_heads": 12,
  "num_hidden_groups": 1,
  "num_hidden_layers": 12,
  "num_memory_blocks": 0,
  "pad_token_id": 0,
  "type_vocab_size": 2,
  "vocab_size": 30000
}

dataset = load_dataset('ZurabDz/tokenized_geo_data')
# dataset = datasets.load_from_disk(
#     '/home/penguin/GeorgianWritingWizard/data/processed_data')
splitted = dataset['train'].train_test_split(test_size=0.01, seed=42)

# config = AlbertConfig(**cf)
# tokenizer = AutoTokenizer.from_pretrained('ZurabDz/GeoSentencePieceBPE')
# config.vocab_size = tokenizer.vocab_size
# model = AutoModelForMaskedLM.from_config(config)

tokenizer = AutoTokenizer.from_pretrained('ZurabDz/albert-geo')
model = AutoModelForMaskedLM.from_pretrained('ZurabDz/albert-geo')


mlm_probability = 0.15
pad_to_multiple_of_8 = False

metric = evaluate.load("accuracy")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm_probability=mlm_probability,
    pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
)


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics
    labels = labels.reshape(-1)
    preds = preds.reshape(-1)
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]
    return metric.compute(predictions=preds, references=labels)


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


training_args = TrainingArguments(
    output_dir=training_args.output_dir,
    do_train=True,
    do_eval=True,
    do_predict=True,
    evaluation_strategy=training_args.evaluation_strategy,
    per_device_train_batch_size=training_args.per_device_train_batch_size,
    per_device_eval_batch_size=training_args.per_device_eval_batch_size,
    gradient_accumulation_steps=training_args.gradient_accumulation_steps,
    eval_accumulation_steps=training_args.eval_accumulation_steps,
    save_strategy=training_args.save_strategy,
    save_total_limit=4,
    save_steps=training_args.save_steps,
    eval_steps=training_args.eval_steps,
    logging_steps=training_args.logging_steps,
    # no_cuda=True,
    jit_mode_eval=training_args.jit_mode_eval,
    push_to_hub=training_args.push_to_hub,
    hub_model_id=training_args.hub_model_id,
    hub_token=training_args.hub_token,
    fp16=True,
    dataloader_num_workers=training_args.dataloader_num_workers,
    torch_compile=training_args.torch_compile
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=splitted['train'],
    eval_dataset=splitted['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

trainer.train()
