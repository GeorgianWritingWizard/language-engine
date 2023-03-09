import evaluate
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling, AlbertConfig, HfArgumentParser
from dataclasses import dataclass, field

@dataclass
class DatasetArguments:
    dataset_name: str 
    test_size: float = field(default=0.05)
    dataset_seed: int = field(default=42)

@dataclass
class TokenizerArguments:
    tokenizer_name: str
    model_max_length: int = field(default=128)

@dataclass
class ModelArgumnts:
    model_name: str
    from_scratch: bool = field(default=True)
    mlm_probability: float = field(default=0.15)
    pad_to_multiple_of_8: bool = field(default=False)

parser = HfArgumentParser((TrainingArguments, DatasetArguments, TokenizerArguments, ModelArgumnts))

training_args, dataset_args, tokenizer_args, model_args = parser.parse_args_into_dataclasses()

dataset = load_dataset(dataset_args.dataset_name)
splitted = dataset['train'].train_test_split(test_size=dataset_args.test_size, seed=dataset_args.dataset_seed)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_args.tokenizer_name, model_max_length=tokenizer_args.model_max_length)

if model_args.from_scratch:
    config = AlbertConfig.from_pretrained(model_args.model_name)
    model = AutoModelForMaskedLM.from_config(config=config)
else:
    print("REUSING WEIGHTS")
    model = AutoModelForMaskedLM.from_pretrained(model_args.model_name)

# in case token vocab is different
embedding_size = model.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size:
    print("RESIZING EMBEDDINGS")
    model.resize_token_embeddings(len(tokenizer))

metric = evaluate.load("accuracy")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm_probability=model_args.mlm_probability
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
    fp16=training_args.fp16,
    dataloader_num_workers=training_args.dataloader_num_workers,
    torch_compile=training_args.torch_compile,
    learning_rate=training_args.learning_rate,
    warmup_ratio=training_args.warmup_ratio,
    num_train_epochs=training_args.num_train_epochs
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
