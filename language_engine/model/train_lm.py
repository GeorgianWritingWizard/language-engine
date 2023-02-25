import evaluate
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling


dataset = load_dataset('ZurabDz/processed_dataset')
model = AutoModelForMaskedLM.from_pretrained('ZurabDz/model')
tokenizer = AutoModelForMaskedLM.from_pretrained('ZurabDz/tokenizer')

splitted = dataset.split('')

pad_to_max_length = True
max_seq_length = 128
mlm_probability = 0.15
pad_to_multiple_of_8 = False

#TODO: model.resize_embed to vocab size
column_names = list(dataset["train"].features)
text_column_name = "text" if "text" in column_names else column_names[0]
padding = "max_length" if pad_to_max_length else False
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
...
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=splitted['train'],
    eval_dataset=splitted['eval'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics
)