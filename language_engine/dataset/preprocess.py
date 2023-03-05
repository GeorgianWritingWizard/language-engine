import os
from itertools import chain

# TODO: wrapp parameters in hydra?
def huggingface_preprocessing(raw_dataset, tokenizer, num_threads=8, max_seq_length=128):
    """Dataset preprocessing and tokenization.

    This is basically the default HF routine from
    https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py
    """
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = getattr(raw_dataset, "column_names", "text")
    text_column_name = "text" if "text" in column_names else column_names[0]

    # TODO: this is dumb
    map_setup = dict(
        batched=True,
        batch_size=1024,
        num_proc=num_threads if num_threads > 0 else None,
        # load_from_cache_file=False,
        # keep_in_memory=False,
    )
    # TODO: define this outside
    # parellism_flag = os.environ["TOKENIZERS_PARALLELISM"]
    if num_threads > 0:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
    # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
    # efficient when it receives the `special_tokens_mask`.

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

    tokenizer.model_max_length = 1e30

    tokenized_dataset = raw_dataset.map(
        tokenize_function, remove_columns=column_names, desc="Running tokenizer on every text in dataset", **map_setup
    )



    # Main data processing function that will concatenate all texts from our dataset and generate chunks of
    # max_seq_length.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= max_seq_length:
            total_length = (total_length // max_seq_length) * max_seq_length
        # Split by chunks of max_len.

        result = {k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)] for k, t in concatenated_examples.items()}
        return result

    # TODO: shuffle data?
    
    tokenized_dataset = tokenized_dataset.map(group_texts, desc=f"Grouping texts in chunks of {max_seq_length}", **map_setup)

    # Finally flatten
    # This is necessary for the save_to_disk call that comes next. If skipped here, the call will be invoked from save_to_disk
    # This way, atleast it shares the same batch parameters and prints a progress bar.
    tokenized_dataset = tokenized_dataset.map(desc="Flattening the indices", **map_setup)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    return tokenized_dataset