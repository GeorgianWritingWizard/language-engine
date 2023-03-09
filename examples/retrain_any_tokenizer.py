# TODO: make it script runnable with hydra

from datasets import load_dataset
from transformers import AutoTokenizer


def get_iterator(data, batch_size):
    for batch in data['train'].iter(batch_size):
        yield batch['text']


data = load_dataset('ZurabDz/geo_small_corpus_dedublicated_trash_off')
iterator = get_iterator(data, 10_000)


albert_tokenizer = AutoTokenizer.from_pretrained('albert-base-v2')
new_tokenizer = albert_tokenizer.train_new_from_iterator(iterator, albert_tokenizer.vocab_size)

new_tokenizer.save_pretrained('new_tokenizer')