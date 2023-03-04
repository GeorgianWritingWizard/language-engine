from datasets import load_dataset
import datasets
from transformers import AutoTokenizer, AlbertTokenizer

raw_dataset = datasets.load_from_disk('/home/penguin/GeorgianWritingWizard/language_engine/examples/filtered_data')

eng_tok = AutoTokenizer.from_pretrained('albert-base-v2')
new_tok = eng_tok.train_new_from_iterator(raw_dataset, vocab_size=30_000)

new_tok.save_pretrained('/home/penguin/GeorgianWritingWizard/language_engine/examples/ff_tok')