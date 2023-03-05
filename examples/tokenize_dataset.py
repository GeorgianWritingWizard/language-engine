"""
For now model is training is planned on kaggle, 2 low hz cpus are worst for tokenizing large data
especially when experimenting, this script will preprocess it and make it `ready to train`
"""

from language_engine.dataset.preprocess import huggingface_preprocessing
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
import os

NUM_THREADS = os.cpu_count()

#TODO :Make This script
tokenizer = PreTrainedTokenizerFast.from_pretrained('ZurabDz/GeoSentencePieceBPE_32768_v2')
dataset = load_dataset('ZurabDz/geo_small_corpus_dedublicated_trash_off')

tokenized_dataset = huggingface_preprocessing(dataset['train'], tokenizer, NUM_THREADS)
tokenized_dataset.save_to_disk('./processed_data')