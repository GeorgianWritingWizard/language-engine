"""
For now model is training is planned on kaggle, 2 low hz cpus are worst for tokenizing large data
especially when experimenting, this script will preprocess it and make it `ready to train`
"""

from language_engine.dataset.preprocess import huggingface_preprocessing
from datasets import load_dataset
from transformers import AutoTokenizer
import os

NUM_THREADS = os.cpu_count()

#TODO :Make This script
tokenizer = AutoTokenizer.from_pretrained('/home/penguin/GeorgianWritingWizard/language_engine/examples/outputs/2023-03-09/22-36-38/tokenizer')
dataset = load_dataset('text', data_files={'train': ['/home/penguin/GeorgianWritingWizard/data/test_data.txt']})
# dataset = load_dataset('ZurabDz/geo_small_corpus')

tokenized_dataset = huggingface_preprocessing(dataset['train'], tokenizer, NUM_THREADS, max_seq_length=128)
tokenized_dataset.save_to_disk('./processed_data')