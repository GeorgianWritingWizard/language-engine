"""
Traning almost any tokenizer
"""
from datasets import load_dataset
import hydra
from language_engine.tokenizer.train_tokenizer import construct_tokenizer


# TODO: fix changing directory warning
# TODO: add on disk data loading
@hydra.main(config_path="../language_engine/configs", config_name="tokenizer", version_base="1.1")
def train_tokenizers(cfg):
    if cfg.dataset_name.endswith('.txt'):
        data = load_dataset('text', data_files={'train': [cfg.dataset_name]})
    else:
        data = load_dataset(cfg.dataset_name)
    tokenizer = construct_tokenizer(data['train'], cfg, path=None)
    tokenizer.save_pretrained('tokenizer')

train_tokenizers()
