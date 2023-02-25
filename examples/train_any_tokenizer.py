from datasets import load_dataset
import hydra
from language_engine.tokenizer.train_tokenizer import construct_tokenizer


# TODO: fix changing directory warning
@hydra.main(config_path="../language_engine/configs", config_name="tokenizer", version_base="1.1")
def train_tokenizers(cfg):
    data = load_dataset('text', cfg.dataset_name)
    tokenizer = construct_tokenizer(data['train'], cfg, path=None)
    tokenizer.save_pretrained('tokenizer')

train_tokenizers()
