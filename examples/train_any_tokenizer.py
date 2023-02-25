from datasets import load_dataset
import hydra
from language_engine.tokenizer.train_tokenizer import construct_tokenizer


@hydra.main(config_path="language_engine/language_engine/configs", config_name="tokenizer")
def train_tokenizers(cfg):
    data = load_dataset('text', data_files=['/home/penguin/GeorgianWritingWizard/data/data.txt'])
    tokenizer = construct_tokenizer(data['train'], cfg, path=None)
    tokenizer.save_pretrained('tokenizer')
