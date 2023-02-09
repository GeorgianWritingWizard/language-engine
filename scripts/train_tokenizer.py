from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import AlbertTokenizerFast
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--text_corpus_path', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--vocab_size', type=int, default=30_000, required=False)

args = parser.parse_args()


tk_tokenizer = Tokenizer(BPE())
tk_tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(vocab_size=args.vocab_size, special_tokens=["<unk>", "<cls>", "<sep>", "<pad>", "<mask>", "<s>", "</s>"])
tk_tokenizer.train(files=[f"{args.text_corpus_path}"], trainer=trainer)


tokenizer = AlbertTokenizerFast(tokenizer_object=tk_tokenizer)

tokenizer.bos_token = "<s>"
tokenizer.bos_token_id = tk_tokenizer.token_to_id("<s>")
tokenizer.pad_token = "<pad>"
tokenizer.pad_token_id = tk_tokenizer.token_to_id("<pad>")
tokenizer.eos_token = "</s>"
tokenizer.eos_token_id = tk_tokenizer.token_to_id("</s>")
tokenizer.unk_token = "<unk>"
tokenizer.unk_token_id = tk_tokenizer.token_to_id("<unk>")
tokenizer.cls_token = "<cls>"
tokenizer.cls_token_id = tk_tokenizer.token_to_id("<cls>")
tokenizer.sep_token = "<sep>"
tokenizer.sep_token_id = tk_tokenizer.token_to_id("<sep>")
tokenizer.mask_token = "<mask>"
tokenizer.mask_token_id = tk_tokenizer.token_to_id("<mask>")

tokenizer.save_pretrained(f'{args.output_dir}')
