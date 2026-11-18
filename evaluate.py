"""Оценка BLEU на тестовой выборке.

Примеры:
    python evaluate.py                  # greedy, 1000 пар
    python evaluate.py --beam 4         # beam search с шириной 4
    python evaluate.py --beam 4 --n 300 # быстрее, на 300 парах
"""
import argparse
import random

import sacrebleu
import sentencepiece as spm
import torch
from datasets import load_from_disk
from tqdm import tqdm

import model as M
from generate import generate, beam_search

parser = argparse.ArgumentParser()
parser.add_argument("--beam", type=int, default=1, help="ширина луча (1 = greedy)")
parser.add_argument("--n", type=int, default=1000, help="сколько тестовых пар взять")
parser.add_argument("--alpha", type=float, default=0.7, help="степень нормировки длины в beam search")
parser.add_argument("--ckpt", default="checkpoints/best.pt")
parser.add_argument("--show", type=int, default=15, help="сколько примеров напечатать")
args = parser.parse_args()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = get_device()
sp = spm.SentencePieceProcessor(model_file="tokenizer/bpe.model")

M.dropout = 0.1
model = M.Transformer().to(device)
ckpt = torch.load(args.ckpt, map_location=device)
model.load_state_dict(ckpt["model"])
model.eval()

# Тот же порог длины, что и при обучении (позиционные эмбеддинги ограничены max_len=128)
ds = load_from_disk("data/tokenized")
test = ds["test"].filter(lambda ex: len(ex["src"]) <= 48 and len(ex["tgt"]) <= 50)
idx = random.Random(0).sample(range(len(test)), args.n)
sub = test.select(idx)
print(f"тестовых пар после фильтра: {len(test)}, берём: {len(sub)}, beam={args.beam}")

hyps, refs, srcs = [], [], []
for ex in tqdm(sub):
    if args.beam == 1:
        out = generate(model, ex["src"])
    else:
        out = beam_search(model, ex["src"], beam=args.beam, alpha=args.alpha)
    hyps.append(sp.decode(out))
    refs.append(sp.decode(ex["tgt"]))
    srcs.append(sp.decode(ex["src"]))

bleu = sacrebleu.corpus_bleu(hyps, [refs])
print(bleu)

for s, h, r in list(zip(srcs, hyps, refs))[: args.show]:
    print("EN :", s)
    print("HYP:", h)
    print("REF:", r)
    print()
