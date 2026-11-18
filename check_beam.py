
from generate import generate, beam_search
import torch
import torch.nn.functional as F
import sentencepiece as spm
import sacrebleu
from tqdm import tqdm
from datasets import load_from_disk
import model as M
import random

ds = load_from_disk("data/tokenized")
test = ds["test"].filter(lambda ex: len(ex["src"]) <= 48 and len(ex["tgt"]) <= 50)
print(len(test))
idx = random.Random(0).sample(range(len(test)), 1000)
sub = test.select(idx)
sp = spm.SentencePieceProcessor(model_file="tokenizer/bpe.model")

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
device = get_device()


M.dropout = 0.1
model = M.Transformer().to(device)
ckpt = torch.load("checkpoints/best.pt", map_location=device)
model.load_state_dict(ckpt["model"])
model.eval()

bos_id, eos_id = 2, 3
device = next(model.parameters()).device

def seq_logprob(model, src_ids, out_ids):
    """Суммарный log-prob перевода out_ids (+EOS) при teacher forcing."""
    with torch.no_grad():
        src = torch.tensor([src_ids], device=device)
        tgt = torch.tensor([[bos_id] + out_ids + [eos_id]], device=device)
        logits = model(src, tgt[:, :-1])
        logp = F.log_softmax(logits, dim=-1)
        tok = logp.gather(-1, tgt[:, 1:].unsqueeze(-1)).squeeze(-1)
        return tok.sum().item()

# --- 1. beam=1 == greedy ---
N1 = 100
bad = 0
for ex in sub.select(range(N1)):
    a = generate(model, ex["src"])
    b = beam_search(model, ex["src"], beam=1)
    if a != b:
        bad += 1
        print("DIFF:", sp.decode(ex["src"]), "|", sp.decode(a), "|", sp.decode(b))
print(f"[1] расхождений greedy vs beam=1: {bad}/{N1}")

# --- 2. форма результата ---
N2 = 100
alpha = 0.7
problems = 0
for ex in sub.select(range(N2)):
    out = beam_search(model, ex["src"], beam=4, alpha=alpha)
    if (not out) or bos_id in out or eos_id in out or len(out) >= 100:
        problems += 1
        print("ПРОБЛЕМА:", sp.decode(ex["src"]), out[:10])
print(f"[2] проблемных выводов: {problems}/{N2}")

# --- 3. beam не хуже greedy по нормированному правдоподобию ---
def norm_lp(src, out):
    return seq_logprob(model, src, out) / (len(out) + 2) ** alpha

N3 = 100
ok = 0
for ex in sub.select(range(N3)):
    g = generate(model, ex["src"])
    b = beam_search(model, ex["src"], beam=4, alpha=alpha)
    if norm_lp(ex["src"], b) >= norm_lp(ex["src"], g) - 1e-4:
        ok += 1
print(f"[3] beam >= greedy по норм. log-prob: {ok}/{N3} (ждём почти все)")

# --- 4 и 5. BLEU и ratio для разных beam ---
refs = [sp.decode(ex["tgt"]) for ex in sub]
for beam in [1, 2, 4, 8]:
    hyps = [sp.decode(beam_search(model, ex["src"], beam=beam, alpha=alpha))
            for ex in tqdm(sub, desc=f"beam={beam}")]
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    print(f"[4,5] beam={beam}: BLEU={bleu.score:.2f}, ratio={bleu.sys_len/bleu.ref_len:.3f}")

# --- 6. спорные примеры ---
tests = ["We want to get to the border.", "Let's take our time.",
         "It makes no difference to me whether Tom comes or not.",
         "I don't think Tom needs any more help."]
for t in tests:
    ids = sp.encode(t)
    print(t, "->", sp.decode(beam_search(model, ids, beam=4)))