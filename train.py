import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from datasets import load_from_disk
from dataset import TranslationDataset, collate_fn
import model as M
import math
import time
import os
import sentencepiece as spm
from generate import generate

sp = spm.SentencePieceProcessor(model_file="tokenizer/bpe.model")
test_sentences = ["I love you.", "Where is the station?", "Tom is my friend.",
                  "What time is it?", "She doesn't like coffee."]


CKPT_DIR = os.environ.get("CKPT_DIR", "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

def show_samples():
    for s in test_sentences:
        print(s, "->", sp.decode(generate(model, sp.encode(s))))
    model.train()

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(model, val_loader):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for (src, tgt_in, tgt_out) in val_loader:
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            logits = model(src, tgt_in)
            loss = F.cross_entropy(logits.view(-1, M.vocab_size), tgt_out.view(-1), ignore_index=M.PAD_ID, reduction="sum")
            total_loss += loss.item()
            total_tokens += (tgt_out != M.PAD_ID).sum().item()
    model.train()
    avg = total_loss / total_tokens
    return avg, math.exp(avg)


device = get_device()
print(device)
M.dropout = 0.1
model = M.Transformer().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.98), weight_decay=0.01)


epochs = 3
start_epoch, step = 0, 0
last_path = f"{CKPT_DIR}/last.pt"
best_val = float("inf")
if os.path.exists(last_path):
    ck = torch.load(last_path, map_location=device)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    start_epoch, step, best_val = ck["epoch"] + 1, ck["step"], ck["best_val"]
    print(f"Resumed: epoch {start_epoch}, step {step}")

ds = load_from_disk("data/tokenized")
train_ds = TranslationDataset(ds["train"])
train_ds = torch.utils.data.Subset(train_ds, range(6400)) 
val_ds = TranslationDataset(ds["validation"])
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, collate_fn=collate_fn)




def get_lr(step, total_steps, peak=5e-4, warmup=1000, min_ratio=0.1):
    if step < warmup:
        return peak * (step+1)/warmup
    progress = (step-warmup) / max(1, total_steps-warmup)
    min_lr = peak * min_ratio
    return min_lr + 0.5 * (peak-min_lr) * (1 + math.cos(math.pi*progress))

model.train()
total_steps = len(train_loader) * epochs
start = time.time()
for epoch in range(start_epoch, epochs):
    for (src, tgt_in, tgt_out) in train_loader:
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
        lr = get_lr(step, total_steps)
        opt.param_groups[0]["lr"] = lr
        logits = model(src, tgt_in)
        loss = F.cross_entropy(logits.view(-1, M.vocab_size), tgt_out.view(-1), ignore_index=M.PAD_ID, label_smoothing=0.1)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 50 == 0:
            print(f"Step {step}: loss={loss.item():.3f}, lr={lr:.2e}, steps/s={50/(time.time()-start):.2f}")
            start = time.time()
        if step % 500 == 0:                  
            val_loss, val_ppl = evaluate(model, val_loader)
            print(f"step {step}: val_loss={val_loss:.3f}, ppl={val_ppl:.1f}")
            show_samples()

    val_loss, val_ppl = evaluate(model, val_loader)
    print(f"epoch {epoch} done: val_loss={val_loss:.3f}, ppl={val_ppl:.1f}")
    if val_loss < best_val:
        best_val = val_loss
        torch.save({"model": model.state_dict(), "step": step, "val_loss": val_loss},
                   f"{CKPT_DIR}/best.pt")
    show_samples()
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "epoch": epoch, "step": step, "best_val": best_val}, last_path)

