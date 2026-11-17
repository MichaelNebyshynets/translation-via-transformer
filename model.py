import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from dataset import TranslationDataset, PAD_ID, collate_fn

dropout=0.2
vocab_size = 16000
n_embd = 256
max_len = 128
n_layers = 3
n_head = 4

class Head(nn.Module):
    def __init__(self, head_size, n_embd):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_q, x_kv, key_pad_mask=None, causal=False):
        q = self.query(x_q)
        k = self.key(x_kv)
        v = self.value(x_kv)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        if key_pad_mask is not None:
            wei = wei.masked_fill(key_pad_mask[:, None, :], float('-inf'))
        if causal:
            Tq, Tk = wei.shape[-2:]
            future = torch.triu(torch.ones(Tq, Tk, dtype=bool, device=wei.device), diagonal=1)
            wei = wei.masked_fill(future, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ v

class MultiHead(nn.Module):
    def __init__(self, num_heads, head_size, n_embd):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads*head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_q, x_kv, key_pad_mask=None, causal=False):
        out = torch.cat([h(x_q, x_kv, key_pad_mask, causal) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)



class EncoderBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHead(n_head, head_size, n_embd)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, pad_mask):
        h = self.ln1(x)
        x = x + self.sa(h, h, pad_mask)
        x = x + self.ffwd(self.ln2(x))
        return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed_table = nn.Embedding(vocab_size, n_embd)
        self.pos_embed_table = nn.Embedding(max_len, n_embd)
        self.blocks = nn.ModuleList([EncoderBlock(n_embd, n_head) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, src):
        B, S = src.shape
        pad_mask = src == PAD_ID
        tok_emb = self.token_embed_table(src)
        pos_emb = self.pos_embed_table(torch.arange(S, device = src.device))
        x = self.drop(tok_emb + pos_emb)
        for block in self.blocks:
            x = block(x, pad_mask)
        x = self.ln_f(x)
        return x ,pad_mask


class DecoderBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHead(n_head, head_size, n_embd)
        self.ca = MultiHead(n_head, head_size, n_embd)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)

    def forward(self, x, enc_out, tgt_pad_mask, src_pad_mask):
        h = self.ln1(x)
        x = x + self.sa(h, h, tgt_pad_mask, causal=True)
        h = self.ln2(x)
        x = x + self.ca(h, enc_out, src_pad_mask)
        x = x + self.ffwd(self.ln3(x))
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed_table = nn.Embedding(vocab_size, n_embd)
        self.pos_embed_table = nn.Embedding(max_len, n_embd)
        self.blocks = nn.ModuleList([DecoderBlock(n_embd, n_head) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, tgt_in, enc_out, src_pad_mask):
        tgt_pad_mask = tgt_in == PAD_ID
        B, T = tgt_in.shape
        tok_emb = self.token_embed_table(tgt_in)
        pos_emb = self.pos_embed_table(torch.arange(T, device=tgt_in.device))
        x = self.drop(tok_emb + pos_emb)
        for block in self.blocks:
            x = block(x, enc_out, tgt_pad_mask, src_pad_mask)
        x = self.ln_f(x)
        x = self.lm_head(x)
        return x


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, src, tgt_in):
        enc_out, src_pad_mask = self.encoder(src)
        return self.decoder(tgt_in, enc_out, src_pad_mask)