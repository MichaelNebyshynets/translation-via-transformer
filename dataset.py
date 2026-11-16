import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


PAD_ID = 0

class TranslationDataset(Dataset):
    def __init__(self, split):
        self.split = split

    def __len__(self):
        return len(self.split)

    def __getitem__(self, i):
        ex = self.split[i]
        return ex["src"], ex["tgt"]

def collate_fn(batch):
    src = [torch.tensor(s) for s, t in batch]
    tgt_in = [torch.tensor(t[:-1]) for s, t in batch]
    tgt_out = [torch.tensor(t[1:]) for s, t in batch]

    src = pad_sequence(src, batch_first=True, padding_value=PAD_ID)
    tgt_in = pad_sequence(tgt_in, batch_first=True, padding_value=PAD_ID)
    tgt_out = pad_sequence(tgt_out, batch_first=True, padding_value=PAD_ID)
    return src, tgt_in, tgt_out