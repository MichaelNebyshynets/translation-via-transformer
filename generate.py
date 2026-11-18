import torch

bos_id=2
eos_id=3


def generate(model, src_ids, max_steps=100):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        src = torch.tensor([src_ids], device=device)
        enc_out, src_pad_mask = model.encoder(src)
        ys = torch.tensor([[bos_id]], device=device)
        for _ in range(max_steps):
            logits = model.decoder(ys, enc_out, src_pad_mask)
            next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
            ys = torch.cat((ys, next_id), dim=1)
            if next_id == eos_id:
                break
        out = ys[0, 1:].tolist()
        if out and out[-1] == eos_id:
            out = out[:-1]
    return out