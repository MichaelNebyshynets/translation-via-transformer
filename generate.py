import torch
import torch.nn.functional as F

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



def beam_search(model, src_ids, beam=4, max_steps=100, alpha=0.7):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        src = torch.tensor([src_ids], device=device)
        enc_out, src_pad_mask = model.encoder(src)

        beams = [([bos_id], 0.0)]      # (токены, сумма log-prob)
        finished = []

        for _ in range(max_steps):
            candidates = []
            for ys_list, score in beams:
                ys = torch.tensor([ys_list], device=device)
                logits = model.decoder(ys, enc_out, src_pad_mask)
                logp = F.log_softmax(logits[0, -1], dim=-1)
                top_lp, top_id = logp.topk(beam)
                for lp, i in zip(top_lp.tolist(), top_id.tolist()):
                    candidates.append((ys_list + [i], score + lp))

            candidates.sort(key=lambda c: c[1], reverse=True)

            beams = []
            for ys_list, score in candidates[:beam]:
                if ys_list[-1] == eos_id:
                    finished.append((ys_list, score))
                else:
                    beams.append((ys_list, score))

            if not beams:
                break

        if not finished:               # ни один луч не дошёл до EOS
            finished = beams

        best = max(finished, key=lambda c: c[1] / len(c[0]) ** alpha)
        out = best[0][1:]              # без BOS
        if out and out[-1] == eos_id:
            out = out[:-1]
        return out
