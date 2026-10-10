"""STAGE 4 - Supervised multi-label head with Asymmetric Loss, initialised from the text prototypes.
Trains on saved embeddings only (seconds on CPU). Keeps the epoch with the best validation mAP.
Output: data/head.npz        Run: python 04_train_head.py
"""
import torch.nn as nn
from sklearn.metrics import average_precision_score
from common import *


class ASL(nn.Module):
    """Asymmetric loss: easy negatives are down-weighted; margin m forgives missing labels."""
    def __init__(self, gp=ASL_GAMMA_POS, gn=ASL_GAMMA_NEG, m=ASL_MARGIN, eps=1e-8):
        super().__init__(); self.gp, self.gn, self.m, self.eps = gp, gn, m, eps

    def forward(self, logits, y):
        p = torch.sigmoid(logits)
        pm = (p - self.m).clamp(min=0)
        lp = y * (1 - p) ** self.gp * torch.log(p.clamp(min=self.eps))
        ln = (1 - y) * pm ** self.gn * torch.log((1 - pm).clamp(min=self.eps))
        return -(lp + ln).sum() / y.shape[0]


def multihot(idx, recs, cidx):
    Y = np.zeros((len(idx), len(cidx)), "float32")
    for row, i in enumerate(idx):
        for c in recs[i]["labels"]:
            Y[row, cidx[c]] = 1
    return Y


if __name__ == "__main__":
    meta = load_meta(); recs = meta["records"]; e = np.load(DATA / "emb.npz")
    Z, T, keep = e["Z"], e["T"], e["keep"]
    cidx = {c: i for i, c in enumerate(meta["vocab"])}; C = len(cidx)
    tr = [i for i, r in enumerate(recs) if r["split"] == "train"]
    va = [i for i, r in enumerate(recs) if r["split"] == "val" and keep[i]]
    X, Y = torch.tensor(Z[tr]), torch.tensor(multihot(tr, recs, cidx))
    Xv, Yv = torch.tensor(Z[va]), multihot(va, recs, cidx)

    lin = nn.Linear(Z.shape[1], C)
    with torch.no_grad():                                      # text-initialised classifier
        lin.weight.copy_(torch.tensor(T) * HEAD_INIT_SCALE); lin.bias.fill_(HEAD_INIT_BIAS)
    opt = torch.optim.AdamW(lin.parameters(), lr=HEAD_LR, weight_decay=HEAD_WD)
    loss_fn = ASL()

    def val_map():
        if not len(va):
            return float("nan")
        with torch.no_grad():
            P = torch.sigmoid(lin(Xv)).numpy()
        aps = [average_precision_score(Yv[:, c], P[:, c]) for c in range(C) if Yv[:, c].sum() > 0]
        return float(np.mean(aps)) if aps else float("nan")

    print(f"epoch   0 (text init only)  val mAP {val_map():.4f}")
    best, best_state = -1.0, None
    for ep in range(1, HEAD_EPOCHS + 1):
        perm = torch.randperm(len(X)); tot = 0.0
        for i in range(0, len(X), HEAD_BS):
            b = perm[i:i + HEAD_BS]
            loss = loss_fn(lin(X[b]), Y[b]); opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        vm = val_map()
        if ep % 5 == 0 or ep == 1:
            print(f"epoch {ep:3d}  loss {tot / len(X):.4f}  val mAP {vm:.4f}")
        if np.isnan(vm) or vm > best:
            best = vm; best_state = (lin.weight.detach().clone(), lin.bias.detach().clone())
    W, b = best_state
    np.savez(DATA / "head.npz", W=W.numpy(), b=b.numpy())
    print(f"saved data/head.npz (best val mAP {best:.4f})")
