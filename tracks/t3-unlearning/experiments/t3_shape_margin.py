"""T3: Adem simulator shape — lower true class under max of others; keep orig ranking.

On flipped images (top entropy fraction): set true-class logit = max_other - L.
On the rest: soft orig (logits / soft_div). Anchor KL to orig on holdout/proxy mix.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path[:0] = [str(Path.cwd()), str(Path.cwd() / "t3-ami")]
from t3_ami import build_model, preprocess  # noqa: E402

TEMP = 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".")
    ap.add_argument("--out", default="candidate_shape_margin.pt")
    ap.add_argument("--frac", type=float, default=0.35)
    ap.add_argument("--margin", type=float, default=10.0, help="true logit = max_other - margin")
    ap.add_argument("--soft-div", type=float, default=1.5)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--anchor-weight", type=float, default=2.0)
    ap.add_argument("--holdout", type=int, default=400)
    a = ap.parse_args()

    d = Path(a.data)
    with np.load(d / "forget_images.npz", allow_pickle=False) as z:
        images = torch.from_numpy(z["images"].copy())
    n = len(images)
    hold = images[n - a.holdout :]
    train = images[: n - a.holdout]

    base = build_model()
    base.load_state_dict(torch.load(d / "model.pt", map_location="cpu", weights_only=True), strict=True)
    base = base.cuda().eval()

    student = build_model()
    student.load_state_dict(base.state_dict())
    student = student.cuda().train()

    # freeze early
    for p in student.parameters():
        p.requires_grad = False
    blocks = list(student.blocks)
    for b in blocks[-a.blocks :]:
        for p in b.parameters():
            p.requires_grad = True
    for name, p in student.named_parameters():
        if any(k in name for k in ("norm", "head", "fc_norm")):
            p.requires_grad = True

    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=a.lr)

    @torch.no_grad()
    def logits_of(model, imgs):
        outs = []
        for i in range(0, len(imgs), 128):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outs.append(model(preprocess(imgs[i : i + 128].cuda(), "cuda")).float().cpu())
        return torch.cat(outs)

    L0 = logits_of(base, train)
    ent = -(F.softmax(L0 / TEMP, 1) * F.log_softmax(L0 / TEMP, 1)).sum(1)
    k = max(1, int(round(a.frac * len(train))))
    flip_idx = ent.topk(k).indices
    flip = torch.zeros(len(train), dtype=torch.bool)
    flip[flip_idx] = True

    # build targets in logit space then softmax T=2 for KL
    targets = L0.clone()
    rows = torch.arange(len(train))
    top = L0.argmax(1)
    # soft non-flip
    targets[~flip] = L0[~flip] / a.soft_div
    # flip: lower true class under max of others
    for i in flip_idx.tolist():
        row = L0[i].clone()
        t = int(top[i])
        others = row.clone()
        others[t] = -1e9
        row[t] = others.max() - a.margin
        targets[i] = row
    q = F.softmax(targets / TEMP, 1)

    print(
        json.dumps(
            {
                "n_flip": int(flip.sum()),
                "frac": a.frac,
                "margin": a.margin,
                "soft_div": a.soft_div,
                "mean_ent_flip": float(ent[flip].mean()),
                "mean_ent_keep": float(ent[~flip].mean()),
            }
        ),
        flush=True,
    )

    for step in range(a.steps):
        idx = torch.randperm(len(train))[: a.batch]
        x = train[idx].cuda()
        qt = q[idx].cuda()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = student(preprocess(x, "cuda")).float()
        p = F.log_softmax(logits / TEMP, 1)
        loss_f = F.kl_div(p, qt, reduction="batchmean")
        # anchor on holdout
        hidx = torch.randint(0, len(hold), (min(a.batch, len(hold)),))
        hx = hold[hidx].cuda()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            t_h = F.softmax(base(preprocess(hx, "cuda")).float() / TEMP, 1)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            s_h = F.log_softmax(student(preprocess(hx, "cuda")).float() / TEMP, 1)
        loss_a = F.kl_div(s_h, t_h, reduction="batchmean")
        loss = loss_f + a.anchor_weight * loss_a
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 10 == 0 or step == a.steps - 1:
            print(f"step={step} loss={float(loss):.4f} forget={float(loss_f):.4f} anchor={float(loss_a):.4f}", flush=True)

    student.eval()
    # quick metrics vs base on flip/keep/hold
    with torch.no_grad():
        Lf = logits_of(student, train[flip][:256] if flip.any() else train[:256])
        Lb = L0[flip][:256] if flip.any() else L0[:256]
        same = (Lf.argmax(1) == Lb.argmax(1)).float().mean()
        Lh = logits_of(student, hold[:256])
        Lhb = logits_of(base, hold[:256])
        hold_same = (Lh.argmax(1) == Lhb.argmax(1)).float().mean()
    meta = {
        "frac": a.frac,
        "margin": a.margin,
        "soft_div": a.soft_div,
        "steps": a.steps,
        "flip_same_as_orig": float(same),
        "hold_same_as_orig": float(hold_same),
    }
    out = Path(a.out)
    torch.save({k: v.detach().cpu().float() for k, v in student.state_dict().items()}, out)
    Path(str(out) + ".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
