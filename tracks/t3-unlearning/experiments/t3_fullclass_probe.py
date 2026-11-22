"""T3 probe: only flip images whose orig top-1 class looks fully-forgotten (~130 imgs).

Adem's simulator: 9 classes appear entirely in the forget set (count ~129-130).
Gold never saw those classes → high CE + structured wrong class.
Touch only those images; keep other forget images close to orig for retain gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
# allow importing helpers next to data or from t3-ami
sys.path[:0] = [str(Path.cwd()), str(Path.cwd() / "t3-ami"), str(ROOT.parent / "t3-ami")]

from t3_ami import build_model, preprocess, kl  # noqa: E402

TEMP = 2.0


@torch.no_grad()
def probs(model, images, batch=128):
    out = []
    for i in range(0, len(images), batch):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(preprocess(images[i : i + batch].cuda(), "cuda")).float()
        out.append(F.softmax(logits, 1).cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".")
    ap.add_argument("--out", default="candidate_fullclass_probe.pt")
    ap.add_argument("--min-count", type=int, default=120, help="class count to treat as fully forgotten")
    ap.add_argument("--max-count", type=int, default=135)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--lr", type=float, default=8e-5)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--anchor-weight", type=float, default=1.5)
    ap.add_argument("--holdout", type=int, default=500)
    ap.add_argument("--target-temp", type=float, default=0.35, help="<1 sharpens 2nd-best target")
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

    p_all = probs(base, images)
    top = p_all.argmax(1)
    counts = torch.bincount(top, minlength=100)
    full = [int(c) for c in range(100) if a.min_count <= int(counts[c]) <= a.max_count]
    print(json.dumps({"class_counts_top": sorted(((int(c), int(counts[c])) for c in range(100)), key=lambda x: -x[1])[:15], "full_classes": full, "n_full_images": int(sum(int(counts[c]) for c in full))}), flush=True)

    p0 = p_all[: n - a.holdout]
    top0 = top[: n - a.holdout]
    mask = torch.zeros(len(train), dtype=torch.bool)
    for c in full:
        mask |= top0 == c

    # Target: zero mass on memorized class, sharpen remaining (prefer 2nd best)
    t = p0.clone()
    rows = torch.arange(len(t))
    t[rows, top0] = 0.0
    t = t / t.sum(1, keepdim=True).clamp_min(1e-12)
    # sharpen
    t = t.clamp_min(1e-12).pow(1.0 / a.target_temp)
    t = t / t.sum(1, keepdim=True)

    # non-full images: stay on orig
    t[~mask] = p0[~mask]

    p_hold0 = probs(base, hold)

    model = build_model()
    model.load_state_dict(torch.load(d / "model.pt", map_location="cpu", weights_only=True), strict=True)
    model = model.cuda()
    depth = len(model.blocks)
    for i, blk in enumerate(model.blocks):
        if i < depth - a.blocks:
            for p in blk.parameters():
                p.requires_grad_(False)
    for p in model.norm.parameters():
        p.requires_grad_(True)
    for p in model.head.parameters():
        p.requires_grad_(True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)

    # sample: oversample full-class images
    idx_full = mask.nonzero(as_tuple=False).view(-1)
    idx_other = (~mask).nonzero(as_tuple=False).view(-1)
    print(f"train full={len(idx_full)} other={len(idx_other)} params={sum(p.numel() for p in params)}", flush=True)

    for s in range(a.steps):
        if len(idx_full) == 0:
            break
        n_f = min(a.batch // 2, len(idx_full))
        n_o = a.batch - n_f
        sel_f = idx_full[torch.randint(0, len(idx_full), (n_f,))]
        sel_o = idx_other[torch.randint(0, len(idx_other), (n_o,))] if len(idx_other) else sel_f[:0]
        idx = torch.cat([sel_f, sel_o])
        xb = train[idx].cuda()
        tb = t[idx].cuda()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(preprocess(xb, "cuda")).float()
        soft = F.softmax(logits / TEMP, 1)
        loss_forget = kl(tb[:n_f], soft[:n_f]) if n_f else soft.new_zeros(())
        # anchor others to orig
        if n_o:
            loss_anchor = kl(tb[n_f:], soft[n_f:])
        else:
            loss_anchor = soft.new_zeros(())
        loss = loss_forget + a.anchor_weight * loss_anchor
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if (s + 1) % 5 == 0 or s == 0:
            print(f"  step {s+1} loss {float(loss):.5f} forget {float(loss_forget):.5f} anchor {float(loss_anchor):.5f}", flush=True)

    model.eval()
    p1 = probs(model, train)
    p_hold = probs(model, hold)
    # CE proxy vs top0 as "label"
    ce = F.nll_loss(p1.clamp_min(1e-12).log(), top0, reduction="none")
    ce_full = ce[mask]
    ce_other = ce[~mask]
    metrics = {
        "full_classes": full,
        "n_full": int(mask.sum()),
        "conf_full": round(float(p1[mask].max(1).values.mean()) if mask.any() else 0, 4),
        "same_class_full": round(float((p1[mask].argmax(1) == top0[mask]).float().mean()) if mask.any() else 0, 4),
        "ce_mean_full": round(float(ce_full.mean()) if len(ce_full) else 0, 4),
        "ce_std_full": round(float(ce_full.std()) if len(ce_full) else 0, 4),
        "ce_mean_other": round(float(ce_other.mean()) if len(ce_other) else 0, 4),
        "same_class_other": round(float((p1[~mask].argmax(1) == top0[~mask]).float().mean()) if (~mask).any() else 0, 4),
        "kl_holdout": round(float(kl(p_hold0, p_hold)), 4),
        "accord_holdout": round(float((p_hold0.argmax(1) == p_hold.argmax(1)).float().mean()), 4),
        "steps": a.steps,
        "lr": a.lr,
        "anchor_weight": a.anchor_weight,
        "target_temp": a.target_temp,
    }
    print(json.dumps(metrics), flush=True)
    Path(a.out).with_suffix(".json").write_text(json.dumps(metrics, indent=2))
    torch.save({k: v.detach().cpu().float() for k, v in model.state_dict().items()}, a.out)
    print("wrote", a.out, Path(a.out).stat().st_size)


if __name__ == "__main__":
    main()
