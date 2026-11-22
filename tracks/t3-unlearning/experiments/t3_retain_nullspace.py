#!/usr/bin/env python3
"""EAC T3: retain-proxy nullspace projection of the CURRENT best PG head delta.

Abliteration-inspired *hypothesis*, not a benchmarked improvement.
Never project pretrained weights directly: only project W_best-W_orig.
Only forget images + mixup/cutmix from those images; NO external labeled data.
Strict float32 .pt outputs; no server/API calls and no SSH operations.

Example on a coordinated H200:
 python t3_retain_nullspace.py --orig ~/eac/t3/model.pt \
 --best ~/eac/t3/t3-methodes/candidates/PG_RDt24r_70_l1016c_s1.pt \
 --forget ~/eac/t3/forget_images.npz \
 --retain ~/eac/t3/t3-methodes/cache/mixes.npy \
 --t3base ~/eac/t3 --out ~/runs/t3_nullspace
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch


def data(path):
    x = np.load(Path(path).expanduser(), allow_pickle=False, mmap_mode="r")
    if isinstance(x, np.lib.npyio.NpzFile):
        x = x["images"] if "images" in x else x["features"]
    return x


def state(path):
    x = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    if not isinstance(x, dict) or "head.weight" not in x:
        raise ValueError("expected full timm ViT state_dict")
    for k, v in x.items():
        if not isinstance(v, torch.Tensor) or (v.is_floating_point() and
            (v.dtype != torch.float32 or not torch.isfinite(v).all())):
            raise ValueError("Invalid state_dict weight: " + k)
    return x


def features(x, orig, t3base, batch):
    if x.ndim == 2:
        return torch.tensor(np.array(x), dtype=torch.float32)
    if x.ndim != 4 or x.shape[1:] != (224, 224, 3) or x.dtype != np.uint8:
        raise ValueError("expected pre_logits[N,D] or uint8 images[N,224,224,3]")
    sys.path.insert(0, str(Path(t3base).expanduser()))
    import task_template as tt
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = tt.build_model().to(device).eval()
    model.load_state_dict(orig, strict=True)
    z = []
    with torch.inference_mode():
        for i in range(0, len(x), batch):
            xb = torch.tensor(np.asarray(x[i:i+batch]).copy(), device=device)
            xb = (xb.permute(0, 3, 1, 2).float() / 255 - .5) / .5
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                zi = model.forward_head(model.forward_features(xb), pre_logits=True)
            z.append(zi.float().cpu())
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(z)


def basis_for_retain(f, r):
    f = torch.nn.functional.normalize(f.double(), dim=1)
    r = torch.nn.functional.normalize(r.double(), dim=1)
    cov = (r.T @ r) / len(r) - (f.T @ f) / len(f)
    ev, V = torch.linalg.eigh((cov + cov.T) / 2)
    ids = torch.where(ev > max(1e-10, ev.abs().max().item() * 1e-5))[0].flip(0)
    return V[:, ids].float(), f.float(), r.float()


def save(path, sd):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(suffix=".pt", dir=str(path.parent))
    os.close(fd)
    try:
        torch.save(sd, name)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    if path.stat().st_size > 450_000_000:
        path.unlink()
        raise ValueError("model exceeds 450MB")
    h = hashlib.sha256()
    with path.open("rb") as fd:
        for b in iter(lambda: fd.read(4*1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("orig", "best", "forget", "retain", "out"):
        p.add_argument("--"+key, required=True)
    p.add_argument("--t3base", default="~/eac/t3")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--ranks", default="4,8,16,32")
    p.add_argument("--strengths", default=".25,.5")
    p.add_argument("--min-forget", type=float, default=.95)
    p.add_argument("--min-retain-reduction", type=float, default=.02)
    p.add_argument("--save-top", type=int, default=3)
    args = p.parse_args()
    orig, best = state(args.orig), state(args.best)
    if orig.keys() != best.keys() or any(orig[k].shape != best[k].shape for k in orig):
        raise ValueError("source and best architecture mismatch")
    f = features(data(args.forget), orig, args.t3base, args.batch)
    r = features(data(args.retain), orig, args.t3base, args.batch)
    if f.ndim != 2 or r.ndim != 2 or f.shape[1] != orig["head.weight"].shape[1]:
        raise ValueError("features/head incompatible")
    U, f, r = basis_for_retain(f, r)
    dw = best["head.weight"] - orig["head.weight"]
    base_f, base_r = f @ dw.T, r @ dw.T
    nf = torch.linalg.vector_norm(base_f).clamp_min(1e-12)
    nr = torch.linalg.vector_norm(base_r).clamp_min(1e-12)
    rows = []
    for rank in map(int, args.ranks.split(",")):
        if rank > U.shape[1] or rank < 1:
            continue
        V = U[:, :rank]
        for alpha in map(float, args.strengths.split(",")):
            if not 0 <= alpha <= 1:
                raise ValueError("alpha must be between 0 and 1")
            d = dw - alpha * (dw @ V) @ V.T
            fr = (torch.linalg.vector_norm(f @ d.T) / nf).item()
            rr = (torch.linalg.vector_norm(r @ d.T) / nr).item()
            rows.append(dict(rank=rank, alpha=alpha, forget_ratio=fr,
                             retain_ratio=rr, retain_improvement=1-rr, delta=d))
            print(f"rank={rank} alpha={alpha:g}: forget_ratio={fr:.4f} retain_ratio={rr:.4f}", flush=True)
    eligible = [x for x in rows if x["forget_ratio"] >= args.min_forget and
                x["retain_improvement"] >= args.min_retain_reduction]
    eligible.sort(key=lambda x: x["retain_improvement"], reverse=True)
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    for x in eligible[:args.save_top]:
        sd = dict(best)
        sd["head.weight"] = (orig["head.weight"]+x["delta"]).float().contiguous()
        tag = f"nullspace_r{x['rank']}_a{int(round(x['alpha']*100))}"
        path = out / (tag + ".pt")
        x["path"], x["sha256"] = str(path), save(path, sd)
        print("CANDIDATE", path, x["sha256"])
    report = [{k: v for k, v in x.items() if k != "delta"} for x in rows]
    (out / "screen.json").write_text(json.dumps({"results":report,
        "note":"Proxy-only. NOT actual forget/retain score or 70% gate."}, indent=2))
    if not eligible:
        print("No candidate passed conservative proxy screen; do not submit.")

if __name__ == "__main__":
    main()
