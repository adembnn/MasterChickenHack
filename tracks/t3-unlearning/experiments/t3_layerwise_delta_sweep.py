#!/usr/bin/env python3
"""T3 / EAC: zero-training layerwise delta scaling of the *current validated best*.
Do NOT submit automatically. Generate alternatives, run dash.py (--sd) and have
Adem's independent verifier approve before his orchestrator submits anything.
No T2 endpoint requests, SSH operations, networking, or GPU jobs in this file.

Example on an authorized H200, after checking GPU allocation on issue #2:
python t3_layerwise_delta_sweep.py --orig ~/eac/t3/model.pt \
  --best ~/eac/t3/t3-methodes/candidates/PG_RDt24r_70_l1016c.pt \
  --out ~/eac/t3/layer_sweep --variants early_lo,early_hi,late_lo,late_hi,head_lo,head_hi
"""
import argparse
import hashlib
import os
from pathlib import Path
import tempfile
import torch

VARIANTS = {
    "early_lo": (("blocks.8.", "blocks.9."), 0.88),
    "early_hi": (("blocks.8.", "blocks.9."), 1.12),
    "late_lo": (("blocks.10.", "blocks.11."), 0.88),
    "late_hi": (("blocks.10.", "blocks.11."), 1.12),
    "head_lo": (("head.", "norm."), 0.88),
    "head_hi": (("head.", "norm."), 1.12),
    "all_lo": (None, 0.97),
    "all_hi": (None, 1.03),
}

def load_state(path):
    s = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(s, dict) or not s or not all(isinstance(v, torch.Tensor) for v in s.values()):
        raise ValueError("Expected full model state_dict with tensor values only: " + str(path))
    return s

def sweep(orig, best, name):
    prefixes, alpha = VARIANTS[name]
    if orig.keys() != best.keys():
        raise ValueError("original/best state_dict keys differ")
    if not any(k.startswith("blocks.11.") for k in orig):
        raise ValueError("Expected ViT-B blocks.11. keys")
    changed = 0
    out = {}
    for k, a in orig.items():
        b = best[k]
        if a.shape != b.shape or torch.is_floating_point(a) != torch.is_floating_point(b):
            raise ValueError("shape/dtype mismatch: " + k)
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError("NaN or Inf: " + k)
        if not torch.is_floating_point(a):
            if not torch.equal(a, b):
                raise ValueError("nonfloating buffer differs: " + k)
            out[k] = a.contiguous()
            continue
        coeff = alpha if prefixes is None or k.startswith(prefixes) else 1.
        av, bv = a.float(), b.float()
        out[k] = (av + coeff * (bv - av)).contiguous()
        if coeff != 1. and not torch.equal(av, bv):
            changed += 1
        if out[k].dtype != torch.float32 or not torch.isfinite(out[k]).all():
            raise ValueError("invalid output tensor: " + k)
    if changed == 0:
        raise ValueError("no changed weights in group " + name)
    return out, changed

def write_atomic(state, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as f:
        temp_name = f.name
    try:
        torch.save(state, temp_name)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    size = path.stat().st_size
    if size > 450_000_000:
        path.unlink()
        raise ValueError("EAC 450 MB cap exceeded")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return size, h.hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--orig", required=True)
    p.add_argument("--best", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    orig, best = load_state(args.orig), load_state(args.best)
    if all(torch.equal(orig[k], best[k]) for k in orig):
        p.error("best is identical to original")
    for name in args.variants.split(","):
        if name not in VARIANTS:
            p.error("unknown variant: " + name)
        state, n = sweep(orig, best, name)
        if args.dry_run:
            print("DRY", name, "scaled tensors", n)
            continue
        output = Path(args.out) / ("PG_RDt24r_70_l1016c_layer_" + name + ".pt")
        size, sha = write_atomic(state, output)
        print(name, str(output), size, sha, "tensors", n, flush=True)

if __name__ == "__main__":
    main()
