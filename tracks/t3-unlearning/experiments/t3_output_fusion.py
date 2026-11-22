"""Screen calibrated probability ensembles of T3 checkpoints before any distillation."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--members", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--images", type=int, default=1536)
    p.add_argument("--batch", type=int, default=96)
    p.add_argument("--capture-device", choices=("cpu", "cuda"), default="cuda")
    q = p.parse_args()
    for key in ("base", "parent", "out"):
        setattr(q, key, getattr(q, key).expanduser().resolve())
    q.members = [x.expanduser().resolve() for x in q.members]
    return q


def suppress_original(prob, labels):
    masked = prob.scatter(1, labels, 0)
    return masked / masked.sum(dim=1, keepdim=True).clamp_min(1e-20)


def ce_preserving_mix(parent, others, labels, weight):
    """Average the false-class distributions at T2, then restore the parent p(y) at T1."""
    base_conditional = suppress_original(F.softmax(parent / 2, dim=1), labels)
    mixed_conditional = (1 - weight) * base_conditional + weight * others
    false_p1 = mixed_conditional.square()
    false_p1 /= false_p1.sum(dim=1, keepdim=True)
    source = F.log_softmax(parent, dim=1)
    keep = source.gather(1, labels)
    # Work in float64 to preserve small false-class probabilities for easy images.
    keep = keep.double()
    other_mass = (-keep.expm1()).clamp_min(1e-20)
    return (false_p1.double().clamp_min(1e-20).log() + other_mass.log()).scatter(
        1, labels, keep
    ).float()


def kl_per_sample(candidate, target):
    return F.kl_div(
        F.log_softmax(candidate / 2, dim=1),
        F.softmax(target / 2, dim=1),
        reduction="none",
    ).sum(dim=1)


@torch.inference_mode()
def main():
    a = args()
    os.environ.update(T3BASE=str(a.base), T3ME=str(a.base))
    sys.path.insert(0, str(a.base / "code"))
    import common

    a.out.mkdir(parents=True)
    device = torch.device(a.capture_device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    model = common.tt.build_model().to(device).eval()
    rng = torch.Generator().manual_seed(90441)
    exact = common.load_forget()
    ids = torch.randperm(len(exact), generator=rng)[:a.images]
    cache = a.base / "cache"
    views = {
        "exact": exact[ids],
        **{n: torch.from_numpy(np.load(cache/f"held_{n}.npy", mmap_mode="r")[ids.numpy()])
           for n in ("jpeg50", "mild", "rot15")},
    }
    original = torch.from_numpy(np.load(cache/"exact_orig_logits.npy"))[ids]
    labels = original.argmax(dim=1, keepdim=True)

    def collect(path):
        model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"), strict=True)
        result = {}
        for view, pixels in views.items():
            batches = []
            for batch in pixels.split(a.batch):
                pixels_float = batch.to(device).permute(0, 3, 1, 2).float() / 127.5 - 1
                with torch.autocast(device.type, dtype=torch.float16 if device.type == "cuda" else torch.bfloat16):
                    batches.append(model(pixels_float).float().cpu())
            result[view] = torch.cat(batches)
        print(f"Captured {path.name}", flush=True)
        return result

    start = time.perf_counter()
    parent = collect(a.parent)
    members = [collect(p) for p in a.members]
    capture_seconds = time.perf_counter() - start
    names = ("RDt24r_70_l1016", "RDt24n_70_l1016", "RD6m2t24_70_l1117")
    targets = {
        n: torch.load(cache/f"tgt_{n}.pt", weights_only=True, map_location="cpu")["z"][ids]
        for n in names if (cache/f"tgt_{n}.pt").is_file()
    }
    if not targets:
        raise SystemExit("No available proxy target")
    for view in views:
        p1 = F.log_softmax(parent[view], dim=1).gather(1, labels)
        print(f"Baseline {view}: CE={-p1.mean().item():.4f}", flush=True)
    rows = []
    splits = {"fit": torch.arange(0, len(ids)//2), "test": torch.arange(len(ids)//2, len(ids))}
    for source in ("uniform", "consensus"):
        for strength in (0.15, 0.35, 0.65, 1.0):
            row = {"source": source, "strength": strength, "fit": {}, "test": {}}
            for view in views:
                false = [suppress_original(F.softmax(x[view]/2, dim=1), labels) for x in members]
                other_mean = torch.stack(false).mean(dim=0)
                weight = strength
                if source == "consensus":
                    winners = torch.stack([p.argmax(1) for p in false])
                    voted = winners.eq(other_mean.argmax(1)).float().mean(0)
                    weight = (strength * voted).unsqueeze(1)
                fused = ce_preserving_mix(parent[view], other_mean, labels, weight)
                ce_delta = (fused.log_softmax(1).gather(1, labels) - parent[view].log_softmax(1).gather(1, labels)).abs().max().item()
                if ce_delta > 3e-5:
                    raise RuntimeError(f"CE invariance failed by {ce_delta:.5g}")
                for section, indices in splits.items():
                    row[section][view] = {
                        n: (kl_per_sample(fused, t) - kl_per_sample(parent[view], t))[indices].mean().item()
                        for n, t in targets.items()
                    }
            rows.append(row)
    # Calibrate on exact/JPEG only. Mild/rot15 stay out of selection.
    best = min(rows, key=lambda r: float(np.mean([x for view in ("exact", "jpeg50") for x in r["fit"][view].values()])))
    summary = {
        "parent": str(a.parent), "members": [str(p) for p in a.members],
        "images": len(ids), "views": list(views), "seconds": capture_seconds,
        "views_per_second": len(ids)*len(views)*(1+len(members))/capture_seconds,
        "peak_reserved_mib": torch.cuda.max_memory_reserved()/2**20 if device.type=="cuda" else None,
        "best": best, "rows": rows,
    }
    (a.out/"screen.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k!="rows"}), flush=True)


if __name__ == "__main__":
    main()
