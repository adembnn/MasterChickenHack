"""Generate class-balanced T3 retain anchors from the supplied ViT alone."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--count", type=int, default=800)
    parser.add_argument("--batch", type=int, default=160)
    parser.add_argument("--steps", type=int, default=64)
    args = parser.parse_args()
    args.base = args.base.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    return args


def main():
    args = arguments()
    os.environ.update(T3BASE=str(args.base), T3ME=str(args.base))
    sys.path.insert(0, str(args.base / "code"))
    import common

    args.out.mkdir(parents=True)
    torch.manual_seed(20261122)
    torch.cuda.reset_peak_memory_stats()
    model = common.load_orig().eval().requires_grad_(False)
    started = time.perf_counter()
    all_pixels, all_targets, all_logits = [], [], []
    generator = torch.Generator().manual_seed(2701)
    for labels_cpu in (torch.arange(args.count) % 100).split(args.batch):
        labels = labels_cpu.cuda()
        low_res = torch.nn.Parameter(torch.randn(len(labels), 3, 56, 56, device="cuda") * .16)
        fine = torch.nn.Parameter(torch.randn(len(labels), 3, 224, 224, device="cuda") * .12)
        opt = torch.optim.Adam([{"params": [low_res], "lr": .055}, {"params": [fine], "lr": .016}])
        for step in range(args.steps):
            opt.zero_grad(set_to_none=True)
            x = torch.tanh(
                F.interpolate(low_res, size=(224, 224), mode="bilinear", align_corners=False) + .45 * fine
            )
            jitter = torch.randint(-6, 7, (2,), generator=generator).tolist()
            view = torch.roll(x, shifts=jitter, dims=(2, 3))
            if step % 2:
                view = view.flip(-1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(view).float()
            margins = F.cross_entropy(logits, labels, reduction="none")
            variation = (x[:, :, 1:] - x[:, :, :-1]).abs().mean() + (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
            loss = (margins - .03).clamp_min(0).mean() + .25 * variation + .04 * x.square().mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            final = torch.tanh(
                F.interpolate(low_res, size=(224, 224), mode="bilinear", align_corners=False) + .45 * fine
            )
            pixels = common.to_u8((final + 1) / 2).cpu()
            logits = common.logits_u8(model, pixels, bs=args.batch)
        all_pixels.append(pixels)
        all_targets.append(labels_cpu)
        all_logits.append(logits)
        print(f"Synthesized {len(labels)}, class match {(logits.argmax(1) == labels_cpu).float().mean().item():.3f}", flush=True)
    pixels = torch.cat(all_pixels).numpy()
    logits = torch.cat(all_logits)
    labels = torch.cat(all_targets)
    np.save(args.out / "synthetic.npy", pixels, allow_pickle=False)
    np.save(args.out / "synthetic_orig_logits.npy", logits.numpy(), allow_pickle=False)
    stats = {
        "images": len(labels),
        "class_match": (logits.argmax(1) == labels).float().mean().item(),
        "mean_class_confidence": logits.softmax(1).gather(1, labels[:, None]).mean().item(),
        "seconds": time.perf_counter() - started,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }
    (args.out / "metrics.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
