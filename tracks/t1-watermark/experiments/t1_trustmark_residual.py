"""Measure whether the TrustMark-Q residual helps localize challenge masks."""

from pathlib import Path
import json

import numpy as np
import torch
import torch.nn.functional as F

from wmloc.data import LocalisationDataset
from wmloc.utils import score
from wmloc.watermarks import load_watermark


def grid(pixels):
    return pixels.reshape(128, 4, 128, 4).mean(axis=(1, 3))


def main():
    torch.set_num_threads(4)
    kit = Path.home() / "eac/kit"
    data = LocalisationDataset(kit / "release", "train", methods=["trustmark"])
    torch.cuda.set_per_process_memory_fraction(0.05)
    detector = load_watermark("trustmark", models_dir=kit / "models", device="cuda").tm
    model = detector.removal.eval()
    residual, canvas, truth = [], [], []
    resolution = detector.model_resolution_remove
    with torch.inference_mode():
        for start in range(0, len(data), 12):
            items = [data[i] for i in range(start, min(start + 12, len(data)))]
            batch = torch.from_numpy(np.stack([item["image"] for item in items]))
            x = batch.permute(0, 3, 1, 2).cuda().float().div_(127.5).sub_(1)
            x = F.interpolate(x, (resolution, resolution), mode="bilinear", align_corners=False, antialias=True)
            difference = (model(x).clamp(-1, 1) - x).abs().mean(1, keepdim=True)
            maps = F.interpolate(difference, (128, 128), mode="area")[:, 0].cpu().numpy()
            for residual_map, item in zip(maps, items):
                low, high = np.quantile(residual_map, [0.10, 0.90])
                residual.append(np.clip((residual_map - low) / max(high - low, 1e-6), 0, 1))
                raw = np.load(kit / "maps_canvas" / (item["image_name"] + ".npy"), allow_pickle=False)
                canvas.append(grid(raw.astype(np.float32)))
                truth.append(item["mask128"])
            if start % 60 == 0:
                print(f"{start + len(items)}/{len(data)} maps", flush=True)
    residual, canvas, truth = map(np.stack, (residual, canvas, truth))
    held = np.arange(len(data)) % 5 == 0
    fit = ~held
    thresholds = np.linspace(.35, .70, 15)

    def select(maps, values):
        _, threshold = max((score(maps[fit] >= t, truth[fit]), t) for t in values)
        return float(threshold), score(maps[held] >= threshold, truth[held])

    raw_threshold, raw_score = select(canvas, thresholds)
    residual_threshold, residual_score = select(residual, np.linspace(.05, .95, 19))
    candidates = []
    for weight in (-.10, -.05, 0., .05, .10):
        fused = canvas + weight * (residual - .5)
        for threshold in thresholds:
            candidates.append((score(fused[fit] >= threshold, truth[fit]), weight, threshold))
    _, weight, threshold = max(candidates)
    fused = canvas + weight * (residual - .5)
    result = {
        "fit": int(fit.sum()), "held": int(held.sum()),
        "canvas_threshold": raw_threshold, "canvas_miou": raw_score,
        "residual_threshold": residual_threshold, "residual_miou": residual_score,
        "fused_weight": weight, "fused_threshold": float(threshold),
        "fused_miou": score(fused[held] >= threshold, truth[held]),
    }
    Path("metrics_trustmark_residual.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
