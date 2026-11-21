"""Test a decoder-guided TrustMark rectangle correction on the labeled split."""

from itertools import product
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn.functional as F

from wmloc.data import LocalisationDataset
from wmloc.utils import miou_image
from wmloc.watermarks import load_watermark


SIDE = 512
GRID = 128


def positions(size, step):
    last = SIDE - size
    return sorted(set(range(0, last + 1, step)) | {last})


def proposal_groups():
    groups = []
    for short, ratio in product((40, 56, 72, 96, 128, 176, 224), (1.0, 1.5, 2.0)):
        long = round(short * ratio / 4) * 4
        step = max(24, round(short / 2 / 4) * 4)
        for w, h in sorted({(long, short), (short, long)}):
            groups.append(np.array([
                (x, y, x+w, y+h) for x, y in product(positions(w, step), positions(h, step))
            ], dtype=np.int32))
    return groups


def shortlist(heat, groups):
    sums = np.pad(heat.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    kept = []
    for boxes in groups:
        x0, y0, x1, y1 = (boxes // 4).T
        mean = (sums[y1, x1] - sums[y0, x1] - sums[y1, x0] + sums[y0, x0]) / ((x1-x0)*(y1-y0))
        selected = np.argsort(mean, kind="stable")[-7:]
        kept.append(boxes[selected])
    return np.unique(np.concatenate(kept), axis=0)


@torch.inference_mode()
def decode(official, image, message, boxes):
    unit = (torch.arange(official.resolution, device=image.device, dtype=torch.float32) + 0.5) / official.resolution
    results = []
    for picks in np.array_split(boxes, max(1, (len(boxes)+23)//24)):
        coords = torch.as_tensor(picks, device=image.device, dtype=torch.float32)
        x0, y0, x1, y1 = coords.unbind(dim=1)
        x = (x0[:, None] + (x1-x0)[:, None] * unit) * (2.0 / SIDE) - 1
        y = (y0[:, None] + (y1-y0)[:, None] * unit) * (2.0 / SIDE) - 1
        xy = torch.stack(torch.broadcast_tensors(x[:, None, :], y[:, :, None]), dim=-1)
        patches = F.grid_sample(image.expand(len(picks), -1, -1, -1), xy, align_corners=False)
        with torch.autocast("cuda", dtype=torch.float16):
            signed = official.logits(patches).float() * message
        agreement = (signed > 0).float().mean(1)
        results.extend(zip(agreement.cpu().tolist(), signed.tanh().mean(1).cpu().tolist()))
    return results


def refine(official, image, message, boxes):
    ranking = sorted(zip(decode(official, image, message, boxes), map(tuple, boxes)), reverse=True)
    seeds = [tuple(int(v) for v in box) for _, box in ranking[:5]]
    for step in (16, 8, 4):
        candidates = set(seeds)
        for box in seeds:
            for edge, direction in product(range(4), (-1, 1)):
                altered = list(box)
                altered[edge] += step * direction
                x0, y0, x1, y1 = altered
                w, h = x1 - x0, y1 - y0
                if 0 <= x0 < x1 <= SIDE and 0 <= y0 < y1 <= SIDE and min(w, h) >= 24 and max(w, h) / min(w, h) <= 2:
                    candidates.add(tuple(altered))
        refined = np.asarray(sorted(candidates))
        ranking = sorted(zip(decode(official, image, message, refined), map(tuple, refined)), reverse=True)
        seeds = [tuple(int(v) for v in box) for _, box in ranking[:5]]
    confidence, _ = ranking[0][0]
    return confidence, tuple(int(v) for v in ranking[0][1])


def main():
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.035)
    kit = Path.home() / "eac/kit"
    data = LocalisationDataset(kit / "release", "train", methods=["trustmark"])
    official = load_watermark("trustmark", kit / "models", "cuda")
    groups = proposal_groups()
    rows = []
    for idx, item in enumerate(data):
        heat = np.load(kit / "maps_canvas" / (item["image_name"] + ".npy"), allow_pickle=False).astype(np.float32)
        heat = heat.reshape(GRID, 4, GRID, 4).mean((1, 3))
        image = torch.from_numpy(item["image"].copy()).cuda().permute(2, 0, 1).unsqueeze(0).float().div_(255)
        bits = torch.as_tensor(item["message"], device=image.device, dtype=torch.float32).mul_(2).sub_(1)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            whole = float((official.logits(image).float() * bits > 0).float().mean())
        truth = item["mask128"]
        baseline = heat >= 0.58
        row = {"name": item["image_name"], "held": idx % 5 == 0, "whole_bits": whole,
               "baseline_miou": miou_image(baseline, truth), "box_bits": 0.0, "box_miou": 0.0}
        if whole < 0.75:
            confidence, box = refine(official, image, bits, shortlist(heat, groups))
            x0, y0, x1, y1 = (v // 4 for v in box)
            mask = np.zeros((GRID, GRID), dtype=np.uint8)
            mask[y0:y1, x0:x1] = 1
            row.update(box_bits=confidence, box_miou=miou_image(mask, truth), box=box)
        rows.append(row)
        if len(rows) % 20 == 0:
            print(f"{len(rows)}/{len(data)} examples", flush=True)

    def evaluate(group, threshold):
        uses = [r["box_bits"] >= threshold for r in group]
        scores = [r["box_miou"] if use else r["baseline_miou"] for use, r in zip(uses, group)]
        accepted = [r["box_miou"] for use, r in zip(uses, group) if use]
        return float(np.mean(scores)), len(accepted), float(np.mean(accepted)) if accepted else None

    fit = [row for row in rows if not row["held"]]
    held = [row for row in rows if row["held"]]
    _, threshold = max((evaluate(fit, t)[0], t) for t in (0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.01))
    miou, replaced, box_miou = evaluate(held, threshold)
    result = {"fit": len(fit), "held": len(held), "threshold": threshold,
              "baseline_miou": float(np.mean([r["baseline_miou"] for r in held])),
              "hybrid_miou": miou, "replaced": replaced, "replaced_box_miou": box_miou}
    Path("trustmark_boxes_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
