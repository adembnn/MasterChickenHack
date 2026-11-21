"""Compare two PixelSeal logit archives on the team split using the starter kit."""

import argparse
import io
import json
import zipfile

import numpy as np
from wmloc.utils import miou_image


def read_maps(path):
    with np.load(path, allow_pickle=False) as archive:
        names, logits = archive["names"].tolist(), archive["logits"]
    if logits.shape != (170, 128, 128) or len(set(names)) != 170 or not np.isfinite(logits).all():
        raise ValueError(f"Invalid PixelSeal logit archive: {path}")
    return dict(zip(names, logits, strict=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_zip")
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    args = parser.parse_args()
    baseline = read_maps(args.baseline)
    candidate = read_maps(args.candidate)
    with zipfile.ZipFile(args.train_zip) as archive:
        metadata = json.loads(archive.read("train/metadata.json"))
        with np.load(io.BytesIO(archive.read("train/masks_128.npz"))) as masks:
            truth = dict(zip(masks["names"].tolist(), masks["masks"], strict=True))

    rows = [(i, row) for i, row in enumerate(metadata) if row["method"] == "pixelseal"]
    if {row["image_name"] for _, row in rows} != baseline.keys() or baseline.keys() != candidate.keys():
        raise ValueError("Predictions and training metadata must cover identical image IDs")
    for held, split_name in [(True, "reserve"), (False, "outside")]:
        scores = []
        for index, row in rows:
            if (index % 5 == 0) != held:
                continue
            name = row["image_name"]
            first, second = baseline[name] > 0, candidate[name] > 0
            scores.append((row["full_watermark"], miou_image(first, truth[name]),
                           miou_image(second, truth[name]), np.any(first != second)))
        for name, group in [("all", scores), ("full", [s for s in scores if s[0]]),
                            ("partial", [s for s in scores if not s[0]])]:
            _, first, second, changed = np.asarray(group).T
            delta = second - first
            samples = np.random.default_rng(20261121).choice(delta, (20000, len(group)))
            lo, hi = np.quantile(samples.mean(axis=1), [0.025, 0.975])
            print(f"{split_name} {name} n={len(group)} base={first.mean():.5f} "
                  f"sam={second.mean():.5f} delta={delta.mean():+.5f} "
                  f"CI95=[{lo:+.5f},{hi:+.5f}] changed={int(changed.sum())} "
                  f"oracle_gain={delta.clip(min=0).mean():.5f}")


if __name__ == "__main__":
    main()
