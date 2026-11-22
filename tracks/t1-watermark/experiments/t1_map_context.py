"""Test whether image-level context improves the T1 canvas-score masks."""

from pathlib import Path
import json

import numpy as np
from scipy.ndimage import uniform_filter
from sklearn.ensemble import HistGradientBoostingClassifier


def miou(predicted, labels):
    predicted = np.asarray(predicted, dtype=bool)
    labels = np.asarray(labels, dtype=bool)
    axes = (1, 2)
    intersections = ((predicted & labels).sum(axis=axes), (~predicted & ~labels).sum(axis=axes))
    unions = ((predicted | labels).sum(axis=axes), (~predicted | ~labels).sum(axis=axes))
    ious = [np.divide(n, d, out=np.ones_like(n, dtype=float), where=d > 0)
            for n, d in zip(intersections, unions)]
    return float(np.mean(ious))


def features(score):
    small = uniform_filter(score, size=5)
    broad = uniform_filter(score, size=25)
    variation = (uniform_filter(score * score, size=15) - uniform_filter(score, size=15) ** 2).clip(0)
    q10, q50, q90 = np.quantile(score, [0.1, 0.5, 0.9])
    stats = [np.full_like(score, x) for x in (q10, q50, q90, score.mean(), score.std())]
    return np.stack((score, small, broad, score - q50, small - broad, variation, *stats), axis=-1)


def main():
    kit = Path.home() / "eac/kit"
    rows = json.loads((kit / "release/train/metadata.json").read_text())
    with np.load(kit / "release/train/masks_128.npz", allow_pickle=False) as source:
        truth = dict(zip(source["names"].tolist(), source["masks"]))
    rng = np.random.default_rng(20261121)
    raw_predictions, learned_predictions, expected = [], [], []
    results = {}
    for method in sorted({row["method"] for row in rows}):
        names = [row["image_name"] for row in rows if row["method"] == method]
        labels = np.stack([truth[name] for name in names])
        maps = np.stack([
            np.load(kit / "maps_canvas" / (name + ".npy"), allow_pickle=False)
            .astype(np.float32).reshape(128, 4, 128, 4).mean(axis=(1, 3))
            for name in names
        ])
        index = np.arange(len(names))
        held = index % 5 == 0
        fit = (~held) & (index % 5 != 1)
        calibration = (~held) & (~fit)
        xs, ys = [], []
        for score_map, mask in zip(maps[fit], labels[fit]):
            sample = rng.choice(mask.size, 384, replace=False)
            xs.append(features(score_map).reshape(mask.size, -1)[sample])
            ys.append(mask.reshape(-1)[sample])
        model = HistGradientBoostingClassifier(
            learning_rate=0.1, max_iter=65, max_leaf_nodes=15, min_samples_leaf=120,
            l2_regularization=2.0, early_stopping=False, random_state=20261121,
        ).fit(np.concatenate(xs), np.concatenate(ys))

        def infer(group):
            return np.stack([
                model.predict_proba(features(heat).reshape(128 * 128, -1))[:, 1].reshape(128, 128)
                for heat in group
            ])

        calibration_maps = infer(maps[calibration])
        _, t = max((miou(calibration_maps >= v, labels[calibration]), v) for v in np.linspace(.3, .7, 17))
        pred = infer(maps[held]) >= t
        _, raw_t = max((miou(maps[~held] >= v, labels[~held]), v) for v in np.arange(.30, .76, .02))
        raw = maps[held] >= raw_t
        result = {"n": int(held.sum()), "canvas_miou": miou(raw, labels[held]),
                  "learned_miou": miou(pred, labels[held]), "threshold": float(t)}
        results[method] = result
        print(method, json.dumps(result), flush=True)
        raw_predictions.extend(raw)
        learned_predictions.extend(pred)
        expected.extend(labels[held])
    results["overall"] = {"canvas_miou": miou(raw_predictions, expected),
                          "learned_miou": miou(learned_predictions, expected)}
    print(json.dumps(results["overall"]), flush=True)
    Path("metrics_map_context.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
