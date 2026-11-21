"""CPU pilot for a PixelSeal point head. Inputs: official train zip and one out-of-fold logit archive."""
import argparse
import io
import json
import zipfile

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from wmloc.utils import miou_image

from compare_pixelseal_maps import read_maps


def features(image, logits, selected):
    luminance = np.asarray(image.convert("L"), dtype=np.float32) / 255
    cells = luminance.reshape(128, 4, 128, 4)
    mean = cells.mean(axis=(1, 3))
    texture = cells.std(axis=(1, 3))
    channels = [logits, uniform_filter(logits, 3), uniform_filter(logits, 9),
                mean, texture, mean - uniform_filter(mean, 5)]
    return np.column_stack([channel.ravel()[selected] for channel in channels])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_zip")
    parser.add_argument("pixelseal_logits")
    args = parser.parse_args()
    maps = read_maps(args.pixelseal_logits)
    train_x, train_y, held = [], [], []
    with zipfile.ZipFile(args.train_zip) as archive:
        metadata = json.loads(archive.read("train/metadata.json"))
        with np.load(io.BytesIO(archive.read("train/masks_128.npz"))) as masks:
            truth = dict(zip(masks["names"].tolist(), masks["masks"], strict=True))
        for index, row in enumerate(metadata):
            if row["method"] != "pixelseal":
                continue
            name = row["image_name"]
            logits = maps[name].astype(np.float32)
            target = truth[name].astype(bool)
            selected = np.argsort(np.abs(logits.ravel()), kind="stable")[:logits.size // 8]
            with Image.open(io.BytesIO(archive.read(f"train/images/{name}.png"))) as image:
                design = features(image, logits, selected)
            if index % 5:
                train_x.append(design)
                train_y.append(target.ravel()[selected])
            else:
                held.append((design, logits > 0, target, selected))

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=250))
    model.fit(np.concatenate(train_x), np.concatenate(train_y))
    rng = np.random.default_rng(20261121)
    scores = []
    for design, base, target, selected in held:
        corrected, oracle, random_oracle = (base.copy() for _ in range(3))
        corrected.flat[selected] = model.predict(design)
        oracle.flat[selected] = target.flat[selected]
        random_cells = rng.choice(base.size, len(selected), replace=False)
        random_oracle.flat[random_cells] = target.flat[random_cells]
        scores.append([miou_image(pred, target) for pred in (base, corrected, oracle, random_oracle)])
    baseline, linear, ceiling, random_ceiling = np.asarray(scores).T
    delta = linear - baseline
    boot = rng.choice(delta, (20000, len(delta))).mean(axis=1)
    print(f"dev_images={len(train_x)} held_images={len(held)} base={baseline.mean():.5f} "
          f"linear={linear.mean():.5f} uncertain_oracle={ceiling.mean():.5f} "
          f"random_oracle={random_ceiling.mean():.5f} paired_delta={delta.mean():+.5f} "
          f"CI95={np.quantile(boot, [.025, .975]).round(5)}")


if __name__ == "__main__":
    main()
