"""Prepare PixelSeal grid features, then fit a small residual point head on CUDA."""
import argparse
import io
import json
import zipfile

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from wmloc.utils import miou_image

from compare_pixelseal_maps import read_maps


def prepare(train_zip, logits_path, output):
    maps = read_maps(logits_path)
    features, targets, bases, indices, reserve = [], [], [], [], []
    with zipfile.ZipFile(train_zip) as archive:
        meta = json.loads(archive.read("train/metadata.json"))
        with np.load(io.BytesIO(archive.read("train/masks_128.npz"))) as masks:
            truth = dict(zip(masks["names"].tolist(), masks["masks"], strict=True))
        for i, row in enumerate(meta):
            if row["method"] != "pixelseal":
                continue
            name = row["image_name"]
            logits = maps[name].astype(np.float32)
            selected = np.argsort(np.abs(logits.ravel()), kind="stable")[:2048]
            y, x = np.unravel_index(selected, logits.shape)
            patches = np.lib.stride_tricks.sliding_window_view(np.pad(logits, 2, mode="edge"), (5, 5))
            with Image.open(io.BytesIO(archive.read(f"train/images/{name}.png"))) as image:
                gray = np.asarray(image.convert("L"), dtype=np.float32) / 255
            cells = gray.reshape(128, 4, 128, 4).transpose(0, 2, 1, 3)[y, x].reshape(-1, 16)
            design = np.column_stack((logits[y, x], patches[y, x].reshape(-1, 25),
                                      cells - cells.mean(1, keepdims=True), cells.mean(1)))
            features.append(design.astype(np.float16))
            targets.append(truth[name].astype(bool))
            bases.append(logits > 0)
            indices.append(selected.astype(np.int16))
            reserve.append(i % 5 == 0)
    np.savez_compressed(output, features=features, targets=targets, bases=bases,
                        indices=indices, reserve=reserve)


def fit(path):
    torch.manual_seed(20261121)
    with np.load(path) as data:
        features, targets, bases, indices, reserve = (data[k] for k in
                                                     ("features", "targets", "bases", "indices", "reserve"))
    point_labels = np.take_along_axis(targets.reshape(len(targets), -1), indices, axis=1)
    x = torch.from_numpy(features[~reserve]).flatten(0, 1).cuda().float()
    y = torch.from_numpy(point_labels[~reserve]).flatten().cuda().float()
    mean, std = x.mean(0), x.std(0).clamp_min(1e-4)
    original = x[:, 0].clone()
    x = (x - mean) / std
    model = nn.Sequential(nn.Linear(x.shape[1], 64), nn.GELU(), nn.Linear(64, 1)).cuda()
    nn.init.zeros_(model[-1].weight)
    nn.init.zeros_(model[-1].bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    model.train()
    for _ in range(6):
        for batch in torch.randperm(len(x), device="cuda").split(4096):
            pred = original[batch] + model(x[batch]).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(pred, y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    scores = []
    model.eval()
    with torch.inference_mode():
        for design, target, base, selected in zip(features[reserve], targets[reserve], bases[reserve], indices[reserve]):
            raw = torch.from_numpy(design).cuda().float()
            logits = raw[:, 0] + model((raw - mean) / std).squeeze(1)
            corrected = base.copy()
            corrected.flat[selected] = (logits > 0).cpu().numpy()
            scores.append([miou_image(base, target), miou_image(corrected, target)])
    before, after = np.asarray(scores).T
    delta = after - before
    boot = np.random.default_rng(20261121).choice(delta, (20000, len(delta))).mean(axis=1)
    print(f"gpu={torch.cuda.get_device_name(0)} dev={sum(~reserve)} held={sum(reserve)} "
          f"baseline={before.mean():.5f} point_head={after.mean():.5f} delta={delta.mean():+.5f} "
          f"CI95={np.quantile(boot,[.025,.975]).round(5)} peak_MiB={torch.cuda.max_memory_allocated()/2**20:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for arg in ("train_zip", "logits", "features"):
        prep.add_argument(arg)
    sub.add_parser("fit").add_argument("features")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.train_zip, args.logits, args.features)
    else:
        fit(args.features)
