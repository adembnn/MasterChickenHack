"""T1, segmentateur sur l'image seule : les retouches laissent des traces visibles.

Hypothese. Le filigrane lui-meme est invisible, mais les retouches qui le revelent ne le sont pas :
recadrage d'une greffe, raccord de flou, difference de netteté sur la zone restauree, bord du
rectangle d'outpainting. Un reseau qui regarde l'image, et pas la carte de decodage, peut apprendre
ces artefacts. C'est une source d'information que notre decodage ignore completement.

Test rapide : un pli de validation, image 256 en entree, masque 128 en sortie. Si le mIoU depasse
largement 0,46 (decodage seul), on deroule la validation croisee complete et on entraine sur tout.

Usage : python seg_image.py --data release --fold 0 --folds 5 --epochs 30
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from wmloc.data import LocalisationDataset
from wmloc.utils import GRID, miou_image


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, cin=3, base=32):
        super().__init__()
        self.d1 = Block(cin, base)
        self.d2 = Block(base, base * 2)
        self.d3 = Block(base * 2, base * 4)
        self.d4 = Block(base * 4, base * 8)
        self.u3 = Block(base * 8 + base * 4, base * 4)
        self.u2 = Block(base * 4 + base * 2, base * 2)
        self.u1 = Block(base * 2 + base, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(F.max_pool2d(d1, 2))
        d3 = self.d3(F.max_pool2d(d2, 2))
        d4 = self.d4(F.max_pool2d(d3, 2))
        u3 = self.u3(torch.cat([F.interpolate(d4, size=d3.shape[-2:], mode="nearest"), d3], 1))
        u2 = self.u2(torch.cat([F.interpolate(u3, size=d2.shape[-2:], mode="nearest"), d2], 1))
        u1 = self.u1(torch.cat([F.interpolate(u2, size=d1.shape[-2:], mode="nearest"), d1], 1))
        return self.head(u1)


def load(data_dir, size=256):
    ds = LocalisationDataset(data_dir, "train")
    xs, ys = [], []
    for it in ds:
        img = torch.from_numpy(it["image"]).permute(2, 0, 1).float() / 255.0
        xs.append(F.interpolate(img[None], size=(size, size), mode="bilinear", align_corners=False)[0])
        ys.append(torch.from_numpy(it["mask128"]).float()[None])
    return torch.stack(xs), torch.stack(ys)


def augment(x, y, gen):
    if torch.rand(1, generator=gen).item() < 0.5:
        x, y = torch.flip(x, [-1]), torch.flip(y, [-1])
    if torch.rand(1, generator=gen).item() < 0.5:
        x, y = torch.flip(x, [-2]), torch.flip(y, [-2])
    if torch.rand(1, generator=gen).item() < 0.3:
        k = torch.randint(0, 4, (1,), generator=gen).item()
        x, y = torch.rot90(x, k, [-2, -1]), torch.rot90(y, k, [-2, -1])
    return x, y


def train(x, y, epochs, seed=0, batch=8, lr=1e-3):
    torch.manual_seed(seed)
    model = UNet().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * ((len(x) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(steps, 1))
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).cuda())
    gen = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        perm = torch.randperm(len(x), generator=gen)
        tot = 0.0
        for i in range(0, len(x), batch):
            idx = perm[i:i + batch]
            xb, yb = [], []
            for j in idx.tolist():
                a, b = augment(x[j], y[j], gen)
                xb.append(a)
                yb.append(b)
            xb = torch.stack(xb).cuda()
            yb = torch.stack(yb).cuda()
            logits = F.adaptive_avg_pool2d(model(xb), GRID)
            loss = lossf(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(idx)
        if ep % 10 == 0:
            print(f"  epoque {ep} perte {tot/len(x):.4f}", flush=True)
    return model


@torch.no_grad()
def predict(model, x, batch=32):
    model.eval()
    out = []
    for i in range(0, len(x), batch):
        out.append(torch.sigmoid(F.adaptive_avg_pool2d(model(x[i:i + batch].cuda()), GRID)).cpu())
    return torch.cat(out)


def miou(pred, gt):
    return float(np.mean([miou_image((p[0] > 0.5).numpy().astype(np.uint8), g[0].numpy().astype(np.uint8))
                          for p, g in zip(pred, gt)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="release")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--size", type=int, default=256)
    a = ap.parse_args()

    x, y = load(a.data, a.size)
    print(f"{len(x)} images, entree {tuple(x.shape[1:])}", flush=True)
    gen = torch.Generator().manual_seed(1234)
    idx = torch.randperm(len(x), generator=gen)
    folds = torch.chunk(idx, a.folds)
    val = folds[a.fold]
    tr = torch.cat([f for j, f in enumerate(folds) if j != a.fold])

    model = train(x[tr], y[tr], a.epochs)
    p = predict(model, x[val])
    print(json.dumps({"fold": a.fold, "val_miou": round(miou(p, y[val]), 4), "n": len(val)}))
    torch.save(model.state_dict(), f"seg_image_fold{a.fold}.pt")


if __name__ == "__main__":
    main()
