"""T1 · E1 : le stem en bf16 detruit-il le residu fin de PixelSeal ?

Hypothese du critique d'equipe : sous autocast bf16, un residu de 1 a 2 niveaux se perd dans
l'arrondi de la premiere convolution, d'ou l'echec de PixelSeal. Correctif propose : passer le stem en
fp32, ou mieux, donner au reseau une entree passe-haut (image moins son flou, amplifiee), ou le bf16
garde sa precision relative.

Test. On genere des images synthetiques avec de VRAIES images de base et le VRAI embedder : la base
propre est prise dans les zones non filigranees des images de train (donc meme domaine que le test,
images lisses generees, pas des photos COCO), la forme vient des vrais masques du train. Deux bras :

  --stem bf16      : autocast bf16 partout, entree image seule (comme seg2.py)
  --stem fp32      : pas d'autocast, entree image + passe-haut amplifie

Evaluation : mIoU officiel sur la reserve reelle (indice multiple de 5), formes jamais vues.

Usage : python synth_seg.py --method pixelseal --stem bf16 --n 1200 --epochs 8
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from wmloc.data import LocalisationDataset
from wmloc.utils import GRID, miou_image

METHODS = ["pixelseal", "trustmark", "stegastamp"]


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
                                 nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU())

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, cin, base=32):
        super().__init__()
        self.stem = nn.Conv2d(cin, base, 7, padding=3)
        self.d1 = Block(base, base)
        self.d2 = Block(base, base * 2)
        self.d3 = Block(base * 2, base * 4)
        self.d4 = Block(base * 4, base * 8)
        self.u3 = Block(base * 8 + base * 4, base * 4)
        self.u2 = Block(base * 4 + base * 2, base * 2)
        self.u1 = Block(base * 2 + base, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        d1 = self.d1(self.stem(x))
        d2 = self.d2(F.max_pool2d(d1, 2))
        d3 = self.d3(F.max_pool2d(d2, 2))
        d4 = self.d4(F.max_pool2d(d3, 2))
        u3 = self.u3(torch.cat([F.interpolate(d4, size=d3.shape[-2:], mode="nearest"), d3], 1))
        u2 = self.u2(torch.cat([F.interpolate(u3, size=d2.shape[-2:], mode="nearest"), d2], 1))
        u1 = self.u1(torch.cat([F.interpolate(u2, size=d1.shape[-2:], mode="nearest"), d1], 1))
        return self.head(u1)


def highpass(x):
    blur = F.avg_pool2d(x, 5, stride=1, padding=2)
    return (x - blur) * 20.0


def input_channels(img, stem):
    return torch.cat([img, highpass(img)], 0) if stem == "fp32" else img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="release")
    ap.add_argument("--method", default="pixelseal")
    ap.add_argument("--stem", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--size", type=int, default=256)
    a = ap.parse_args()

    torch.manual_seed(0)
    rng = random.Random(0)
    from wmloc.watermarks import load_watermark
    wm = load_watermark(a.method, "models", "cuda")

    ds = LocalisationDataset(a.data, "train", methods=[a.method])
    shapes_part, shapes_hole, bases, messages = [], [], [], []
    for i, m in enumerate(ds.meta):
        it = ds[i]
        mk = it["mask"]
        if m.get("full_watermark"):
            shapes_hole.append(~mk)
        else:
            shapes_part.append(mk)
        if i % 5:                                  # la reserve ne sert pas aux formes
            img = it["image"]
            # on ne garde que des zones vraiment propres : un recadrage dont 95 % des pixels sont hors masque
            for _ in range(20):
                s = 512
                top = rng.randrange(0, img.shape[0] - s + 1)
                left = rng.randrange(0, img.shape[1] - s + 1)
                sub = mk[top:top + s, left:left + s]
                if sub.mean() < 0.05:
                    bases.append(img[top:top + s, left:left + s].copy())
                    break
        messages.append(it["message"])
    print(f"{a.method}: {len(shapes_hole)} formes pleines, {len(shapes_part)} formes partielles, "
          f"{len(bases)} bases propres", flush=True)

    # Le pliage de PixelSeal coute une a deux secondes par image : on filigrane chaque base UNE fois,
    # puis on tire les formes. On garde une correspondance base -> version filigranee pour rester coherent
    # avec le message de cette image.
    embs = []
    for k, base in enumerate(bases):
        msg = messages[k % len(messages)]
        embs.append(wm.embed(base, msg))
        if k % 10 == 0:
            print(f"  filigrane des bases {k}/{len(bases)}", flush=True)

    def make_sample():
        k = rng.randrange(len(bases))
        base, emb = bases[k], embs[k]
        if shapes_hole and rng.random() < 0.5:
            shape = shapes_hole[rng.randrange(len(shapes_hole))]
            img = emb.copy()
            img[shape] = base[shape]
            mask = ~shape
        else:
            shape = shapes_part[rng.randrange(len(shapes_part))]
            img = base.copy()
            img[shape] = emb[shape]
            mask = shape
        return img, mask

    xs, ys = [], []
    for k in range(a.n):
        img, mask = make_sample()
        x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        x = F.interpolate(x[None], size=(a.size, a.size), mode="bilinear", align_corners=False)[0]
        y = torch.from_numpy(mask.astype(np.float32))[None]
        y = F.adaptive_avg_pool2d(y[None], GRID)[0]
        xs.append(input_channels(x, a.stem))
        ys.append(y)
        if k % 200 == 0:
            print(f"  synthetique {k}/{a.n}", flush=True)
    X, Y = torch.stack(xs), torch.stack(ys)

    model = UNet(cin=X.shape[1]).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * ((len(X) + a.batch - 1) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=max(steps, 1))
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).cuda())

    def forward(batch):
        if a.stem == "bf16":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(batch.bfloat16())
            return out.float()
        return model(batch.float())

    for ep in range(a.epochs):
        perm = torch.randperm(len(X))
        tot = 0.0
        for i in range(0, len(X), a.batch):
            idx = perm[i:i + a.batch]
            logits = F.adaptive_avg_pool2d(forward(X[idx].cuda()), GRID)
            loss = lossf(logits, Y[idx].cuda())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(idx)
        print(f"  epoque {ep} perte {tot/len(X):.4f}", flush=True)

    # evaluation sur la reserve reelle
    model.eval()
    scores, n_res = [], 0
    for i, m in enumerate(ds.meta):
        it = ds[i]
        out = []
        x = torch.from_numpy(it["image"]).permute(2, 0, 1).float() / 255.0
        x = F.interpolate(x[None], size=(a.size, a.size), mode="bilinear", align_corners=False)[0]
        with torch.no_grad():
            p = torch.sigmoid(forward(input_channels(x, a.stem)[None].cuda()))
        pred = (F.adaptive_avg_pool2d(p, GRID)[0, 0] > 0.5).cpu().numpy().astype(np.uint8)
        if i % 5 == 0:
            scores.append(miou_image(pred, it["mask128"]))
            n_res += 1
    res = {"method": a.method, "stem": a.stem, "n_synth": a.n, "epochs": a.epochs,
           "reserve_miou": round(float(np.mean(scores)), 4), "n_reserve": n_res}
    print(json.dumps(res))
    Path(f"synth_{a.method}_{a.stem}.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
