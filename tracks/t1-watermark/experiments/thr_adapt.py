"""T1, seuils adaptatifs par image : le seuil global est-il le bon choix ?

Le seuil calibre est global par methode (0,56 a 0,58). Mais la carte d'accord n'a pas la meme
distribution selon l'image : une image dont le filigrane couvre tout le cadre vit autour de 0,9, une
greffe d'objet donne une carte presque plate a 0,5. Un seuil fixe traite les deux pareil.

On teste des regles qui deduisent le seuil de la carte elle-meme : quantile, moyenne plus k ecart-type,
et une variante qui force un contraste minimum. Tout se calcule sur les cartes deja sauvegardees,
sans GPU.

Usage : python thr_adapt.py --data release --maps maps_canvas
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from wmloc.data import LocalisationDataset
from wmloc.utils import GRID, miou_image


def to_grid(acc, thr):
    t = torch.from_numpy(acc)[None, None].float()
    g = F.adaptive_avg_pool2d(t, GRID)[0, 0].numpy()
    return (g >= thr).astype(np.uint8)


def rule(grid, kind, param):
    if kind == "global":
        return (grid >= param).astype(np.uint8)
    if kind == "quantile":
        return (grid >= np.quantile(grid, param)).astype(np.uint8)
    if kind == "mean_std":
        return (grid >= float(grid.mean() + param * grid.std())).astype(np.uint8)
    if kind == "otsu_like":
        # seuil au milieu des deux modes estimes par les quantiles 20 et 80
        lo, hi = np.quantile(grid, 0.2), np.quantile(grid, 0.8)
        return (grid >= 0.5 * (lo + hi)).astype(np.uint8)
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="release")
    ap.add_argument("--maps", default="maps_canvas")
    ap.add_argument("--calib", default="calib_final.json")
    a = ap.parse_args()

    calib = json.loads(Path(a.calib).read_text()) if Path(a.calib).exists() else {}
    ds = LocalisationDataset(a.data, "train")
    grids, gts, methods = [], [], []
    for it in ds:
        mp = Path(a.maps) / f"{it['image_name']}.npy"
        if not mp.exists():
            continue
        acc = np.load(mp).astype(np.float32)
        thr = calib.get(it["method"], {}).get("thr", 0.5)
        t = torch.from_numpy(acc)[None, None].float()
        grids.append(F.adaptive_avg_pool2d(t, GRID)[0, 0].numpy())
        gts.append(it["mask128"])
        methods.append(it["method"])
    print(len(grids), "images")

    results = {}
    # reference : seuil global calibre
    preds = [to_grid_from(g, calib.get(m, {}).get("thr", 0.5)) for g, m in zip(grids, methods)]
    results["seuil global calibre"] = round(float(np.mean([miou_image(p, g) for p, g in zip(preds, gts)])), 4)

    for q in (0.4, 0.5, 0.6, 0.7):
        preds = [rule(g, "quantile", q) for g in grids]
        results[f"quantile {q}"] = round(float(np.mean([miou_image(p, g) for p, g in zip(preds, gts)])), 4)
    for k in (-0.5, 0.0, 0.5, 1.0):
        preds = [rule(g, "mean_std", k) for g in grids]
        results[f"moyenne + {k} ecart-type"] = round(float(np.mean([miou_image(p, g) for p, g in zip(preds, gts)])), 4)
    preds = [rule(g, "otsu_like", 0) for g in grids]
    results["milieu des modes 20/80"] = round(float(np.mean([miou_image(p, g) for p, g in zip(preds, gts)])), 4)

    for k, v in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {k:28} {v}")


def to_grid_from(grid, thr):
    return (grid >= thr).astype(np.uint8)


if __name__ == "__main__":
    main()
