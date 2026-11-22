"""Où reste la perte d'une carte ? Par population et par taille de la zone rare (trou des pleines, zone filigranée des partielles), images hors réserve prédites hors pli.
« perte » = somme de (1 - mIoU) sur les images de la ligne ; « part » = sa part de la perte totale du filigrane. usage : pertes.py <filigrane> <dossier>"""
import sys, json, numpy as np
from pathlib import Path
import ens
me, d = sys.argv[1], sys.argv[2]; names, L = ens.load(d, me, "train"); rows = []
for n, l in zip(names, L):
    if ens.pos[n] % 5 == 0: continue
    y = ens.Y[n]; fu = ens.full[n]; rare = (1 - y.mean()) if fu else y.mean(); ys, xs = np.where(y)
    rect = (not fu) and y.any() and y.sum() >= 0.9 * (ys.max() + 1 - ys.min()) * (xs.max() + 1 - xs.min())
    rows.append(("pleine" if fu else "rectangle" if rect else "objet", rare, ens.miou(l > 0, y), ((l > 0) != (l > 0).flat[0]).any()))
tot = sum(1 - r[2] for r in rows); print(f"{me} {d} : {len(rows)} images hors réserve, mIoU {np.mean([r[2] for r in rows]):.3f}")
for pop in ("pleine", "rectangle", "objet"):
    for lo, hi in ((0, .01), (.01, .03), (.03, .08), (.08, .2), (.2, 1.01)):
        v = [r for r in rows if r[0] == pop and lo <= r[1] < hi]
        if v: print(f"   {pop:9s} zone rare {lo*100:4.1f} à {hi*100:5.1f} % : {len(v):3d} images | mIoU {np.mean([r[2] for r in v]):.3f} | perte {sum(1 - r[2] for r in v):5.1f} | part {sum(1 - r[2] for r in v) / tot:.2f} | cartes d'une seule classe {np.mean([not r[3] for r in v]):.2f}")
