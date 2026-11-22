"""Images pleines : les trous (zones propres) rendus sont-ils systematiquement trop petits ou mal bordes ? (22/11, apres pires.py)
Test : sur les images rendues pleines (plus de la moitie des cases filigranees), on transforme la zone propre du candidat : dilatation de r cases
(r = -1 : erosion), avec ou sans enveloppe convexe de chaque composante. r et l'option sont choisis sur les 4 plis d'apprentissage (masques de
validation croisee du candidat), le 5e pli est tenu a l'ecart. Usage : python morph.py 38"""
import sys, numpy as np
from scipy import ndimage as ndi
from skimage.morphology import convex_hull_image, disk
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); from t1lib import *
n = sys.argv[1]; rng = np.random.default_rng(0); OPS = [(r, h) for r in (-1, 0, 1, 2, 3, 4) for h in (False, True)]; D = {}
def op(M, r, hull):
    out = M.copy()
    for i in np.where(M.reshape(len(M), -1).mean(1) > 0.5)[0]:
        c = ~M[i]
        if not c.any(): continue
        if hull:
            lab, k = ndi.label(c); h = np.zeros_like(c)
            for j in range(1, k + 1): h |= convex_hull_image(lab == j)
            c = h
        if r > 0: c = ndi.binary_dilation(c, structure=disk(r))
        elif r < 0: c = ndi.binary_erosion(c, structure=disk(-r))
        out[i] = ~c
    return out
for m in METHODS:
    rows = {r["image_name"]: r for r in load_train(m)}; z = np.load(ME / f"candidat_{n}_cv_{m}.npz"); names = z["names"].tolist(); M = z["masks"]; fold = z["fold"]; G = np.stack([rows[k]["gt"] for k in names]); res = np.array([rows[k]["reserve"] for k in names])
    base = miou(M, G); SC = {o: miou(op(M, *o), G) for o in OPS}; new = base.copy(); ch = {}
    for f in range(5):
        fit = fold != f; best = max(OPS, key=lambda o: SC[o][fit].mean()); ch[f] = best; new[fold == f] = SC[best][fold == f]
    d = new - base; D[m] = d; boot = [rng.choice(d, len(d)).mean() for _ in range(4000)]
    print(f"{m:10s} candidat {n} {base.mean():.4f} -> {new.mean():.4f} | ecart {d.mean():+.4f} IC95 [{np.quantile(boot, .025):+.4f} ; {np.quantile(boot, .975):+.4f}] ({int((d > 1e-6).sum())} mieux, {int((d < -1e-6).sum())} pires) | reserve {d[res].mean():+.4f} | choix par pli {ch}")
    print("      toutes images, par operation (dilatation, enveloppe) : " + " ; ".join(f"{o[0]}{'h' if o[1] else ''} {SC[o].mean() - base.mean():+.4f}" for o in OPS))
boot = [np.mean([rng.choice(D[m], len(D[m])).mean() for m in METHODS]) for _ in range(4000)]
print(f"moyenne des trois filigranes : {np.mean([D[m].mean() for m in METHODS]):+.4f}, IC95 [{np.quantile(boot, .025):+.4f} ; {np.quantile(boot, .975):+.4f}]")
