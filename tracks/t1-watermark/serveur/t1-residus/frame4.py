"""Balayage LARGE du cadre (décalage jusqu'à ±96 px, échelle 0,8 à 1,3) avec le critère « re-tamponnage » (R3) mesuré dans la zone filigranée.
Pour les partielles « objet » et les pleines. Chemin exact du tampon ; l'image est prolongée par recopie du bord."""
import argparse, time
import numpy as np, torch
from rlib import *
ap = argparse.ArgumentParser(); ap.add_argument("--method", default="trustmark"); ap.add_argument("--n", type=int, default=3); ap.add_argument("--K", type=int, default=2)
ap.add_argument("--pop", default="objets"); ap.add_argument("--rad", type=int, default=96); ap.add_argument("--step", type=int, default=8); a = ap.parse_args()
dev = "cuda"; wm = load(a.method, dev); rng = np.random.default_rng(0); PAD = 224
def r3(imgp, b, msgs, mask):
    b = tuple(int(round(v)) for v in b); crop = np.ascontiguousarray(imgp[b[0] + PAD:b[2] + PAD, b[1] + PAD:b[3] + PAD]); en = tmpl(wm, crop, msgs).pow(2).sum(1)
    y0, x0 = max(0, b[0]), max(0, b[1]); y1, x1 = min(512, b[2]), min(512, b[3])
    if y1 <= y0 or x1 <= x0: return 9.0
    en = en[:, y0 - b[0]:y1 - b[0], x0 - b[1]:x1 - b[1]] * torch.from_numpy(mask[y0:y1, x0:x1]).to(dev).float(); e = en.sum((1, 2)); return float(e[0] / (e[1:].mean() + 1e-20))
tr = meta("train"); M_ = []
for e in tr:
    if e["method"] != a.method or e["gidx"] % 5 == 0: continue
    if a.pop == "pleines":
        if e.get("full_watermark"): M_.append(e)
    elif not e.get("full_watermark"):
        mk = read_mask(e["image_name"]); ys, xs = np.where(mk)
        if mk[ys.min():ys.max() + 1, xs.min():xs.max() + 1].mean() <= 0.98 and mk.mean() > 0.03: M_.append(e)
M_ = M_[:a.n]; print("FRAME4", a.method, a.pop, "n=%d ; R3 dans la zone filigranée (1,0 = rien reconnu ; fabriqué bien cadré : 0,3)" % len(M_), flush=True); t0 = time.time()
for i, e in enumerate(M_):
    img = read_img("train", e["image_name"]); m = bits_of(e); mk = read_mask(e["image_name"]); imgp = np.pad(img, ((PAD, PAD), (PAD, PAD), (0, 0)), mode="edge")
    msgs = np.concatenate([m[None], rng.integers(0, 2, (a.K, wm.nbits)).astype(np.float32)])
    offs = range(-a.rad, a.rad + 1, a.step); V = {(dy, dx): r3(imgp, (dy, dx, 512 + dy, 512 + dx), msgs, mk) for dy in offs for dx in offs}
    k = min(V, key=V.get); best = (k[0], k[1], 1.0, V[k]); vals = np.array(list(V.values()))
    for s in (0.8, 0.85, 0.9, 0.95, 1.05, 1.1, 1.15, 1.2, 1.3):                         # échelle autour du centre de l'image, puis au meilleur décalage
        for (dy, dx) in ((0, 0), k):
            h = 512 * s; v = r3(imgp, (256 - h / 2 + dy, 256 - h / 2 + dx, 256 + h / 2 + dy, 256 + h / 2 + dx), msgs, mk)
            if v < best[3]: best = (dy, dx, s, v)
    b = best
    for st in (4, 2, 1):                                                               # affinage décalage + taille
        improved = True
        while improved:
            improved = False
            for ddy, ddx, dh in ((st, 0, 0), (-st, 0, 0), (0, st, 0), (0, -st, 0), (0, 0, st), (0, 0, -st)):
                dy, dx, s = b[0] + ddy, b[1] + ddx, b[2] + dh * 2 / 512; h = 512 * s
                v = r3(imgp, (256 - h / 2 + dy, 256 - h / 2 + dx, 256 + h / 2 + dy, 256 + h / 2 + dx), msgs, mk)
                if v < b[3] - 1e-4: b = (dy, dx, s, v); improved = True
    print("  image %d part filigranée %.2f | R3 cadre image entière %.2f | grille de décalages : min %.2f en (dy,dx)=%s, médiane %.2f | après affinage : décalage (%d,%d) taille %d px R3 %.2f | %d s" % (
        i, mk.mean(), V[(0, 0)], vals.min(), k, np.median(vals), b[0], b[1], round(512 * b[2]), b[3], time.time() - t0), flush=True)
