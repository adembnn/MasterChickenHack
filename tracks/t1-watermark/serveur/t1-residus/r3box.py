"""Critère « re-tamponnage » dans le cadre de la boîte : R3 = énergie de ce que le tampon ajouterait au rognage avec le VRAI message,
divisée par la même énergie avec des messages au hasard. Si le filigrane est déjà là et bien cadré, le tampon ajoute moins (R3 < 1).
Question : R3 est-il plus pointu que le décodeur autour de la vraie boîte ? Réel et fabriqué, rectangles hors réserve."""
import argparse, time
import numpy as np, torch
from rlib import *
ap = argparse.ArgumentParser(); ap.add_argument("--method", required=True); ap.add_argument("--n", type=int, default=16); ap.add_argument("--K", type=int, default=6); a = ap.parse_args()
dev = "cuda"; wm = load(a.method, dev); rng = np.random.default_rng(0)
def bbox(mask):
    ys, xs = np.where(mask > 0); return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1
def cb(b):
    y0, x0, y1, x1 = b; y0 = max(0, min(y0, 500)); x0 = max(0, min(x0, 500)); return (y0, x0, max(y0 + 8, min(y1, 512)), max(x0 + 8, min(x1, 512)))
def r3(img, b, m, mf):
    T = tmpl(wm, np.ascontiguousarray(img[b[0]:b[2], b[1]:b[3]]), np.concatenate([m[None], mf])); en = T.pow(2).sum((1, 2, 3))
    return float(en[0] / en[1:].mean())
P = {"exact": [(0, 0, 0, 0)]}
for d in (1, 2, 4, 8): P["décalage %d px" % d] = [(d, d, d, d), (-d, -d, -d, -d), (d, 0, d, 0), (0, d, 0, d)]
for d in (1, 2, 4, 8): P["plus grand %d px/bord" % d] = [(-d, -d, d, d)]
for d in (1, 2, 4, 8): P["plus petit %d px/bord" % d] = [(d, d, -d, -d)]
items = []
for e in meta("train"):
    if e["method"] != a.method or e.get("full_watermark") or e["gidx"] % 5 == 0: continue
    mk = read_mask(e["image_name"]); g = bbox(mk)
    if mk[g[0]:g[2], g[1]:g[3]].mean() > 0.98 and min(g[2] - g[0], g[3] - g[1]) >= 40: items.append((e, g))
items = items[:a.n]; others = [e for e in meta("train") if e["method"] != a.method and e["gidx"] % 5 != 0]
acc = {"réel": {k: [] for k in P}, "fabriqué": {k: [] for k in P}}; t0 = time.time()
for i, (e, g) in enumerate(items):
    real = read_img("train", e["image_name"]); m = bits_of(e); mf = rng.integers(0, 2, (a.K, wm.nbits)).astype(np.float32)
    fab = read_img("train", others[i * 7 % len(others)]["image_name"]).copy(); m2 = rng.integers(0, 2, wm.nbits).astype(np.float32)
    fab[g[0]:g[2], g[1]:g[3]] = wm.embed(np.ascontiguousarray(fab[g[0]:g[2], g[1]:g[3]]), m2)
    for nm, img, mm in (("réel", real, m), ("fabriqué", fab, m2)):
        for k, ds in P.items(): acc[nm][k].append(np.mean([r3(img, cb((g[0] + d[0], g[1] + d[1], g[2] + d[2], g[3] + d[3])), mm, mf) for d in ds]))
print("R3BOX", a.method, "n=%d rectangles (petit côté >= 40 px), %d s. R3 moyen (1,0 = le tampon ne reconnaît rien ; plus bas = filigrane reconnu)" % (len(items), time.time() - t0))
print("  perturbation            réel     fabriqué")
for k in P: print("  %-22s  %.3f    %.3f" % (k, np.mean(acc["réel"][k]), np.mean(acc["fabriqué"][k])))
