"""Quel critère épingle le mieux la vraie boîte ? Montée de coordonnées depuis la vraie boîte et depuis une boîte décalée,
pour plusieurs critères tirés des logits du rognage ; on mesure le recouvrement (IoU) final avec la vraie boîte. Rectangles hors réserve."""
import argparse, json, time
import numpy as np, torch
from torchvision.ops import roi_align
from rlib import *
ap = argparse.ArgumentParser(); ap.add_argument("--method", required=True); ap.add_argument("--n", type=int, default=20); ap.add_argument("--fab", action="store_true"); a = ap.parse_args()
dev = "cuda"; wm = load(a.method, dev); RES = {"pixelseal": 256, "trustmark": 245, "stegastamp": 400}[a.method]; rng = np.random.default_rng(0)

@torch.no_grad()
def logits(x, boxes):
    out = []
    for i in range(0, len(boxes), 256):
        b = torch.tensor([(0, bb[1], bb[0], bb[3], bb[2]) for bb in boxes[i:i + 256]], dtype=torch.float32, device=dev)
        cr = roi_align(x, b, output_size=(RES, RES), spatial_scale=1.0, sampling_ratio=2, aligned=True).clamp(0, 1)
        out.append(wm.logits(cr).float().to(dev))
    return torch.cat(out)
OBJ = {
    "z": lambda sl: sl.sum(1) / sl.pow(2).sum(1).sqrt().clamp_min(1e-9),
    "marge": lambda sl: sl.mean(1),
    "marge_bas20": lambda sl: sl.sort(1).values[:, :max(1, sl.shape[1] // 5)].mean(1),
    "logvrais": lambda sl: torch.nn.functional.logsigmoid(sl).mean(1),
    "tanh": lambda sl: torch.tanh(sl / 2).mean(1),
}
def clipbox(b):
    y0, x0, y1, x1 = b; y0 = max(0, min(y0, 500)); x0 = max(0, min(x0, 500)); y1 = max(y0 + 8, min(y1, 512)); x1 = max(x0 + 8, min(x1, 512)); return (y0, x0, y1, x1)
def iou_box(b, g):
    iy = max(0, min(b[2], g[2]) - max(b[0], g[0])); ix = max(0, min(b[3], g[3]) - max(b[1], g[1])); inter = iy * ix
    return inter / ((b[2] - b[0]) * (b[3] - b[1]) + (g[2] - g[0]) * (g[3] - g[1]) - inter)
def moves(b, st):
    c = []
    for k in range(4):
        for d in (-st, st): t = list(b); t[k] += d; c.append(clipbox(tuple(t)))
    for d in (-st, st):
        c.append(clipbox((b[0] + d, b[1], b[2] + d, b[3]))); c.append(clipbox((b[0], b[1] + d, b[2], b[3] + d))); c.append(clipbox((b[0] - d, b[1] - d, b[2] + d, b[3] + d)))
    return [t for t in set(c) if t != b]
def ascent(x, b, s, f, steps=(8, 4, 2, 1)):
    best = float(f(logits(x, [b]) * s)[0])
    for st in steps:
        while True:
            c = moves(b, st); v = f(logits(x, c) * s).cpu().numpy(); j = int(np.argmax(v))
            if v[j] > best + 1e-5: best = float(v[j]); b = c[j]
            else: break
    return b, best
def bbox(mask):
    ys, xs = np.where(mask > 0); return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1
def plateau(x, b, s, f, rounds=2, span=16, frac=0.15):
    """Par bord : on balaie la position du bord, on prend le milieu de la zone où le critère reste près de son maximum."""
    b = list(b)
    for _ in range(rounds):
        for k in range(4):
            ds = list(range(-span, span + 1)); c = []
            for d in ds: t = list(b); t[k] += d; c.append(clipbox(tuple(t)))
            v = f(logits(x, c) * s).cpu().numpy(); j = int(np.argmax(v)); th = v.max() - frac * (v.max() - np.median(v)) - 1e-6
            lo = j
            while lo > 0 and v[lo - 1] >= th: lo -= 1
            hi = j
            while hi < len(v) - 1 and v[hi + 1] >= th: hi += 1
            b = list(c[int(round((lo + hi) / 2))])
    return tuple(b)
def multi(x, b, s, f, n=8, r=6):
    out = []
    for _ in range(n):
        d = rng.integers(-r, r + 1, 4); out.append(ascent(x, clipbox(tuple(int(b[k] + d[k]) for k in range(4))), s, f, steps=(4, 2, 1))[0])
    return tuple(int(v) for v in np.round(np.median(np.array(out), 0)))
items = []
for e in meta("train"):
    if e["method"] != a.method or e.get("full_watermark") or e["gidx"] % 5 == 0: continue
    mk = read_mask(e["image_name"]); g = bbox(mk)
    if mk[g[0]:g[2], g[1]:g[3]].mean() > 0.98 and min(g[2] - g[0], g[3] - g[1]) >= 32: items.append((e, g))
items = items[:a.n]
if a.fab:                                                   # rectangles FABRIQUÉS : même boîte, fond = vraie image d'un autre filigrane, tampon posé sur le rognage
    others = [e for e in meta("train") if e["method"] != a.method and e["gidx"] % 5 != 0]
    fab = {}
    for i, (e, g) in enumerate(items):
        base = read_img("train", others[i * 7 % len(others)]["image_name"]).copy(); m = rng.integers(0, 2, wm.nbits)
        base[g[0]:g[2], g[1]:g[3]] = wm.embed(np.ascontiguousarray(base[g[0]:g[2], g[1]:g[3]]), m); fab[e["image_name"]] = (base, m.astype(np.float32))
EST = ["montée z", "plateau marge 15%"]
res = {k: {"iou": [], "off": []} for k in EST}; t0 = time.time()
for e, g in items:
    img, m = fab[e["image_name"]] if a.fab else (read_img("train", e["image_name"]), bits_of(e))
    x = to_t(img, dev); s = torch.from_numpy(2 * m - 1).float().to(dev)
    d = rng.integers(-8, 9, 4); p = clipbox(tuple(int(g[k] + d[k]) for k in range(4)))
    b0, _ = ascent(x, p, s, OBJ["z"])
    cand = {"montée z": b0, "plateau marge 15%": plateau(x, b0, s, OBJ["marge"])}
    zz = OBJ["z"](logits(x, [g, b0]) * s).cpu().numpy(); res["montée z"].setdefault("zg", []).append(zz[0]); res["montée z"].setdefault("zb", []).append(zz[1])
    for k, b in cand.items(): res[k]["iou"].append(iou_box(b, g)); res[k]["off"].append([b[i] - g[i] for i in range(4)])
print("OBJTEST2", "FABRIQUÉ" if a.fab else "RÉEL", a.method, "n=%d rectangles hors réserve (petit côté >= 32 px), départ = vraie boîte décalée de ±8 px, %d s" % (len(items), time.time() - t0))
print("  estimateur                    IoU moyen   IoU médian   écart moyen par bord (y0,x0,y1,x1)   écart absolu moyen")
for k, v in res.items():
    off = np.array(v["off"]); print("  %-28s  %.3f       %.3f        %-28s   %.1f px" % (k, np.mean(v["iou"]), np.median(v["iou"]), np.round(off.mean(0), 1), np.abs(off).mean()))
print("  z à la vraie boîte : %.2f ; z à la boîte trouvée : %.2f" % (np.mean(res["montée z"]["zg"]), np.mean(res["montée z"]["zb"])))
