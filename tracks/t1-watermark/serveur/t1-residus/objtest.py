"""Quel critère épingle le mieux la vraie boîte ? Montée de coordonnées depuis la vraie boîte et depuis une boîte décalée,
pour plusieurs critères tirés des logits du rognage ; on mesure le recouvrement (IoU) final avec la vraie boîte. Rectangles hors réserve."""
import argparse, json, time
import numpy as np, torch
from torchvision.ops import roi_align
from rlib import *
ap = argparse.ArgumentParser(); ap.add_argument("--method", required=True); ap.add_argument("--n", type=int, default=20); a = ap.parse_args()
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
items = []
for e in meta("train"):
    if e["method"] != a.method or e.get("full_watermark") or e["gidx"] % 5 == 0: continue
    mk = read_mask(e["image_name"]); g = bbox(mk)
    if mk[g[0]:g[2], g[1]:g[3]].mean() > 0.98 and min(g[2] - g[0], g[3] - g[1]) >= 32: items.append((e, g))
items = items[:a.n]; res = {k: {"gt": [], "pert": [], "off": []} for k in OBJ}; t0 = time.time()
for e, g in items:
    x = to_t(read_img("train", e["image_name"]), dev); s = torch.from_numpy(2 * bits_of(e) - 1).float().to(dev)
    d = rng.integers(-8, 9, 4); p = clipbox(tuple(int(g[k] + d[k]) for k in range(4)))
    for k, f in OBJ.items():
        b1, _ = ascent(x, g, s, f, steps=(4, 2, 1)); b2, _ = ascent(x, p, s, f)
        res[k]["gt"].append(iou_box(b1, g)); res[k]["pert"].append(iou_box(b2, g)); res[k]["off"].append([b2[i] - g[i] for i in range(4)])
print("OBJTEST", a.method, "n=%d rectangles hors réserve (petit côté >= 32 px), %d s" % (len(items), time.time() - t0))
print("  critère        IoU final (départ = vraie boîte)   IoU final (départ = boîte décalée de ±8 px)   écart moyen par bord (y0,x0,y1,x1)   écart absolu moyen")
for k, v in res.items():
    off = np.array(v["off"]); print("  %-13s  %.3f                              %.3f                                         %s   %.1f px" % (k, np.mean(v["gt"]), np.mean(v["pert"]), np.round(off.mean(0), 1), np.abs(off).mean()))
