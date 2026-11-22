"""Étape 3 : recherche de boîte pour les images partielles « rectangle » (le tampon est calé sur le rectangle).
Partie A (netteté) : le score chute-t-il vite quand on déplace / redimensionne la boîte vraie ?
  zdec = z exact des logits du décodeur sur le rognage ; zmf = filtre adapté (résidu du rognage contre gabarit du rognage, écart à K faux messages).
Partie B (recherche) : grille grossière au décodeur, puis montée de coordonnées (zdec, puis zmf si --refine mf). IoU de la boîte trouvée, mIoU officiel."""
import argparse, json, time
import numpy as np, torch, torch.nn.functional as F
from rlib import *

ap = argparse.ArgumentParser(); ap.add_argument("--method", required=True); ap.add_argument("--n", type=int, default=0)
ap.add_argument("--part", default="AB"); ap.add_argument("--K", type=int, default=6); ap.add_argument("--den", default="g2"); a = ap.parse_args()
dev = "cuda"; wm = load(a.method, dev); rng = np.random.default_rng(0); RES = {"pixelseal": 256, "trustmark": 245, "stegastamp": 400}[a.method]
MAXAR = 2.0 if a.method == "trustmark" else 16.0

def bbox(mask):
    ys, xs = np.where(mask > 0); return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1

@torch.no_grad()
def zdec(x, boxes, m, bs=64):
    """z exact du décodeur pour chaque boîte (y0,x0,y1,x1) : somme((2m-1).l)/racine(somme(l^2)). Retour z, bits justes."""
    s = torch.from_numpy(2 * m - 1).float(); zs, accs = [], []
    for i in range(0, len(boxes), bs):
        mode = "bicubic" if a.method == "stegastamp" else "bilinear"
        cr = torch.cat([F.interpolate(x[:, :, y0:y1, x0:x1], size=(RES, RES), mode=mode, antialias=True, align_corners=False) for y0, x0, y1, x1 in boxes[i:i + bs]]).clamp(0, 1)
        lg = wm.logits(cr).cpu()
        zs.append((lg * s).sum(1) / lg.pow(2).sum(1).sqrt().clamp_min(1e-9)); accs.append(((lg > 0).float() * 2 - 1 == s).float().mean(1))
    return torch.cat(zs).numpy(), torch.cat(accs).numpy()

@torch.no_grad()
def zmf(img, x, box, m, msgs_f):
    """Filtre adapté dans le cadre de la boîte : corrélation normalisée entre le résidu du rognage et le gabarit du rognage, en écarts-types des faux messages."""
    y0, x0, y1, x1 = box; crop = np.ascontiguousarray(img[y0:y1, x0:x1])
    T = tmpl(wm, crop, np.concatenate([m[None], msgs_f])); xc = x[:, :, y0:y1, x0:x1]
    h = hp(a.den, xc); E = hp(a.den, T)
    rho = ((h * E).sum((1, 2, 3)) / ((E * E).sum((1, 2, 3)) * (h * h).sum()).sqrt().clamp_min(1e-20)).cpu().numpy()
    return float((rho[0] - rho[1:].mean()) / (rho[1:].std() + 1e-12)), float(rho[0])

def clipbox(b):
    y0, x0, y1, x1 = b; y0 = max(0, min(y0, 504)); x0 = max(0, min(x0, 504)); y1 = max(y0 + 8, min(y1, 512)); x1 = max(x0 + 8, min(x1, 512)); return (y0, x0, y1, x1)

def iou_box(b, g):
    iy = max(0, min(b[2], g[2]) - max(b[0], g[0])); ix = max(0, min(b[3], g[3]) - max(b[1], g[1])); inter = iy * ix
    return inter / ((b[2] - b[0]) * (b[3] - b[1]) + (g[2] - g[0]) * (g[3] - g[1]) - inter)

def box_mask128(b):
    mk = np.zeros((512, 512), np.uint8); mk[b[0]:b[2], b[1]:b[3]] = 1; return downsample_mask(mk)

def coarse_grid():
    sizes = [32, 48, 64, 96, 128, 176, 240, 320]; out = []
    for h in sizes:
        for w in sizes:
            if max(h, w) / min(h, w) > min(MAXAR, 4.0): continue
            st = max(8, min(h, w) // 3)
            for y in list(range(0, 512 - h + 1, st)):
                for x_ in list(range(0, 512 - w + 1, st)): out.append((y, x_, y + h, x_ + w))
    return out
GRID = coarse_grid()

def ascent(x, b, m, steps=(16, 8, 4, 2, 1), score=None):
    """Montée de coordonnées : on bouge un bord à la fois, on garde si le score monte."""
    best = score([b])[0]; n_eval = 1
    for st in steps:
        improved = True
        while improved:
            improved = False
            cands = []
            for k in range(4):
                for d in (-st, st):
                    c = list(b); c[k] += d; c = clipbox(tuple(c))
                    if c != b and (c[2] - c[0]) >= 12 and (c[3] - c[1]) >= 12: cands.append(c)
            for d in (-st, st):                                    # déplacements d'ensemble
                cands.append(clipbox((b[0] + d, b[1], b[2] + d, b[3]))); cands.append(clipbox((b[0], b[1] + d, b[2], b[3] + d)))
            sc = score(cands); n_eval += len(cands); j = int(np.argmax(sc))
            if sc[j] > best + 1e-6: best = float(sc[j]); b = cands[j]; improved = True
    return b, best, n_eval

tr = [e for e in meta("train") if e["method"] == a.method and not e.get("full_watermark")]
items = []
for e in tr:
    mk = read_mask(e["image_name"]); b = bbox(mk); rect = bool(mk[b[0]:b[2], b[1]:b[3]].mean() > 0.98); items.append((e, mk, b, rect))
rects = [it for it in items if it[3]]; print("BOX", a.method, "partielles=%d rectangles=%d objets=%d" % (len(items), len(rects), len(items) - len(rects)), flush=True)
if a.n: rects = rects[:a.n]
if "A" in a.part:
    pert = {"exact": [(0, 0, 0, 0)]}
    for d in (2, 4, 8, 16): pert[f"décalage {d}px"] = [(d, d, d, d), (-d, -d, -d, -d), (d, 0, d, 0), (0, d, 0, d)]
    for d in (2, 4, 8, 16): pert[f"plus grand {d}px/bord"] = [(-d, -d, d, d)]
    for d in (2, 4, 8, 16): pert[f"plus petit {d}px/bord"] = [(d, d, -d, -d)]
    acc = {k: {"zdec": [], "bits": [], "zmf": []} for k in pert}; t0 = time.time()
    for e, mk, b, _ in rects:
        img = read_img("train", e["image_name"]); x = to_t(img, dev); m = bits_of(e); msgs_f = rng.integers(0, 2, (a.K, wm.nbits)).astype(np.float32)
        if min(b[2] - b[0], b[3] - b[1]) < 40: continue           # les décalages de 16 px n'ont pas de sens sur une boîte minuscule
        for k, ds in pert.items():
            bs = [clipbox((b[0] + d[0], b[1] + d[1], b[2] + d[2], b[3] + d[3])) for d in ds]
            z, ac = zdec(x, bs, m); acc[k]["zdec"].append(z.mean()); acc[k]["bits"].append(ac.mean())
            acc[k]["zmf"].append(np.mean([zmf(img, x, bb, m, msgs_f)[0] for bb in bs]))
    print("PARTIE A (netteté), n=%d rectangles de côté >= 40 px, %d s" % (len(acc["exact"]["zdec"]), time.time() - t0))
    print("  perturbation            z décodeur   bits justes   z filtre adapté")
    for k, v in acc.items(): print("  %-22s  %6.2f       %.3f         %6.2f" % (k, np.mean(v["zdec"]), np.mean(v["bits"]), np.mean(v["zmf"])), flush=True)
if "B" in a.part:
    rows = []; t0 = time.time()
    for e, mk, b, _ in rects:
        img = read_img("train", e["image_name"]); x = to_t(img, dev); m = bits_of(e); msgs_f = rng.integers(0, 2, (a.K, wm.nbits)).astype(np.float32)
        z, _ = zdec(x, GRID, m); j = int(np.argmax(z)); b0 = GRID[j]
        b1, z1, n1 = ascent(x, b0, m, score=lambda bs: zdec(x, bs, m)[0])
        b2, z2, n2 = ascent(x, b1, m, steps=(4, 2, 1), score=lambda bs: np.array([zmf(img, x, bb, m, msgs_f)[0] for bb in bs]))
        g128 = downsample_mask(mk); zg = zdec(x, [b], m)[0][0]
        rows.append((iou_box(b0, b), iou_box(b1, b), iou_box(b2, b), miou_image(box_mask128(b0), g128), miou_image(box_mask128(b1), g128), miou_image(box_mask128(b2), g128), z[j], z1, zg, e["gidx"] % 5 == 0))
        if len(rows) % 5 == 0: print("  ", len(rows), round(time.time() - t0), "s", np.round(np.mean(np.array(rows, float), 0), 3), flush=True)
    R = np.array(rows, float); res = R[:, -1] > 0
    print("PARTIE B (recherche), n=%d rectangles (%d en réserve), grille=%d boîtes, %d s" % (len(R), res.sum(), len(GRID), time.time() - t0))
    for nm, sel in (("tous", np.ones(len(R), bool)), ("réserve", res)):
        if sel.sum(): print("  %-8s IoU boîte : grille %.3f -> montée décodeur %.3f -> montée filtre adapté %.3f | mIoU officiel : %.3f -> %.3f -> %.3f | z grille %.1f, z final %.1f, z boîte vraie %.1f" % ((nm,) + tuple(R[sel][:, :9].mean(0))))
    json.dump(rows, open(OUT / f"box_{a.method}.json", "w"), default=float)
