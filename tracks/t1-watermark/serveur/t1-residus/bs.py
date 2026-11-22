"""Recherche de boîte par z-score du décodeur (images partielles « rectangle » : le tampon est calé sur le rectangle).
z(boîte) = somme((2m-1).l) / racine(somme(l^2)) sur les logits l du rognage ; au hasard z ~ N(0,1), maximum = racine(nbits).
1) routeur : bits justes de l'image entière > 0,75 = « pleine » (ignorée) ; 2) grille grossière multi-tailles ; 3) montée de coordonnées
depuis les meilleures graines. Sortie : boxes_<methode>_<split>.json (boîte, z = confiance) ; sur le train : IoU de boîte et mIoU officiel."""
import argparse, json, time
import numpy as np, torch, torch.nn.functional as F
from torchvision.ops import roi_align
from rlib import *

ap = argparse.ArgumentParser(); ap.add_argument("--method", required=True); ap.add_argument("--split", default="train")
ap.add_argument("--n", type=int, default=0); ap.add_argument("--ratio", type=float, default=1.22); ap.add_argument("--stride", type=float, default=2.5)
ap.add_argument("--minside", type=int, default=0); ap.add_argument("--seeds", type=int, default=6); ap.add_argument("--all", action="store_true")
ap.add_argument("--bs", type=int, default=0); ap.add_argument("--init", default=""); ap.add_argument("--netinit", default=""); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1); ap.add_argument("--tag", default=""); a = ap.parse_args()
dev = "cuda"; wm = load(a.method, dev)
if a.method == "stegastamp": ss_decoder_only(wm)
RES = {"pixelseal": 256, "trustmark": 245, "stegastamp": 400}[a.method]
MAXAR = 2.1 if a.method == "trustmark" else 16.5; MINS = a.minside or {"pixelseal": 28, "trustmark": 22, "stegastamp": 22}[a.method]
BS = a.bs or {"pixelseal": 512, "trustmark": 512, "stegastamp": 128}[a.method]

@torch.no_grad()
def zdec(x, boxes, s):
    """boxes : liste de (y0,x0,y1,x1) en bords de pixels. Retour z et bits justes (numpy)."""
    zs, accs = [], []
    for i in range(0, len(boxes), BS):
        b = torch.tensor([(0, bb[1], bb[0], bb[3], bb[2]) for bb in boxes[i:i + BS]], dtype=torch.float32, device=dev)
        cr = roi_align(x, b, output_size=(RES, RES), spatial_scale=1.0, sampling_ratio=2, aligned=True).clamp(0, 1)
        if a.method == "stegastamp": lg = wm.logits(cr, bs=BS).to(dev)
        else:
            with torch.autocast("cuda", dtype=torch.float16): lg = wm.logits(cr)
            lg = lg.float()
        zs.append((lg * s).sum(1) / lg.pow(2).sum(1).sqrt().clamp_min(1e-9)); accs.append(((lg > 0).float() * 2 - 1 == s).float().mean(1))
    return torch.cat(zs).cpu().numpy(), torch.cat(accs).cpu().numpy()

def make_grid():
    sizes = [MINS]
    while sizes[-1] * a.ratio <= 512: sizes.append(sizes[-1] * a.ratio)
    sizes = sorted(set(int(round(v)) for v in sizes) | {512}); out = []
    for h in sizes:
        for w in sizes:
            if max(h, w) / min(h, w) > MAXAR: continue
            sy = max(4, h / a.stride); sx = max(4, w / a.stride)
            ys = np.unique(np.round(np.linspace(0, 512 - h, max(1, int(np.ceil((512 - h) / sy)) + 1))).astype(int))
            xs = np.unique(np.round(np.linspace(0, 512 - w, max(1, int(np.ceil((512 - w) / sx)) + 1))).astype(int))
            out += [(int(y), int(x_), int(y) + h, int(x_) + w) for y in ys for x_ in xs]
    return out
GRID = [] if a.netinit else make_grid(); print("BS", a.method, a.split, "grille=%d boîtes" % len(GRID), flush=True)

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
        c.append(clipbox((b[0] + d, b[1], b[2] + d, b[3]))); c.append(clipbox((b[0], b[1] + d, b[2], b[3] + d)))
        c.append(clipbox((b[0] - d, b[1] - d, b[2] + d, b[3] + d))); c.append(clipbox((b[0] - d, b[1], b[2] + d, b[3]))); c.append(clipbox((b[0], b[1] - d, b[2], b[3] + d)))
    return [t for t in set(c) if t != b]

def ascent(x, b, s, steps=(12, 6, 3, 2, 1)):
    best = float(zdec(x, [b], s)[0][0]); n = 1
    for st in steps:
        while True:
            c = moves(b, st); z = zdec(x, c, s)[0]; n += len(c); j = int(np.argmax(z))
            if z[j] > best + 1e-4: best = float(z[j]); b = c[j]
            else: break
    return b, best, n

def search(x, s, init=None):
    if init is not None:                                  # recherche locale dense autour d'une boîte de départ (décodeur très pointu)
        y0, x0, y1, x1 = init; c = set()
        for d0 in range(-16, 17, 4):
            for d1 in range(-16, 17, 4):
                for d2 in range(-16, 17, 4):
                    for d3 in range(-16, 17, 4): c.add(clipbox((y0 + d0, x0 + d1, y1 + d2, x1 + d3)))
        cands = list(c)
    else: cands = GRID
    z, _ = zdec(x, cands, s); order = np.argsort(-z); seeds = []
    for j in order[:200]:
        if all(iou_box(cands[j], t) < 0.5 for t in seeds): seeds.append(cands[j])
        if len(seeds) >= a.seeds: break
    best = (None, -1e9); n = len(cands)
    for sd in seeds:
        b, zb, k = ascent(x, sd, s); n += k
        if zb > best[1]: best = (b, zb)
    zg = float(z[order[0]]); return best[0], best[1], zg, cands[order[0]], n

def net_boxes(L, kmax=3):
    """Boîtes de départ tirées de la carte d'un réseau (128x128, > 0 = filigrané) : boîtes des plus grosses composantes connexes."""
    from scipy import ndimage as ndi
    p = L > 0; lab, k = ndi.label(p)
    if k == 0: p = L > np.quantile(L, 0.97); lab, k = ndi.label(p)
    sc = ndi.sum(np.maximum(L, 0) + 0.1, lab, range(1, k + 1)); out = []
    for j in np.argsort(-sc)[:kmax]:
        yy, xx = np.where(lab == j + 1)
        if len(yy) >= 3: out.append((int(yy.min()) * 4, int(xx.min()) * 4, int(yy.max()) * 4 + 4, int(xx.max()) * 4 + 4))
    return out

def local_cands(b, scales, rng_=0.32, step=0.04):
    y0, x0, y1, x1 = b; h = y1 - y0; w = x1 - x0; cy = (y0 + y1) / 2; cx = (x0 + x1) / 2; out = set(); offs = np.arange(-rng_, rng_ + 1e-9, step)
    for sh, sw in scales:
        hh = min(512, h * sh); ww = min(512, w * sw)
        if max(hh, ww) / max(8, min(hh, ww)) > MAXAR: continue
        for dy in offs:
            for dx in offs:
                ty = int(round(cy + dy * hh - hh / 2)); tx = int(round(cx + dx * ww - ww / 2)); out.add(clipbox((ty, tx, ty + int(round(hh)), tx + int(round(ww)))))
    return list(out)

def search_net(x, s, inits):
    """Décodeur pointu (StegaStamp) : recherche locale dense autour des boîtes du réseau, tailles élargies par paliers tant que rien ne ressort."""
    S1 = [(1, 1)]; S2 = [(a_, b_) for a_ in (0.8, 1, 1.25) for b_ in (0.8, 1, 1.25) if (a_, b_) != (1, 1)]
    S3 = [(a_, b_) for a_ in (0.62, 0.8, 1, 1.25, 1.6) for b_ in (0.62, 0.8, 1, 1.25, 1.6) if a_ in (0.62, 1.6) or b_ in (0.62, 1.6)]
    best = (None, -1e9); n = 0; zg = -1e9; bg = None
    for scales in (S1, S2, S3):
        seeds = []
        for b in inits:
            c = local_cands(b, scales); z, _ = zdec(x, c, s); n += len(c); order = np.argsort(-z)
            if z[order[0]] > zg: zg = float(z[order[0]]); bg = c[order[0]]
            for j in order[:3]:
                if z[j] > 3.0 and all(iou_box(c[j], t) < 0.6 for t in seeds): seeds.append(c[j])
        for sd in seeds[:4]:
            bb, zb, k = ascent(x, sd, s, steps=(4, 2, 1)); n += k
            if zb > best[1]: best = (bb, zb)
        if best[1] > 6.0: break
    if best[0] is None: best = (bg, zg)
    return best[0], best[1], zg, bg, n

def bbox(mask):
    ys, xs = np.where(mask > 0); return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1

def box_mask128(b):
    mk = np.zeros((512, 512), np.uint8); mk[b[0]:b[2], b[1]:b[3]] = 1; return downsample_mask(mk)

M = [e for e in meta(a.split) if e["method"] == a.method][a.shard::a.nshards]
if a.nshards > 1: a.tag = a.tag + "_s%dof%d" % (a.shard, a.nshards)
NET = {}
if a.netinit:
    zz = np.load(a.netinit, allow_pickle=True); NET = dict(zip(zz["names"].tolist(), zz["logits"].astype(np.float32)))
init = json.load(open(a.init)) if a.init else {}
out = []; t0 = time.time(); done = 0
for e in M:
    img = read_img(a.split, e["image_name"]); x = to_t(img, dev); m = bits_of(e); s = torch.from_numpy(2 * m - 1).float().to(dev)
    zf, af = zdec(x, [(0, 0, 512, 512)], s); rec = {"name": e["image_name"], "gidx": e["gidx"], "acc_full": float(af[0]), "z_full": float(zf[0])}
    truth_partial = (a.split == "train") and not e.get("full_watermark")
    if a.split == "train":
        mk = read_mask(e["image_name"]); g = bbox(mk); rec["full"] = bool(e.get("full_watermark")); rec["gt_box"] = g; rec["rect"] = bool(mk[g[0]:g[2], g[1]:g[3]].mean() > 0.98); rec["frac"] = float(mk.mean())
    if (af[0] < 0.75) or (a.all and truth_partial):
        if a.netinit:
            ib = net_boxes(NET[e["image_name"]]); rec["net_boxes"] = ib
            if not ib: out.append(rec); continue
            b, zb, zg, bg, n = search_net(x, s, ib)
        else: b, zb, zg, bg, n = search(x, s, init.get(e["image_name"]))
        rec.update(box=b, z=zb, z_grid=zg, box_grid=bg, n_eval=n)
        if a.split == "train" and not rec["full"]:
            g128 = downsample_mask(mk); rec.update(iou=iou_box(b, g), iou_grid=iou_box(bg, g), miou_box=miou_image(box_mask128(b), g128), miou_zero=miou_image(np.zeros((128, 128)), g128), z_gt=float(zdec(x, [g], s)[0][0]))
        done += 1
        if done % 5 == 0 or (a.n and done >= a.n): print("  %d images cherchées, %d s ; dernière : z=%.1f %s" % (done, time.time() - t0, zb, ("IoU=%.2f rect=%s" % (rec.get("iou", -1), rec.get("rect"))) if "iou" in rec else ""), flush=True)
    out.append(rec)
    if done and done % 5 == 0: json.dump(out, open(OUT / f"boxes_{a.method}_{a.split}{a.tag}.json", "w"), default=float)
    if a.n and done >= a.n: break
json.dump(out, open(OUT / f"boxes_{a.method}_{a.split}{a.tag}.json", "w"), default=float)
print("FIN", a.method, a.split, "n=%d cherchées=%d sec=%d" % (len(out), done, time.time() - t0))
if a.split == "train":
    P = [r for r in out if not r["full"]]; Fu = [r for r in out if r["full"]]
    print("ROUTEUR (bits justes image entière < 0,75 = partielle) : pleines prises pour partielles %d/%d ; partielles prises pour pleines %d/%d" % (sum(r["acc_full"] < 0.75 for r in Fu), len(Fu), sum(r["acc_full"] >= 0.75 for r in P), len(P)))
    S = [r for r in P if "box" in r and "iou" in r]
    for nm, sel in (("rectangles", [r for r in S if r["rect"]]), ("objets", [r for r in S if not r["rect"]])):
        if not sel: continue
        z = np.array([r["z"] for r in sel]); iou = np.array([r["iou"] for r in sel]); res = np.array([r["gidx"] % 5 == 0 for r in sel])
        print("%-10s n=%d | z trouvé : médiane %.1f [q10 %.1f ; q90 %.1f] | z boîte vraie médiane %.1f | IoU boîte : moyenne %.3f, médiane %.3f, part > 0,9 : %.2f | mIoU officiel si masque = boîte : %.3f (réserve %.3f, n=%d) ; si tout-0 : %.3f" % (
            nm, len(sel), np.median(z), np.quantile(z, .1), np.quantile(z, .9), np.median([r["z_gt"] for r in sel]), iou.mean(), np.median(iou), (iou > 0.9).mean(),
            np.mean([r["miou_box"] for r in sel]), np.mean([r["miou_box"] for r, k in zip(sel, res) if k]) if res.any() else float("nan"), res.sum(), np.mean([r["miou_zero"] for r in sel])))
        if nm == "rectangles":
            d = np.array([[r["box"][k] - r["gt_box"][k] for k in range(4)] for r in sel if r["iou"] > 0.7])
            if len(d): print("           écart médian boîte trouvée - boîte vraie (y0,x0,y1,x1) sur les IoU > 0,7 :", np.median(d, 0), "n=%d" % len(d))
