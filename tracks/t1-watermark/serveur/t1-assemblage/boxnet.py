"""Recherche de boite par le decodeur, amorcee par la carte du reseau (rapide : ~1000 rognages par image au lieu de 13 000 a 48 000).
Pour chaque image classee partielle : boite englobante des 2 plus grosses composantes de la carte seuillee -> grille locale (decalages du centre
de -24 a +24 px, tailles x0,85 / x1 / x1,18) notee par le z du decodeur -> montee bord par bord sur la marge brute (rognage exact).
Sortie : boxnet/boxes_<filigrane>_<split>_net.json (name, box = y0, x0, y1, x1, z, bits) ; sur le train, IoU avec la vraie boite.
Usage : python boxnet.py stegastamp reel/pred train|test [n]"""
import sys, json, time, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); from asm import *
sys.path.insert(0, str(HERE / "kit")); from wmloc.watermarks import load_watermark
m, base, split = sys.argv[1], sys.argv[2], sys.argv[3]; NMAX = int(sys.argv[4]) if len(sys.argv) > 4 else 0; dev = "cuda"; torch.set_num_threads(4)
wm = load_watermark(m, models_dir=HERE / "kit" / "models", device=dev); RES = {"pixelseal": 256, "trustmark": 245, "stegastamp": 400}[m]; MODE = "bicubic" if m == "stegastamp" else "bilinear"
def logits(x, boxes):
    cr = torch.cat([F.interpolate(x[:, :, b[0]:b[2], b[1]:b[3]], size=(RES, RES), mode=MODE, align_corners=False, antialias=True).clamp(0, 1) for b in boxes]); out = []
    for i in range(0, len(cr), 48):
        with torch.no_grad(): out.append((wm.logits(cr[i:i + 48], bs=48) if m == "stegastamp" else wm.logits(cr[i:i + 48])).float().to(dev))
    return torch.cat(out)
def score(x, boxes, s):
    lg = logits(x, boxes) * s; return (lg.sum(1) / lg.pow(2).sum(1).sqrt().clamp_min(1e-9)).cpu().numpy(), lg.mean(1).cpu().numpy(), (lg > 0).float().mean(1).cpu().numpy()
def clip(b): y0, x0, y1, x1 = [int(round(v)) for v in b]; y0 = max(0, min(y0, 496)); x0 = max(0, min(x0, 496)); return (y0, x0, max(y0 + 16, min(y1, 512)), max(x0 + 16, min(x1, 512)))
def ascent(x, b, s, steps=(6, 3, 2, 1)):
    best = float(score(x, [b], s)[1][0])
    for st in steps:
        for _ in range(30):
            c = []
            for k in range(4):
                for d in (-st, st): t = list(b); t[k] += d; c.append(clip(t))
            for d in (-st, st): c += [clip((b[0] + d, b[1], b[2] + d, b[3])), clip((b[0], b[1] + d, b[2], b[3] + d)), clip((b[0] - d, b[1] - d, b[2] + d, b[3] + d))]
            c = [t for t in set(c) if t != b]; v = score(x, c, s)[1]; j = int(np.argmax(v))
            if v[j] > best + 1e-5: best = float(v[j]); b = c[j]
            else: break
    return b
def iou(b, g):
    iy = max(0, min(b[2], g[2]) - max(b[0], g[0])); ix = max(0, min(b[3], g[3]) - max(b[1], g[1])); i = iy * ix
    return i / ((b[2] - b[0]) * (b[3] - b[1]) + (g[2] - g[0]) * (g[3] - g[1]) - i)
tr, te, S, notes = discover(); P = populations(tr, te); rows = tr[m] if split == "train" else te[m]; L = S[m][base][0 if split == "train" else 1]; part = ~P[m]["tr" if split == "train" else "te"]
sh = {r["name"]: r for r in json.loads((ME / "shapes_oracle.json").read_text())} if split == "train" else {}
(ME / "boxnet").mkdir(exist_ok=True); out = []; t0 = time.time(); done = 0
for i in np.where(part)[0]:
    r = rows[i]; x = torch.from_numpy(load_img(split, r["image_name"]).copy()).permute(2, 0, 1)[None].float().div(255).to(dev); s = torch.tensor([int(c) for c in r["message"]], device=dev).float() * 2 - 1
    lab, n = ndi.label(L[i] > 0); seeds = []
    if n:
        sz = np.bincount(lab.ravel())[1:]
        for k in np.argsort(-sz)[:2]:
            ys, xs = np.where(lab == k + 1); seeds.append((ys.min() * 4, xs.min() * 4, ys.max() * 4 + 4, xs.max() * 4 + 4))
    rec = dict(name=r["image_name"], n_comp=int(n))
    if seeds:
        cands = set()
        for (y0, x0, y1, x1) in seeds:
            cy, cx, h, w = (y0 + y1) / 2, (x0 + x1) / 2, max(y1 - y0, 24), max(x1 - x0, 24)
            for fh in (0.85, 1.0, 1.18):
                for fw in (0.85, 1.0, 1.18):
                    for dy in range(-24, 25, 6):
                        for dx in range(-24, 25, 6): cands.add(clip((cy + dy - h * fh / 2, cx + dx - w * fw / 2, cy + dy + h * fh / 2, cx + dx + w * fw / 2)))
        cands = list(cands); z, mg, ac = score(x, cands, s); top = [cands[j] for j in np.argsort(-z)[:3]]
        fin = [ascent(x, b, s) for b in top]; z2, mg2, ac2 = score(x, fin, s); j = int(np.argmax(mg2)); b = fin[j]
        rec.update(box=[int(v) for v in b], z=float(z2[j]), marge=float(mg2[j]), bits=float(ac2[j]), z_grille=float(z.max()), n_eval=len(cands))
        if split == "train" and not r["full"]:
            g = sh[r["image_name"]]["bbox"]; g = (g[0], g[1], g[0] + g[2], g[1] + g[3]); rec.update(rect=bool(sh[r["image_name"]]["rect"] > 0.98), gt_box=list(g), iou=float(iou(b, g)), iou_init=float(max(iou(sd, g) for sd in seeds)))
    out.append(rec); done += 1
    if done % 10 == 0: json.dump(out, open(ME / "boxnet" / f"boxes_{m}_{split}_net.json", "w")); print(done, "/", int(part.sum()), round(time.time() - t0), "s", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in rec.items() if k in ("z", "bits", "iou", "iou_init", "rect")}, flush=True)
    if NMAX and done >= NMAX: break
json.dump(out, open(ME / "boxnet" / f"boxes_{m}_{split}_net.json", "w")); print("FIN", m, split, done, round(time.time() - t0), "s", flush=True)
if split == "train":
    T = [r for r in out if "iou" in r]
    for nom, sel in (("rectangles", [r for r in T if r["rect"]]), ("objets", [r for r in T if not r["rect"]])):
        if sel: print(f"{nom:10s} n={len(sel)} | z mediane {np.median([r['z'] for r in sel]):.1f} [q10 {np.quantile([r['z'] for r in sel], .1):.1f} ; q90 {np.quantile([r['z'] for r in sel], .9):.1f}] | IoU boite du reseau {np.mean([r['iou_init'] for r in sel]):.3f} -> boite du decodeur {np.mean([r['iou'] for r in sel]):.3f} (part > 0,9 : {np.mean([r['iou'] > .9 for r in sel]):.2f})")
    R_ = [r for r in T if r["rect"]]
    for z0 in (5, 6, 7, 8): print(f"   seuil z >= {z0} : rectangles pris {np.mean([r['z'] >= z0 for r in R_]):.2f}, IoU de ceux-la {np.mean([r['iou'] for r in R_ if r['z'] >= z0] or [0]):.3f} | objets pris a tort {np.mean([r['z'] >= z0 for r in T if not r['rect']]):.2f}")
