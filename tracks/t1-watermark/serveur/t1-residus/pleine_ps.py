"""Même mesure que pleine.py pour PixelSeal (ou StegaStamp) : chemin exact du tampon, une toile par appel.
Toile = carré de côté 512 + s, image = rognage placé en (haut, gauche) : 3 paramètres (structure mesurée sur TrustMark), puis descente sur les 4 marges.
Critère global R3 = énergie(vrai message) / énergie(K messages au hasard) sur la partie image. Carte locale : rapport lissé, 128x128.
--fab : pleines FABRIQUÉES à marges connues (témoin : la recherche retrouve-t-elle la toile, et avec quel pas ?)."""
import argparse, time
import numpy as np, torch, torch.nn.functional as F
from rlib import *
ap = argparse.ArgumentParser(); ap.add_argument("--method", default="pixelseal"); ap.add_argument("--n", type=int, default=10); ap.add_argument("--K", type=int, default=2); ap.add_argument("--Kmap", type=int, default=8)
ap.add_argument("--step", type=int, default=4); ap.add_argument("--smax", type=int, default=80); ap.add_argument("--fab", action="store_true"); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1); a = ap.parse_args()
dev = "cuda"; wm = load(a.method, dev); rng = np.random.default_rng(3); PAD = a.smax + 16; SIGS = (6.0, 12.0, 20.0, 32.0)
def energy(imgp, f, msgs):
    """Énergie par pixel (K x 512 x 512) de ce que le tampon ajouterait à la toile f = (y0,x0,y1,x1), restreinte à l'image."""
    crop = np.ascontiguousarray(imgp[f[0] + PAD:f[2] + PAD, f[1] + PAD:f[3] + PAD]); return tmpl(wm, crop, msgs).pow(2).sum(1)[:, -f[0]:-f[0] + 512, -f[1]:-f[1] + 512]
def r3(imgp, f, msgs):
    e = energy(imgp, f, msgs).sum((1, 2)); return float(e[0] / e[1:].mean())
def search(imgp, msgs):
    best = (None, 9.0); vals = []
    for s in range(0, a.smax + 1, a.step):
        for t in range(0, s + 1, a.step):
            for l in range(0, s + 1, a.step):
                f = (-t, -l, 512 + s - t, 512 + s - l); v = r3(imgp, f, msgs); vals.append(v)
                if v < best[1]: best = (f, v)
    b, bv = best
    for st in (2, 1):
        while True:
            c = [tuple(b[k] + (d if k == q else 0) for k in range(4)) for q in range(4) for d in (-st, st)] + [(b[0] + d, b[1], b[2] + d, b[3]) for d in (-st, st)] + [(b[0], b[1] + d, b[2], b[3] + d) for d in (-st, st)]
            c = [t for t in c if t[0] <= 0 and t[1] <= 0 and t[2] >= 512 and t[3] >= 512 and min(t[0], t[1]) > -PAD and max(t[2], t[3]) < 512 + PAD]
            v = [r3(imgp, t, msgs) for t in c]; j = int(np.argmin(v))
            if v[j] < bv - 1e-4: b, bv = c[j], float(v[j])
            else: break
    return b, bv, float(np.median(vals)), vals[0]
tr = meta("train"); others = [e for e in tr if e["method"] != a.method and e["gidx"] % 5 != 0]
real = [e for e in tr if e["method"] == a.method and e["gidx"] % 5 != 0 and e.get("full_watermark")][a.shard::a.nshards][:a.n]
print("PLEINE_PS", a.method, "FABRIQUÉ" if a.fab else "RÉEL", "n=%d pas %d" % (len(real), a.step), flush=True); t0 = time.time(); A = []; R3s = []
for i, e in enumerate(real):
    mk = read_mask(e["image_name"])
    if a.fab:
        base = read_img("train", others[i * 5 % len(others)]["image_name"]); m = rng.integers(0, 2, wm.nbits).astype(np.float32); s = int(rng.integers(8, 64)); t = int(rng.integers(0, s + 1)); l = int(rng.integers(0, s + 1))
        c = np.pad(base, ((t, s - t), (l, s - l), (0, 0)), mode="reflect"); img = wm.embed(c, m)[t:t + 512, l:l + 512].copy(); img[mk == 0] = base[mk == 0]; truth = (-t, -l, 512 + s - t, 512 + s - l)
    else: img = read_img("train", e["image_name"]); m = bits_of(e); truth = None
    imgp = np.pad(img, ((PAD, PAD), (PAD, PAD), (0, 0)), mode="reflect"); msgs = np.concatenate([m[None], rng.integers(0, 2, (a.K, wm.nbits)).astype(np.float32)])
    f, v, med, v0 = search(imgp, msgs); msgs2 = np.concatenate([m[None], rng.integers(0, 2, (a.Kmap, wm.nbits)).astype(np.float32)]); en = energy(imgp, f, msgs2)[:, None]; g = downsample_mask(mk); aucs = []
    for sg in SIGS:
        es = gauss(en, sg)[:, 0]; den = es[1:].mean(0); R = F.avg_pool2d(((es[0] + 0.05 * den.mean()) / (den + 0.05 * den.mean()))[None, None], 4)[0, 0].cpu().numpy()
        aucs.append(auc(-R[g > 0], -R[g == 0]) if (g == 0).sum() > 10 else np.nan)
    A.append(aucs); R3s.append(v); extra = ""
    if truth: extra = " | VRAIE toile marges (%d,%d,%d,%d), R3 à la vraie toile %.2f, à +2 px %.2f, à +4 px %.2f, à +8 px %.2f" % (-truth[0], -truth[1], truth[2] - 512, truth[3] - 512, r3(imgp, truth, msgs), r3(imgp, (truth[0] - 2, truth[1], truth[2] - 2, truth[3]), msgs), r3(imgp, (truth[0] - 4, truth[1], truth[2] - 4, truth[3]), msgs), r3(imgp, (truth[0] - 8, truth[1], truth[2] - 8, truth[3]), msgs))
    print("  %d : %d s | R3 image entière %.2f -> toile marges (%d,%d,%d,%d) R3 %.2f (médiane grille %.2f) | part filigranée %.2f, AUC des trous par lissage %s%s" % (i + 1, time.time() - t0, v0, -f[0], -f[1], f[2] - 512, f[3] - 512, v, med, mk.mean(), " ".join("%.2f" % q for q in aucs), extra), flush=True)
A = np.array(A); R3s = np.array(R3s)
print("RESULT", a.method, "FABRIQUÉ" if a.fab else "RÉEL", "n=%d | R3 à la toile trouvée : médiane %.2f, < 0,8 : %d | AUC des trous moyenne par lissage %s, médiane %s" % (len(A), np.median(R3s), (R3s < 0.8).sum(), np.round(np.nanmean(A, 0), 3), np.round(np.nanmedian(A, 0), 3)))
