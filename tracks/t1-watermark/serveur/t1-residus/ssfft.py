"""StegaStamp : peut-on trouver la POSITION d'un rectangle par corrélation croisée (FFT, toutes les positions d'un coup)
entre l'image filtrée passe-haut et le motif que le tampon écrit pour notre message ? Rectangles réels hors réserve, taille vraie connue.
Variantes de fond pour calculer le motif : rognage vrai (oracle), gris uni, image entière réduite."""
import argparse, time
import numpy as np, torch, torch.nn.functional as F
from rlib import *
ap = argparse.ArgumentParser(); ap.add_argument("--method", default="stegastamp"); ap.add_argument("--n", type=int, default=20); ap.add_argument("--den", default="g2"); a = ap.parse_args()
dev = "cuda"; wm = load(a.method, dev); rng = np.random.default_rng(0)
def bbox(mask):
    ys, xs = np.where(mask > 0); return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1
def xcorr(hx, t):
    """hx : 1x3x512x512 (image passe-haut) ; t : 3xhxw (motif passe-haut). Carte de corrélation normalisée pour chaque position du coin haut-gauche."""
    h, w = t.shape[-2:]; t = t - t.mean((1, 2), keepdim=True)
    num = F.conv2d(hx, t[None])[0, 0]                                                    # (513-h) x (513-w)
    en = F.avg_pool2d((hx * hx).sum(1, keepdim=True), (h, w), stride=1)[0, 0] * h * w
    return num / (en.sqrt() * t.norm() + 1e-12)
items = []
for e in meta("train"):
    if e["method"] != a.method or e.get("full_watermark") or e["gidx"] % 5 == 0: continue
    mk = read_mask(e["image_name"]); g = bbox(mk)
    if mk[g[0]:g[2], g[1]:g[3]].mean() > 0.98 and min(g[2] - g[0], g[3] - g[1]) >= 24: items.append((e, g))
items = items[:a.n]; V = ["rognage vrai (oracle)", "gris uni", "image entière réduite", "rognage vrai, taille +15%", "gris, faux message"]
res = {k: {"err": [], "z": [], "hit": []} for k in V}; cs = []; t0 = time.time()
for e, g in items:
    img = read_img("train", e["image_name"]); x = to_t(img, dev); m = bits_of(e); h, w = g[2] - g[0], g[3] - g[1]; hx = hp(a.den, x)
    crop = np.ascontiguousarray(img[g[0]:g[2], g[1]:g[3]]); gray = np.full((h, w, 3), 128, np.uint8)
    whole = np.asarray(Image.fromarray(img).resize((w, h), Image.BILINEAR)); mf = rng.integers(0, 2, wm.nbits).astype(np.float32)
    T = {"rognage vrai (oracle)": tmpl(wm, crop, m)[0], "gris uni": tmpl(wm, gray, m)[0], "image entière réduite": tmpl(wm, whole, m)[0], "gris, faux message": tmpl(wm, gray, mf)[0]}
    h2, w2 = int(h * 1.15), int(w * 1.15)
    if h2 <= 512 and w2 <= 512: T["rognage vrai, taille +15%"] = F.interpolate(T["rognage vrai (oracle)"][None], size=(h2, w2), mode="bilinear")[0]
    cs.append((cos(T["gris uni"], T["rognage vrai (oracle)"]), cos(T["image entière réduite"], T["rognage vrai (oracle)"])))
    for k, t in T.items():
        c = xcorr(hx, hp(a.den, t[None])[0]); j = int(c.argmax()); py, px = j // c.shape[1], j % c.shape[1]
        gy, gx = (g[0], g[1]) if "+15%" not in k else (g[0] - (h2 - h) // 2, g[1] - (w2 - w) // 2)
        err = max(abs(py - gy), abs(px - gx)); res[k]["err"].append(err); res[k]["z"].append(float((c.max() - c.mean()) / c.std())); res[k]["hit"].append(err <= 4)
print("SSFFT", a.method, "n=%d rectangles (petit côté >= 24 px), filtre %s, %d s" % (len(items), a.den, time.time() - t0))
print("  ressemblance du motif calculé sur un autre fond avec le motif du vrai rognage (cosinus médian) : gris %.2f ; image entière réduite %.2f" % tuple(np.median(np.array(cs), 0)))
print("  fond du motif                 pic à <= 4 px de la vraie position   erreur médiane (px)   hauteur du pic (écarts-types)")
for k, v in res.items():
    if v["err"]: print("  %-28s  %.2f                                 %5.0f                 %.1f" % (k, np.mean(v["hit"]), np.median(v["err"]), np.median(v["z"])))
