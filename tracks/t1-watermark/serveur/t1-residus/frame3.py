"""Images PLEINES : balayage fin (pas 1 px) du cadre autour de l'image entière avec le critère « re-tamponnage » (R3), chemin exact du tampon.
Familles : cadre agrandi/rétréci de d px par bord ; cadre de taille 512 décalé ; cadre 520 décalé. L'image est prolongée par recopie du bord."""
import argparse, time
import numpy as np, torch
from rlib import *
ap = argparse.ArgumentParser(); ap.add_argument("--method", default="trustmark"); ap.add_argument("--n", type=int, default=4); ap.add_argument("--K", type=int, default=2)
ap.add_argument("--fab", action="store_true"); ap.add_argument("--pop", default="pleines"); a = ap.parse_args()
dev = "cuda"; wm = load(a.method, dev); rng = np.random.default_rng(0); PAD = 32
def r3(imgp, b, msgs, mask=None):
    crop = np.ascontiguousarray(imgp[b[0] + PAD:b[2] + PAD, b[1] + PAD:b[3] + PAD]); T = tmpl(wm, crop, msgs); en = T.pow(2).sum(1)      # K x h x w
    y0, x0 = max(0, b[0]), max(0, b[1]); y1, x1 = min(512, b[2]), min(512, b[3]); en = en[:, y0 - b[0]:y1 - b[0], x0 - b[1]:x1 - b[1]]
    if mask is not None: en = en * torch.from_numpy(mask[y0:y1, x0:x1]).to(dev).float()
    e = en.sum((1, 2)); return float(e[0] / e[1:].mean())
tr = meta("train"); M_ = [e for e in tr if e["method"] == a.method and e["gidx"] % 5 != 0 and e.get("full_watermark")]
if a.pop == "objets":
    M_ = []
    for e in tr:
        if e["method"] != a.method or e["gidx"] % 5 == 0 or e.get("full_watermark"): continue
        mk = read_mask(e["image_name"]); ys, xs = np.where(mk)
        if mk[ys.min():ys.max() + 1, xs.min():xs.max() + 1].mean() <= 0.98 and mk.mean() > 0.03: M_.append(e)
M_ = M_[:a.n]
others = [e for e in tr if e["method"] != a.method and e["gidx"] % 5 != 0]
print("FRAME3", a.method, "FABRIQUÉ (tampon sur l'image entière, moitié gauche remise propre)" if a.fab else "RÉEL", a.pop, "n=%d ; R3 mesuré dans la zone filigranée (1,0 = rien reconnu)" % len(M_), flush=True)
t0 = time.time()
for i, e in enumerate(M_):
    if a.fab:
        img = read_img("train", others[i]["image_name"]); m = rng.integers(0, 2, wm.nbits).astype(np.float32); w = wm.embed(img, m).copy(); w[:, :256] = img[:, :256]; img = w
        mk = np.ones((512, 512), np.uint8); mk[:, :256] = 0
    else: img = read_img("train", e["image_name"]); m = bits_of(e); mk = read_mask(e["image_name"])
    imgp = np.pad(img, ((PAD, PAD), (PAD, PAD), (0, 0)), mode="edge"); msgs = np.concatenate([m[None], rng.integers(0, 2, (a.K, wm.nbits)).astype(np.float32)])
    A = {d: r3(imgp, (-d, -d, 512 + d, 512 + d), msgs, mk) for d in range(-8, 9)}
    best = {}
    for nm, sz in (("512", 512), ("520", 520)):
        v = {(dy, dx): r3(imgp, (dy - (sz - 512) // 2, dx - (sz - 512) // 2, dy - (sz - 512) // 2 + sz, dx - (sz - 512) // 2 + sz), msgs, mk) for dy in range(-6, 7, 1) for dx in range(-6, 7, 1)}
        k = min(v, key=v.get); best[nm] = (k, v[k], float(np.median(list(v.values()))))
    A_out = r3(imgp, (0, 0, 512, 512), msgs, 1 - mk)
    print("  image %d (part filigranée %.2f, R3 HORS zone au cadre image : %.2f) | cadre agrandi de d px par bord : " % (i, mk.mean(), A_out) + " ".join("%d:%.2f" % (d, A[d]) for d in range(-8, 9)), flush=True)
    print("           meilleur décalage, taille 512 : %s R3 %.2f (médiane %.2f) ; taille 520 : %s R3 %.2f (médiane %.2f) | %d s" % (best["512"] + best["520"] + (time.time() - t0,)), flush=True)
