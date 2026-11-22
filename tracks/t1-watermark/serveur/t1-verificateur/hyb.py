"""PixelSeal : (1) recalcul de l'écart candidat N − R en validation croisée, par pli et par type ; (2) règle hybride : rendre la carte brute au seuil 0
au lieu du candidat quand la zone rare PRÉDITE par la carte est petite. Les deux seuils d'aire (pleines, partielles devinées) sont choisis sur 4 plis
(index % 5) et appliqués au cinquième : aucune image ne sert à régler sa propre règle."""
import sys
import numpy as np
from verif import HERE, miou_vec, load_truth
N = sys.argv[1] if len(sys.argv) > 1 else "36"; R = sys.argv[2] if len(sys.argv) > 2 else "35"; ME = sys.argv[3] if len(sys.argv) > 3 else "pixelseal"; TAG = sys.argv[4] if len(sys.argv) > 4 else "ps_f_x1+f_s1+f_deca"
rng = np.random.default_rng(20261122); meta, g = load_truth(check_png=False); idx = {m["image_name"]: i for i, m in enumerate(meta)}; bym = {m["image_name"]: m for m in meta}
z = np.load(HERE / "t1-reel" / TAG / f"pred_train_{ME}.npz"); names = [str(a) for a in z["names"]]; L = z["logits"].astype(np.float32); G = np.stack([g[a] for a in names]); n = len(names)
def lm(c):
    q = np.load(HERE / "t1-assemblage" / f"candidat_{c}_cv_{ME}.npz"); d = {str(a): m.astype(bool) for a, m in zip(q["names"], q["masks"])}; return np.stack([d[a] for a in names])
CN, CR = lm(N), lm(R); fold = np.array([idx[a] % 5 for a in names]); full = np.array([bym[a]["full_watermark"] for a in names])
f = np.load(HERE / "t1-assemblage" / "feat_train.npz"); acc = dict(zip([str(a) for a in f["names"]], f["acc"])); A = np.array([acc[a] for a in names])
RARE = np.where(full[:, None, None], ~G, G); rf = RARE.reshape(n, -1).mean(1)
def bbfill(m):
    ys, xs = np.where(m); return m.sum() / ((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)) if len(ys) else 0
rect = ~full & (np.array([bbfill(RARE[i]) for i in range(n)]) >= 0.95); obj = ~full & ~rect
sN, sR, s0 = miou_vec(CN, G), miou_vec(CR, G), miou_vec(L > 0, G)
def ci(x): b = x[rng.integers(0, len(x), (20000, len(x)))].mean(1); return np.percentile(b, 2.5), np.percentile(b, 97.5)
def show(lab, d, k=None):
    k = np.ones(n, bool) if k is None else k; lo, hi = ci(d[k]); print(f"  {lab:28s} n={k.sum():3d} écart {d[k].mean():+.4f} IC95 [{lo:+.4f} ; {hi:+.4f}] ; réserve {d[k & (fold == 0)].mean():+.4f} ({(k & (fold == 0)).sum()}) ; hors réserve {d[k & (fold != 0)].mean():+.4f}")
print(f"=== (1) {ME} : candidat {N} {sN.mean():.4f} contre candidat {R} {sR.mean():.4f} (carte brute au seuil 0 : {s0.mean():.4f})"); d = sN - sR; show("170 images", d)
print("  par pli : " + " ; ".join(f"pli {k} {d[fold == k].mean():+.4f} ({(fold == k).sum()})" for k in range(5)) + f" ; écart-type par image {d.std():.3f}")
show("pleines", d, full); show("rectangles", d, rect); show("objets", d, obj); j = np.argsort(d)[:3]; print("  trois plus fortes baisses : " + " ; ".join(f"{names[i]} {d[i]:+.3f} (zone {rf[i]:.1%}, {'réserve' if fold[i] == 0 else 'pli ' + str(fold[i])})" for i in j))
print(f"=== (2) règle hybride sur le candidat {N} : carte brute si la zone rare prédite par la carte est sous un seuil d'aire")
P0 = L > 0; GRID = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.15, 1.01]
def hybrid(pf, pr, cf, cp): use = np.where(pf, pr < cf, pr < cp); return np.where(use[:, None, None], P0, CN), use
H = np.zeros_like(CN); USE = np.zeros(n, bool); chosen = []
for k in range(5):
    fit = fold != k; ev = fold == k; th = max(np.unique(A), key=lambda t: (np.mean((A[fit] > t) == full[fit]), -abs(t - 0.75))); pf = A > th; pr = np.where(pf, 1 - P0.reshape(n, -1).mean(1), P0.reshape(n, -1).mean(1))
    best = max(((miou_vec(hybrid(pf, pr, cf, cp)[0][fit], G[fit]).mean(), -cf - cp, cf, cp) for cf in GRID for cp in GRID)); h, u = hybrid(pf, pr, best[2], best[3]); H[ev] = h[ev]; USE[ev] = u[ev]; chosen.append((best[2], best[3]))
sH = miou_vec(H, G); print(f"  seuils choisis par pli (pleines, partielles) : {chosen} ; carte brute rendue pour {USE.sum()} images sur {n}")
print(f"  hybride {sH.mean():.4f} contre candidat {N} {sN.mean():.4f}"); dh = sH - sN; show("hybride − candidat", dh); show("  pleines", dh, full); show("  rectangles", dh, rect); show("  objets", dh, obj)
print("  par pli : " + " ; ".join(f"pli {k} {dh[fold == k].mean():+.4f}" for k in range(5)) + f" ; images mieux / pires {(dh > 1e-9).sum()} / {(dh < -1e-9).sum()}")
pf = A > 0.789; pr = np.where(pf, 1 - P0.reshape(n, -1).mean(1), P0.reshape(n, -1).mean(1))
print("  pour mémoire, sans validation croisée (optimiste) : score selon le seuil d'aire des partielles (pleines : jamais) : " + " ; ".join(f"{c:g} : {miou_vec(hybrid(pf, pr, 0.0, c)[0], G).mean():.4f}" for c in GRID))
print("  idem selon le seuil d'aire des pleines (partielles : jamais) : " + " ; ".join(f"{c:g} : {miou_vec(hybrid(pf, pr, c, 0.0)[0], G).mean():.4f}" for c in GRID))
dh2 = sH - sR; show(f"hybride − candidat {R}", dh2)
