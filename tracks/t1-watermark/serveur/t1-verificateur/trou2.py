"""Complément à trou.py : (i) plafond « localisation grossière » : le trou des pleines connu seulement sur une grille k x k (bloc propre si plus de la
moitié de ses pixels le sont) ; (j) épaisseur du trou (part qui survit à une érosion 3x3, une et deux fois). Base : masques honnêtes du candidat N."""
import sys
import numpy as np
from scipy.ndimage import binary_erosion
from verif import HERE, METHODS, miou_vec, load_truth
N = sys.argv[1] if len(sys.argv) > 1 else "6"
A = HERE / "t1-assemblage"; meta, g = load_truth(check_png=False); idx = {m["image_name"]: i for i, m in enumerate(meta)}; bym = {m["image_name"]: m for m in meta}
def grid(Z, k):
    b = 128 // k; return np.kron(Z.reshape(k, b, k, b).mean((1, 3)) > 0.5, np.ones((b, b), bool))
TOT = {}
for me in METHODS:
    z = np.load(A / f"candidat_{N}_honnete_{me}.npz"); names = [str(a) for a in z["names"]]; P = z["masks"].astype(bool); G = np.stack([g[a] for a in names])
    res = np.array([idx[a] % 5 == 0 for a in names]); full = np.array([bym[a]["full_watermark"] for a in names]); s = miou_vec(P, G)
    H = ~G[full]; e1 = np.array([binary_erosion(h, np.ones((3, 3))).sum() / h.sum() for h in H]); e2 = np.array([binary_erosion(h, np.ones((3, 3)), iterations=2).sum() / h.sum() for h in H])
    print(f"\n[{me}] trou des pleines : part qui survit à une érosion 3x3 médiane {np.median(e1):.2f}, à deux érosions {np.median(e2):.2f} ; trous qui disparaissent après deux érosions : {np.mean(e2 == 0):.2f}")
    Z = G[~full]; f1 = np.array([binary_erosion(h, np.ones((3, 3)), iterations=2).sum() / h.sum() for h in Z]); print(f"   zone des partielles : part qui survit à deux érosions médiane {np.median(f1):.2f}")
    for k in (4, 8, 16, 32):
        Pk = P.copy(); Pk[full] = ~np.stack([grid(~G[i], k) for i in np.where(full)[0]]); v = miou_vec(Pk, G); vm = np.where(full, np.maximum(v, s), s)
        print(f"   (i) trou des pleines connu sur une grille {k:2d}x{k:<2d} : pleines seules {v[full].mean():.4f} (base {s[full].mean():.4f}) | réserve {v[res].mean():.4f} ({v[res].mean() - s[res].mean():+.4f}) | hors réserve {v[~res].mean():.4f} ({v[~res].mean() - s[~res].mean():+.4f})")
        TOT.setdefault(k, []).append((v[res].mean() - s[res].mean(), v[~res].mean() - s[~res].mean()))
print("\nMOYENNE DES TROIS : " + " ; ".join(f"grille {k}x{k} : réserve {np.mean(v, 0)[0]:+.4f}, hors réserve {np.mean(v, 0)[1]:+.4f}" for k, v in TOT.items()))
