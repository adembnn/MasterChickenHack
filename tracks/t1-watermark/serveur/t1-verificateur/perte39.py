"""Pourquoi le gain de la nouvelle carte PixelSeal (+0,018 seule) se réduit à +0,008 après assemblage (candidat 39 contre 38) ?
Par type d'image : carte brute ancienne / nouvelle au seuil 0, candidat 38 / 39 ; plafond « meilleur des deux par image » ; et règles sans vérité
pour choisir entre le masque du 38 et celui du 39 (accord avec la nouvelle carte), choisies sur 4 plis et appliquées au cinquième."""
import numpy as np
from scipy import ndimage as ndi
from verif import HERE, miou_vec, load_truth
rng = np.random.default_rng(20261122); meta, g = load_truth(check_png=False); idx = {m["image_name"]: i for i, m in enumerate(meta)}; bym = {m["image_name"]: m for m in meta}; A = HERE / "t1-assemblage"; R = HERE / "t1-reel"
def cand(c):
    q = np.load(A / f"candidat_{c}_cv_pixelseal.npz"); return {str(a): m.astype(bool) for a, m in zip(q["names"], q["masks"])}
def L(tag):
    z = np.load(R / tag / "pred_train_pixelseal.npz"); return {str(a): l.astype(np.float32) for a, l in zip(z["names"], z["logits"])}
c38, c39, lo, ln = cand("38"), cand("39"), L("ps_f_x2+f_x1+f_dx"), L("ps_pop_x2+x2b+x1+dx__st+x2"); names = sorted(c38); n = len(names); S = lambda d: np.stack([d[a] for a in names])
G = np.stack([g[a] for a in names]); C38, C39, LO, LN = S(c38), S(c39), S(lo), S(ln); full = np.array([bym[a]["full_watermark"] for a in names]); fold = np.array([idx[a] % 5 for a in names]); RARE = np.where(full[:, None, None], ~G, G)
def bb(m):
    ys, xs = np.where(m); return (m.sum() / ((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)), min(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1) * 4) if len(ys) else (0, 0)
B = np.array([bb(RARE[i]) for i in range(n)]); rect = ~full & (B[:, 0] >= .95); obj = ~full & ~rect
s38, s39, so, sn = miou_vec(C38, G), miou_vec(C39, G), miou_vec(LO > 0, G), miou_vec(LN > 0, G)
print("PixelSeal : carte ancienne | carte nouvelle | candidat 38 | candidat 39 | meilleur des deux candidats par image (plafond)")
for lab, k in (("toutes", np.ones(n, bool)), ("pleines", full), ("partielles", ~full), ("rectangles < 64 px", rect & (B[:, 1] < 64)), ("rectangles >= 64 px", rect & (B[:, 1] >= 64)), ("objets", obj)):
    print(f"  {lab:20s} n={k.sum():3d} : {so[k].mean():.3f} | {sn[k].mean():.3f} | {s38[k].mean():.3f} | {s39[k].mean():.3f} | {np.maximum(s38, s39)[k].mean():.3f} ; 39 mieux / pires que 38 : {(s39 > s38 + 1e-9)[k].sum()} / {(s39 < s38 - 1e-9)[k].sum()} ; écart moyen quand mieux {np.mean((s39 - s38)[k & (s39 > s38 + 1e-9)]) if (k & (s39 > s38 + 1e-9)).any() else 0:+.3f}, quand pire {np.mean((s39 - s38)[k & (s39 < s38 - 1e-9)]) if (k & (s39 < s38 - 1e-9)).any() else 0:+.3f}")
d = s39 - s38; j = np.argsort(d); print("  cinq plus fortes baisses du 39 : " + " ; ".join(f"{names[i]} {d[i]:+.2f} ({'rect' if rect[i] else 'objet' if obj[i] else 'pleine'}, zone {RARE[i].mean():.1%}, carte nouvelle {sn[i]:.2f})" for i in j[:5]))
print("  cinq plus fortes hausses : " + " ; ".join(f"{names[i]} {d[i]:+.2f} ({'rect' if rect[i] else 'objet' if obj[i] else 'pleine'}, zone {RARE[i].mean():.1%})" for i in j[-5:]))
# règles sans vérité : accord du masque avec une carte = moyenne de la probabilité que chaque case soit du bon côté
def agree(M, Z): p = 1 / (1 + np.exp(-Z)); return np.where(M, p, 1 - p).reshape(n, -1).mean(1)
def soft_iou(M, Z):
    p = 1 / (1 + np.exp(-Z)); M = M.reshape(n, -1).astype(np.float32); p = p.reshape(n, -1); i1 = (M * p).sum(1) / np.maximum((M + p - M * p).sum(1), 1e-6); i0 = ((1 - M) * (1 - p)).sum(1) / np.maximum(((1 - M) + (1 - p) - (1 - M) * (1 - p)).sum(1), 1e-6); return (i1 + i0) / 2
pf = (LN > 0).reshape(n, -1).mean(1) > 0.5; diff = (C38 != C39).reshape(n, -1).any(1)
F = {"accord avec la carte nouvelle": agree(C39, LN) - agree(C38, LN), "mIoU espéré sous la carte nouvelle": soft_iou(C39, LN) - soft_iou(C38, LN), "mIoU espéré sous la moyenne des deux cartes": soft_iou(C39, (LN + LO) / 2) - soft_iou(C38, (LN + LO) / 2)}
def ci(x): b = x[rng.integers(0, len(x), (20000, len(x)))].mean(1); return np.percentile(b, 2.5), np.percentile(b, 97.5)
print(f"  images où les deux candidats diffèrent : {diff.sum()} ; plafond du choix par image : {np.maximum(s38, s39).mean():.4f} (38 : {s38.mean():.4f}, 39 : {s39.mean():.4f})")
for lab, f in F.items():
    H = np.zeros(n)
    for k in range(5):
        fit = fold != k; ev = fold == k; best = max(np.quantile(f[fit & diff], np.linspace(0, 1, 21)).tolist() + [1e9, -1e9], key=lambda t: np.where(f[fit] > t, s39[fit], s38[fit]).mean()); H[ev] = np.where(f[ev] > best, s39[ev], s38[ev])
    lo_, hi_ = ci(H - s38); lo2, hi2 = ci(H - s39); print(f"  choix par « {lab} » (seuil appris sans le pli) : {H.mean():.4f} ; contre 38 : {(H - s38).mean():+.4f} IC95 [{lo_:+.4f} ; {hi_:+.4f}] ; contre 39 : {(H - s39).mean():+.4f} [{lo2:+.4f} ; {hi2:+.4f}] ; réserve contre 38 : {(H - s38)[fold == 0].mean():+.4f}")
# combinaison fixe : pleines du 38, partielles du 39 (population prédite par la carte)
Hs = np.where(pf, s38, s39); lo_, hi_ = ci(Hs - s38); print(f"  pleines prédites du 38 + partielles prédites du 39 : {Hs.mean():.4f} ; contre 38 : {(Hs - s38).mean():+.4f} IC95 [{lo_:+.4f} ; {hi_:+.4f}] ; réserve {(Hs - s38)[fold == 0].mean():+.4f}")
