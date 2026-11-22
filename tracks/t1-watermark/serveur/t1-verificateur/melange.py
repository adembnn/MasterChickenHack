"""Mélange appris des cartes PixelSeal de t1-reel (piste « jamais essayée » n° 12) : au lieu d'une moyenne choisie à la main, une régression logistique
par case sur les logits des 5 modèles (et leur version lissée), apprise sur 4 plis d'images (index % 5) et appliquée au cinquième. Seuil 0.
Comparé à la carte livrée ps_pop_x2+x2b+x1+dx__st+x2 et à la moyenne simple."""
import numpy as np
from pathlib import Path
from scipy import ndimage as ndi
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from verif import HERE, miou_vec, load_truth
rng = np.random.default_rng(20261122); R = HERE / "t1-reel"; C = Path("comp"); meta, g = load_truth(check_png=False); idx = {m["image_name"]: i for i, m in enumerate(meta)}; bym = {m["image_name"]: m for m in meta}
def L(p):
    z = np.load(p); return {str(n): l.astype(np.float32) for n, l in zip(z["names"], z["logits"])}
TAGS = ("f_x2", "f_x2b", "f_x1", "f_dx", "f_st"); K = {t: L(C / f"{t}_train_pixelseal.npz") for t in TAGS}; NEW = L(R / "ps_pop_x2+x2b+x1+dx__st+x2" / "pred_train_pixelseal.npz"); names = sorted(NEW); n = len(names)
G = np.stack([g[a] for a in names]); X = np.stack([np.stack([K[t][a] for a in names]) for t in TAGS], -1)             # n x 128 x 128 x 5
full = np.array([bym[a]["full_watermark"] for a in names]); fold = np.array([idx[a] % 5 for a in names]); res = fold == 0
XS = np.stack([np.stack([ndi.gaussian_filter(X[i, :, :, k], 2) for k in range(len(TAGS))], -1) for i in range(n)])
frac = (X.mean(-1) > 0).reshape(n, -1).mean(1); PF = np.broadcast_to((frac > 0.5)[:, None, None, None], (n, 128, 128, 1)).astype(np.float32)   # population prédite, connue au test
def ci(x): b = x[rng.integers(0, len(x), (20000, len(x)))].mean(1); return np.percentile(b, 2.5), np.percentile(b, 97.5)
sNEW = miou_vec(np.stack([NEW[a] for a in names]) > 0, G); sAVG = miou_vec(X.mean(-1) > 0, G)
def report(lab, s):
    d = s - sNEW; lo, hi = ci(d); print(f"  {lab:46s} hors réserve {s[~res].mean():.4f} | réserve {s[res].mean():.4f} | 170 : {s.mean():.4f} | contre la carte livrée {d.mean():+.4f} IC95 [{lo:+.4f} ; {hi:+.4f}] ; pleines {d[full].mean():+.4f}, partielles {d[~full].mean():+.4f} ; réserve {d[res].mean():+.4f}")
print("PixelSeal, seuil 0"); report("carte livrée (composition choisie à la main)", sNEW); report("moyenne simple des 5", sAVG)
def cvfit(F, make, by_pop):
    P = np.zeros((n, 128, 128), np.float32)
    for k in range(5):
        tr = np.where(fold != k)[0]; ev = np.where(fold == k)[0]
        for pop in ((True, False) if by_pop else (None,)):
            a = tr if pop is None else tr[(frac[tr] > 0.5) == pop]; e = ev if pop is None else ev[(frac[ev] > 0.5) == pop]
            if len(e) == 0: continue
            Xa = F[a].reshape(-1, F.shape[-1]); ya = G[a].reshape(-1); sub = rng.choice(len(ya), min(len(ya), 400000), replace=False); m = make().fit(Xa[sub], ya[sub]); P[e] = m.decision_function(F[e].reshape(-1, F.shape[-1])).reshape(len(e), 128, 128)
    return P
lr = lambda: LogisticRegression(C=1.0, max_iter=300); gb = lambda: HistGradientBoostingClassifier(max_iter=150, max_depth=6, learning_rate=0.1)
for lab, F, make, bp in (("régression logistique, 5 logits, par population", X, lr, True), ("régression logistique, 5 logits + lissés, par population", np.concatenate([X, XS], -1), lr, True),
                         ("arbres boostés, 5 logits + lissés + population", np.concatenate([X, XS, PF], -1), gb, False)):
    P = cvfit(F, make, bp); report(lab, miou_vec(P > 0, G))
    if "5 logits, par" in lab:
        for pop in (True, False):
            a = np.where((frac > 0.5) == pop)[0]; Xa = X[a].reshape(-1, 5); ya = G[a].reshape(-1); sub = rng.choice(len(ya), 400000, replace=False); m = lr().fit(Xa[sub], ya[sub]); print(f"     poids appris sur tout le train, {'pleines' if pop else 'partielles'} prédites : " + ", ".join(f"{t} {w:+.2f}" for t, w in zip(TAGS, m.coef_[0])) + f" ; biais {m.intercept_[0]:+.2f}")
