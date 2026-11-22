"""Lecture des sondes par filigrane : gain public observé entre deux candidats pour un filigrane, comparé à ce que prédisent la validation croisée
complète, la réserve seule (pli 0) et les plis hors réserve. Tout est exprimé en points du score GLOBAL (un filigrane pèse un tiers).
Usage : sondes.py N R gain_ps gain_tm gain_ss (gains publics observés, en points du score global)"""
import sys
import numpy as np
from verif import HERE, METHODS, miou_vec, load_truth
N, R = sys.argv[1], sys.argv[2]; OBS = dict(zip(METHODS, map(float, sys.argv[3:6]))); rng = np.random.default_rng(20261122); A = HERE / "t1-assemblage"
meta, g = load_truth(check_png=False); idx = {m["image_name"]: i for i, m in enumerate(meta)}
def lm(c, me):
    z = np.load(A / f"candidat_{c}_cv_{me}.npz"); return {str(a): m.astype(bool) for a, m in zip(z["names"], z["masks"])}
print(f"candidat {N} − candidat {R}, par filigrane, en points du score global (public : 306 images, environ 102 par filigrane)")
for me in METHODS:
    cN, cR = lm(N, me), lm(R, me); nn = sorted(cN); G = np.stack([g[a] for a in nn]); d = miou_vec(np.stack([cN[a] for a in nn]), G) - miou_vec(np.stack([cR[a] for a in nn]), G); res = np.array([idx[a] % 5 == 0 for a in nn])
    sp = d[rng.integers(0, len(d), (20000, 102))].mean(1).std() / 3                     # bruit d'une sonde : 102 images publiques de ce filigrane
    out = f"[{me:10s}] public observé {OBS[me]:+.4f} (bruit d'échantillonnage ±{1.96 * sp:.4f} à 95 %)"
    for lab, k in (("510 images", np.ones(len(d), bool)), ("réserve", res), ("hors réserve", ~res)):
        x = d[k]; m = x.mean() / 3; se = x[rng.integers(0, len(x), (20000, len(x)))].mean(1).std() / 3; z = (OBS[me] - m) / np.hypot(se, sp); out += f" | prévu par {lab} {m:+.4f} (±{1.96 * se:.4f}), écart {z:+.1f} erreurs types"
    print(out)
