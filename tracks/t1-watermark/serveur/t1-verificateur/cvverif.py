"""Règle de 2 h : candidat N contre R sur les masques en validation croisée complète (candidat_<n>_cv_*.npz de t1-assemblage, cv5.py),
recalculé avec mon évaluateur ; contrôles : plis = index % 5, noms, sensibilité à la référence (R honnête au lieu de R cv)."""
import sys
import numpy as np
from verif import HERE, METHODS, miou_vec, load_truth
N, R = sys.argv[1], sys.argv[2]; rng = np.random.default_rng(20261122); A = HERE / "t1-assemblage"; meta, g = load_truth(check_png=False)
idx = {m["image_name"]: i for i, m in enumerate(meta)}; bym = {m["image_name"]: m for m in meta}
def lm(c, me, kind):
    z = np.load(A / f"candidat_{c}_{kind}_{me}.npz"); names = [str(a) for a in z["names"]]
    if kind == "cv": assert all(int(f) == idx[a] % 5 for a, f in zip(names, z["fold"])), "plis != index % 5"; assert "fait" not in z or z["fait"].all()
    return {a: m.astype(bool) for a, m in zip(names, z["masks"])}
def ci(x): b = x[rng.integers(0, len(x), (20000, len(x)))].mean(1); return np.percentile(b, 2.5), np.percentile(b, 97.5)
def compare(kN, kR, lab):
    print(f"\n=== candidat {N} ({kN}) contre candidat {R} ({kR}) : {lab}")
    d = {}; sN = {}; sR = {}; res = {}
    for me in METHODS:
        cN, cR = lm(N, me, kN), lm(R, me, kR); assert set(cN) == set(cR) == {m["image_name"] for m in meta if m["method"] == me}; nn = sorted(cN); G = np.stack([g[a] for a in nn])
        sN[me] = miou_vec(np.stack([cN[a] for a in nn]), G); sR[me] = miou_vec(np.stack([cR[a] for a in nn]), G); d[me] = sN[me] - sR[me]; res[me] = np.array([idx[a] % 5 == 0 for a in nn]); full = np.array([bym[a]["full_watermark"] for a in nn])
        lo, hi = ci(d[me]); out = f"[{me:10s}] 170 images : {sN[me].mean():.4f} contre {sR[me].mean():.4f}, écart {d[me].mean():+.4f} IC95 [{lo:+.4f} ; {hi:+.4f}], mieux/pires {(d[me] > 1e-9).sum()}/{(d[me] < -1e-9).sum()}"
        for l2, k in (("pleines", full), ("partielles", ~full), ("réserve", res[me]), ("hors réserve", ~res[me])): lo, hi = ci(d[me][k]); out += f" | {l2} {sN[me][k].mean():.4f} ({d[me][k].mean():+.4f} [{lo:+.3f} ; {hi:+.3f}])"
        print(out); print("             par pli : " + " ; ".join(f"pli {f} {d[me][np.array([idx[a] % 5 == f for a in nn])].mean():+.4f}" for f in range(5)))
    for l2, sel in (("510 images", {m: np.ones(len(d[m]), bool) for m in METHODS}), ("réserve (pli 0)", res), ("hors réserve", {m: ~res[m] for m in METHODS})):
        b = np.zeros(20000)
        for me in METHODS: x = d[me][sel[me]]; b += x[rng.integers(0, len(x), (20000, len(x)))].mean(1) / 3
        print(f"MOYENNE DES TROIS, {l2:16s} : {np.mean([sN[m][sel[m]].mean() for m in METHODS]):.4f} contre {np.mean([sR[m][sel[m]].mean() for m in METHODS]):.4f}, écart apparié {np.mean([d[m][sel[m]].mean() for m in METHODS]):+.4f}, erreur type {b.std():.4f}, IC95 [{np.percentile(b, 2.5):+.4f} ; {np.percentile(b, 97.5):+.4f}], P(>= 0,02) = {np.mean(b >= 0.02):.3f}")
compare("cv", "cv", "règle de 2 h")
compare("cv", "honnete", "sensibilité : référence aux seuils ajustés sur les 408 images")
compare("honnete", "honnete", "ancien format")
