"""Incertitude réelle d'un écart entre deux candidats (masques honnêtes) : bootstrap apparié réserve / hors réserve / 510, écart public attendu
(306 images), décomposition du gain par filigrane x population, concentration."""
import sys
import numpy as np
from verif import HERE, METHODS, miou_vec, load_truth
N, R = sys.argv[1], sys.argv[2]; PUB = float(sys.argv[3]) if len(sys.argv) > 3 else None
rng = np.random.default_rng(20261122); A = HERE / "t1-assemblage"; meta, g = load_truth(check_png=False); idx = {m["image_name"]: i for i, m in enumerate(meta)}; bym = {m["image_name"]: m for m in meta}
def lm(c, me):
    z = np.load(A / f"candidat_{c}_honnete_{me}.npz"); return {str(a): m.astype(bool) for a, m in zip(z["names"], z["masks"])}
d = {}; res = {}; full = {}
for me in METHODS:
    cN, cR = lm(N, me), lm(R, me); nn = sorted(cN); G = np.stack([g[a] for a in nn]); d[me] = miou_vec(np.stack([cN[a] for a in nn]), G) - miou_vec(np.stack([cR[a] for a in nn]), G)
    res[me] = np.array([idx[a] % 5 == 0 for a in nn]); full[me] = np.array([bym[a]["full_watermark"] for a in nn])
def bs(sel, n=None, B=20000):
    out = np.zeros(B)
    for me in METHODS:
        x = d[me][sel[me]]; k = n or len(x); out += x[rng.integers(0, len(x), (B, k))].mean(1) / 3
    return out
ALL = {me: np.ones(len(d[me]), bool) for me in METHODS}; SETS = {"réserve (102)": res, "hors réserve (408)": {m: ~res[m] for m in METHODS}, "510 images": ALL}
print(f"=== candidat {N} − candidat {R}")
E = {}
for lab, sel in SETS.items():
    b = bs(sel); m = np.mean([d[me][sel[me]].mean() for me in METHODS]); E[lab] = (m, b.std())
    print(f"{lab:20s} écart {m:+.4f} ; erreur type {b.std():.4f} ; IC95 [{np.percentile(b, 2.5):+.4f} ; {np.percentile(b, 97.5):+.4f}] ; par filigrane " + " / ".join(f"{d[me][sel[me]].mean():+.4f}" for me in METHODS))
# écart public attendu : 306 images, ~102 par filigrane, tirées dans la loi des 510 écarts
bp = bs(ALL, n=102); sp = (bp - bp.mean()).std(); print(f"bruit d'échantillonnage d'un écart apparié sur 306 images publiques (loi des 510 écarts) : erreur type {sp:.4f}")
if PUB is not None:
    for lab in E:
        m, se = E[lab]; tot = np.hypot(se, sp); print(f"   écart public observé {PUB:+.4f} contre {m:+.4f} prédit par {lab} : différence {PUB - m:+.4f}, soit {(PUB - m) / tot:+.1f} erreurs types ({tot:.4f})")
# décomposition
print("contribution à la moyenne des trois, par filigrane x population (réserve | hors réserve) :")
for me in METHODS:
    for l2, k in (("pleines", full[me]), ("partielles", ~full[me])):
        o = []
        for sel in (res[me], ~res[me]):
            x = np.where(k[sel], d[me][sel], 0.0) / 3; b = x[rng.integers(0, len(x), (20000, len(x)))].mean(1); o.append(f"{x.mean():+.4f} [{np.percentile(b, 2.5):+.4f} ; {np.percentile(b, 97.5):+.4f}] ({(k & sel).sum()} images, {(np.abs(d[me][k & sel]) > 1e-9).sum()} changées)")
        print(f"   {me:10s} {l2:10s} {o[0]} | {o[1]}")
# concentration sur la réserve
c = np.concatenate([d[me][res[me]] / (3 * res[me].sum()) for me in METHODS]); tot = c.sum(); s = np.sort(c)[::-1]
print(f"réserve : gain total {tot:+.4f} ; les 5 images les plus favorables portent {s[:5].sum():+.4f} ({s[:5].sum() / tot:.0%}), les 10 : {s[:10].sum():+.4f} ; les 5 plus défavorables {s[-5:].sum():+.4f} ; images avec |écart| > 0,05 : {(np.abs(np.concatenate([d[me][res[me]] for me in METHODS])) > 0.05).sum()} sur 102")
