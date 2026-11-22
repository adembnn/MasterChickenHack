"""Reunit les sorties par pli de cv5.py (une par processus) en candidat_<n>_cv_<filigrane>.npz et compare deux candidats en ecart apparie.
Usage : python cv5merge.py 30 [27]"""
import sys, glob, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); from t1lib import *
def merged(n, m):
    f = ME / f"candidat_{n}_cv_{m}.npz"
    if f.exists(): z = np.load(f); return z["names"].tolist(), z["masks"], z["fold"]
    parts = sorted(glob.glob(str(ME / f"candidat_{n}_cv_{m}_plis*.npz"))); assert parts, f"rien pour {n} {m}"
    z0 = np.load(parts[0]); names = z0["names"].tolist(); M = np.zeros_like(z0["masks"]); done = np.zeros(len(names), bool)
    for p in parts: z = np.load(p); k = z["fait"]; M[k] = z["masks"][k]; done |= k
    assert done.all(), f"plis manquants pour {n} {m} : {int((~done).sum())} images"
    np.savez_compressed(f, names=np.array(names), masks=M, fold=z0["fold"]); return names, M, z0["fold"]
n = sys.argv[1]; ref = sys.argv[2] if len(sys.argv) > 2 else None; rng = np.random.default_rng(0); allsc = {}; allref = {}
for m in METHODS:
    rows = {r["image_name"]: r for r in load_train(m)}; names, M, fold = merged(n, m); G = np.stack([rows[k]["gt"] for k in names]); sc = miou(M, G); full = np.array([rows[k]["full"] for k in names]); allsc[m] = sc
    line = f"{m:10s} candidat {n} : validation croisee 5 plis sur 170 images {sc.mean():.4f} (pleines {sc[full].mean():.4f}, partielles {sc[~full].mean():.4f})"
    if ref:
        nr, Mr, _ = merged(ref, m); assert nr == names; scr = miou(Mr, G); allref[m] = scr; d = sc - scr
        line += f" | contre {ref} : {d.mean():+.4f} ({int((d > 1e-6).sum())} images mieux, {int((d < -1e-6).sum())} pires)"
    print(line)
tot = np.mean([allsc[m].mean() for m in METHODS]); print(f"moyenne des trois filigranes : {tot:.4f}", end="")
if ref:
    d = [allsc[m] - allref[m] for m in METHODS]; boot = [np.mean([rng.choice(x, len(x)).mean() for x in d]) for _ in range(4000)]
    print(f" | ecart apparie contre {ref} : {np.mean([x.mean() for x in d]):+.4f}, IC95 [{np.quantile(boot, .025):+.4f} ; {np.quantile(boot, .975):+.4f}]")
else: print()
