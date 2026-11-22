"""Ecart apparie d'un candidat contre la base, avec et sans TrustMark (demande du verificateur, 22/11 4 h 34).
Validation croisee 5 plis (candidat_<n>_cv_<filigrane>.npz, 170 images par filigrane) et reserve (index % 5 == 0).
"Sans TrustMark" : l'ecart TrustMark est compte a 0 dans la moyenne des trois filigranes. "TrustMark sur la reserve" : son ecart est celui des 39 images de reserve.
Usage : python ecart.py 39 38"""
import sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); from t1lib import *
n, ref = sys.argv[1], sys.argv[2]; rng = np.random.default_rng(0); D = {}; R = {}
for m in METHODS:
    rows = {r["image_name"]: r for r in load_train(m)}; a = np.load(ME / f"candidat_{n}_cv_{m}.npz"); b = np.load(ME / f"candidat_{ref}_cv_{m}.npz"); names = a["names"].tolist(); assert names == b["names"].tolist()
    G = np.stack([rows[k]["gt"] for k in names]); res = np.array([rows[k]["reserve"] for k in names]); d = miou(a["masks"], G) - miou(b["masks"], G); D[m] = d; R[m] = d[res]
    print(f"{m:10s} ecart en validation croisee {d.mean():+.4f} ({int((d > 1e-6).sum())} mieux, {int((d < -1e-6).sum())} pires, {int((np.abs(d) <= 1e-6).sum())} identiques) | sur les {int(res.sum())} images de reserve {d[res].mean():+.4f}")
bs = lambda x: np.array([rng.choice(x, len(x)).mean() for _ in range(4000)]) if len(x) else np.zeros(4000)
B = {m: bs(D[m]) for m in METHODS}; BR = {m: bs(R[m]) for m in METHODS}; ic = lambda v: f"IC95 [{np.quantile(v, .025):+.4f} ; {np.quantile(v, .975):+.4f}]"
tot = sum(D[m].mean() for m in METHODS) / 3; print(f"moyenne des trois, avec TrustMark en validation croisee : {tot:+.4f}, {ic(sum(B.values()) / 3)}")
o = [m for m in METHODS if m != "trustmark"]; print(f"moyenne des trois, sans TrustMark (ecart TrustMark compte 0) : {sum(D[m].mean() for m in o) / 3:+.4f}, {ic(sum(B[m] for m in o) / 3)}")
print(f"moyenne des trois, TrustMark compte sur la reserve seulement : {(sum(D[m].mean() for m in o) + R['trustmark'].mean()) / 3:+.4f}, {ic((sum(B[m] for m in o) + BR['trustmark']) / 3)}")
print(f"reserve seule, trois filigranes : {sum(R[m].mean() for m in METHODS) / 3:+.4f}, {ic(sum(BR.values()) / 3)}")
