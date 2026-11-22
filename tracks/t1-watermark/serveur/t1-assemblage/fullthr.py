"""Images pleines : que vaudrait un seuil choisi PAR IMAGE au lieu d'un seuil commun ? (22/11 5 h 50)
Protocole a 5 plis comme cv5.py, sur les images predites pleines. Trois chiffres par carte :
  commun  : recette actuelle (lissage, seuil commun, plancher), reglee sur les 4 plis d'apprentissage ;
  plafond : meilleur seuil par image choisi avec la verite (borne haute de toute regle de seuil par image) ;
  predit  : part de cases propres predite par une regression ridge sur des statistiques de la carte (histogramme cumule, moyenne, ecart-type,
            justesse du decodeur), apprise sur les 4 plis ; le masque propre = les cases les plus basses, dans cette proportion.
Usage : python fullthr.py pixelseal=reelp3/pred trustmark=reelx3/pred"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); import asm; from asm import *
BASES = dict(kv.split("=") for kv in sys.argv[1:]); OPT.update(rect=False, snap=False, only="|".join(v.split("/")[0] for v in BASES.values()), skip="rang", boxes="xxx")
tr, te, S, notes = discover(); ztr = np.load(ME / "feat_train.npz"); ACC = dict(zip(ztr["names"].tolist(), ztr["acc"])); LV = np.linspace(-3, 1, 9)
def stats(Zs, acc):
    f = Zs.reshape(len(Zs), -1); return np.concatenate([np.stack([(f < v).mean(1) for v in LV], 1), f.mean(1, keepdims=True), f.std(1, keepdims=True), np.quantile(f, [0.01, 0.05, 0.2], axis=1).T, acc[:, None]], 1)
def by_frac(Zs, fr):
    f = Zs.reshape(len(Zs), -1); t = np.array([np.quantile(f[i], min(max(fr[i], 0.0), 0.6)) for i in range(len(f))]); return Zs > t[:, None, None]
FR = np.concatenate([[0.0], np.geomspace(0.001, 0.6, 40)])
for m, base in BASES.items():
    k0 = [k for k in S[m] if k.endswith(base)][0]; rows = tr[m]; names = [r["image_name"] for r in rows]; G = np.stack([r["gt"] for r in rows]); fold = np.array([r["fold"] for r in rows]); full_true = np.array([r["full"] for r in rows]); acc = np.array([ACC[n] for n in names]).astype(np.float32)
    n = len(rows); a = np.full(n, np.nan); o = a.copy(); pr = a.copy(); tf = a.copy(); pfr = a.copy()
    for f in range(5):
        fit = fold != f; ev = fold == f; cands = np.unique(acc); th = max(cands, key=lambda t: (np.mean((acc[fit] > t) == full_true[fit]), -abs(t - 0.75))); pf = acc > th; fi = pf & fit; ei = pf & ev
        Z = (S[m][k0][0] - S[m][k0][0][fit].mean()) / (S[m][k0][0][fit].std() + 1e-6); p, _ = tune_seuil(Z[fi], G[fi], True); a[ei] = miou(base_mask(Z[ei], p, True), G[ei])
        Zs = smooth(Z, p["sigma"]); sc = np.stack([miou(by_frac(Zs[pf], np.full(int(pf.sum()), q)), G[pf]) for q in FR], 1); bestq = FR[sc.argmax(1)]; idx = np.where(pf)[0]; pos = {j: c for c, j in enumerate(idx)}
        o[ei] = sc.max(1)[[pos[j] for j in np.where(ei)[0]]]; tf[ei] = bestq[[pos[j] for j in np.where(ei)[0]]]
        X = stats(Zs, acc); mu, sd = X[fi].mean(0), X[fi].std(0) + 1e-6; Xn = (X - mu) / sd; y = np.sqrt(bestq[[pos[j] for j in np.where(fi)[0]]]); best = None
        for lam in (0.3, 1, 3, 10, 30, 100):                                                       # penalite choisie par validation croisee interne sur les 4 plis
            e = []
            for g in sorted(set(fold[fi])):
                A = fi & (fold != g); B_ = fi & (fold == g); ya = np.sqrt(bestq[[pos[j] for j in np.where(A)[0]]]); w = np.linalg.solve(Xn[A].T @ Xn[A] + lam * np.eye(Xn.shape[1]), Xn[A].T @ (ya - ya.mean()))
                e.append(miou(by_frac(Zs[B_], np.clip(Xn[B_] @ w + ya.mean(), 0, 1) ** 2), G[B_]))
            s = float(np.concatenate(e).mean())
            if best is None or s > best[0]: best = (s, lam)
        w = np.linalg.solve(Xn[fi].T @ Xn[fi] + best[1] * np.eye(Xn.shape[1]), Xn[fi].T @ (y - y.mean())); fr = np.clip(Xn[ei] @ w + y.mean(), 0, 1) ** 2; pr[ei] = miou(by_frac(Zs[ei], fr), G[ei]); pfr[ei] = fr
        print(f"{m} pli {f} : n={int(ei.sum())} commun {a[ei].mean():.4f} | plafond {o[ei].mean():.4f} | predit {pr[ei].mean():.4f} (lambda {best[1]}, interne {best[0]:.4f}) | recette {p}", flush=True)
    k = ~np.isnan(a); rng = np.random.default_rng(0); d = np.zeros(n); d[k] = pr[k] - a[k]; boot = [rng.choice(d, n).mean() for _ in range(4000)]
    print(f"{m} {base} · {int(k.sum())} images predites pleines : seuil commun {a[k].mean():.4f} | plafond seuil par image {o[k].mean():.4f} | part propre predite {pr[k].mean():.4f} | correlation part predite / part optimale {np.corrcoef(pfr[k], tf[k])[0, 1]:.2f}")
    print(f"   ecart predit - commun sur les {n} images : {d.mean():+.4f}, IC95 [{np.quantile(boot, .025):+.4f} ; {np.quantile(boot, .975):+.4f}] ; sur la moyenne des trois : {d.mean() / 3:+.4f} | plafond sur la moyenne des trois : {(o[k] - a[k]).sum() / n / 3:+.4f}", flush=True)
