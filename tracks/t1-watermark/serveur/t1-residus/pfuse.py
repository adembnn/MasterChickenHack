"""Fusion « carte du réseau + carte de re-tamponnage » sur les images PLEINES TrustMark. Local, numpy seul.
Carte normalisée n = rapport lissé / médiane de l'image (≈ 1 dans le filigrane, > 1 dans un trou). Règles comparées, réglées HORS réserve :
  seul   : trou si n > t (réseau ignoré)            | ou : trou si n > t OU le réseau dit trou | lin : filigrané si logit_réseau - a.ln(n) > c
Images dont la toile n'est pas trouvée (R3 global >= seuil) : réseau seul. Usage : python pfuse.py <pred_train.npz> <maps1.npz> [maps2.npz ...]"""
import sys, json, numpy as np
from pathlib import Path
R = Path("E:/EAC-lab/t1/release/train"); m128 = np.load(R / "masks_128.npz"); G = dict(zip(m128["names"].tolist(), m128["masks"]))
def miou(p, g):
    s = 0.0
    for c in (0, 1):
        u = ((p == c) | (g == c)).sum(); s += 1.0 if u == 0 else ((p == c) & (g == c)).sum() / u
    return s / 2
z = np.load(sys.argv[1], allow_pickle=True); NET = dict(zip(z["names"].tolist(), z["logits"].astype(np.float32)))
names, ratio, r3, gidx = [], [], [], []
for f in sys.argv[2:]:
    m = np.load(f, allow_pickle=True); names += m["names"].tolist(); ratio.append(m["ratio"].astype(np.float32)); r3.append(m["r3"]); gidx.append(m["gidx"])
ratio = np.concatenate(ratio); r3 = np.concatenate(r3); gidx = np.concatenate(gidx); res = gidx % 5 == 0; N = len(names)
g = np.stack([G[n] > 0 for n in names]); L = np.stack([NET[n] for n in names]); nrm = ratio / np.median(ratio, axis=(2, 3), keepdims=True); SIG = (3, 6, 10, 16, 24)
base = np.array([miou(L[i] > 0, g[i]) for i in range(N)]); ones = np.array([miou(np.ones((128, 128), bool), g[i]) for i in range(N)])
print("PFUSE n=%d pleines (%d en réserve) | réseau seul : hors réserve %.4f, RÉSERVE %.4f | tout-filigrané : %.4f / %.4f | toile trouvée (R3 < 0,6) : %d" % (N, res.sum(), base[~res].mean(), base[res].mean(), ones[~res].mean(), ones[res].mean(), (r3 < 0.6).sum()))
def evalrule(pred_fn, r3max):
    s = np.array([miou(pred_fn(i), g[i]) if r3[i] < r3max else base[i] for i in range(N)]); return s
best = {}
for r3max in (0.5, 0.6, 0.7):
    for k in range(5):
        for t in (1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.3):
            for nm, fn in (("seul", lambda i, k=k, t=t: nrm[i, k] <= t), ("ou", lambda i, k=k, t=t: (nrm[i, k] <= t) & (L[i] > 0))):
                s = evalrule(fn, r3max); key = (nm,)
                if key not in best or s[~res].mean() > best[key][0]: best[key] = (s[~res].mean(), s, "lissage %d, n > %.1f, toile R3 < %.1f" % (SIG[k], t, r3max))
        for a_ in (1, 2, 4, 8, 16):
            for c in (-4, -2, -1, 0, 1, 2):
                s = evalrule(lambda i, k=k, a_=a_, c=c: (L[i] - a_ * np.log(np.maximum(nrm[i, k], 1e-3))) > c, r3max)
                if ("lin",) not in best or s[~res].mean() > best[("lin",)][0]: best[("lin",)] = (s[~res].mean(), s, "lissage %d, a=%d, c=%d, toile R3 < %.1f" % (SIG[k], a_, c, r3max))
for key, (v, s, desc) in best.items():
    d = s - base; print("  règle %-5s (%s) : hors réserve %.4f (%+.4f ± %.4f) | RÉSERVE %.4f (%+.4f ± %.4f ; n=%d ; améliorées %d, dégradées %d)" % (key[0], desc, s[~res].mean(), d[~res].mean(), d[~res].std() / np.sqrt((~res).sum()),
        s[res].mean(), d[res].mean(), d[res].std() / np.sqrt(res.sum()), res.sum(), (d[res] > 1e-6).sum(), (d[res] < -1e-6).sum()))
# validation croisée à 5 plis sur le hors réserve pour la règle « ou » (réglage refait sur 4 plis, mesure sur le 5e)
idx = np.where(~res)[0]; fold = gidx[idx] % 5; cv = np.zeros(len(idx)); grid = [(r3max, k, t) for r3max in (0.5, 0.6, 0.7) for k in range(5) for t in (1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.3)]
S = {gq: evalrule(lambda i, k=gq[1], t=gq[2]: (nrm[i, k] <= t) & (L[i] > 0), gq[0]) for gq in grid}
for f in np.unique(fold):
    tr = idx[fold != f]; te = fold == f; gq = max(grid, key=lambda q: S[q][tr].mean()); cv[te] = S[gq][idx[te]]
print("  règle « ou », validation croisée sur le hors réserve (plis par index %% 5) : %.4f contre réseau %.4f (%+.4f ± %.4f)" % (cv.mean(), base[idx].mean(), (cv - base[idx]).mean(), (cv - base[idx]).std() / np.sqrt(len(idx))))
