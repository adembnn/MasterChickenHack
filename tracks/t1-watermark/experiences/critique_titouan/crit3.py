"""E4 : la marge du decodeur officiel (image entiere) predit-elle la taille du trou des pleines ?
Si oui : seuil par image choisi pour que l'aire propre predite = aire estimee par le decodeur."""
import sys, numpy as np
from scipy.stats import spearmanr
sys.argv = sys.argv[:3] + ["logits"]
exec(open("crit1.py").read().split("base = run(")[0])
z = np.load("margins_train.npz"); MF = dict(zip(z["names"].tolist(), z["feats"]))
X = np.array([MF[n] for n in names])            # marge moyenne, acc, mediane, tanh
clean = 1 - G.reshape(len(G), -1).mean(1)
for pop, pm in (("pleines", F), ("partielles", ~F)):
    for j, nm in enumerate(("marge", "acc", "mediane", "tanh")):
        r = spearmanr(X[pm, j], clean[pm]).correlation
        print(f"{meth} {pop:<10} Spearman({nm}, part propre) = {r:+.3f}")

def area(tr, te):
    # regression lineaire part propre ~ marge, tanh (pleines), puis seuil par image au quantile predit ; melange avec seuil global
    A = np.c_[np.ones(len(tr)), X[tr][:, [0, 3]]]
    w = np.linalg.lstsq(A, clean[tr], rcond=None)[0]
    q = lambda idx: np.clip(np.c_[np.ones(len(idx)), X[idx][:, [0, 3]]] @ w, 0.003, 0.6)
    t0 = tune_thr(L, tr)
    best = None
    for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
        for mult in (0.75, 1.0, 1.5):
            def mk(idx, lam=lam, mult=mult):
                out = []
                for i, qi in zip(idx, q(idx)):
                    tq = np.quantile(L[i], min(qi * mult, 0.9))
                    out.append(L[i] > (lam * tq + (1 - lam) * t0))
                return out
            s = np.mean([miou(m, G[i]) for m, i in zip(mk(tr), tr)])
            if best is None or s > best[0]: best = (s, lam, mult, mk)
    print(f" lam={best[1]} mult={best[2]}", end="")
    return best[3](te)

def pf(tr, te):   # pleines : regle aire ; partielles : base
    res = [None] * len(te)
    for pop, fn in ((True, area), (False, base_thr)):
        pos = [j for j, i in enumerate(te) if F[i] == pop]
        for j, m in zip(pos, fn(tr[F[tr] == pop], te[pos])): res[j] = m
    return res

base = run(per_pop(base_thr), "base"); print()
new = run(pf, "aire"); print()
report(base, new, "E4 aire par decodeur")
