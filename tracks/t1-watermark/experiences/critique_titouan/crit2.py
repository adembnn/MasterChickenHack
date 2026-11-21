"""E3 : rehausser les petits creux de la carte sur les PLEINES (chapeau haut-de-forme noir sur les logits).
L' = L - a * (fermeture(L, s) - L), seuil re-regle. Reglage (a, s, t) hors reserve, CV 4 plis + reserve, comme crit1."""
import sys, numpy as np
from scipy.ndimage import grey_closing
sys.argv = sys.argv[:3] + ["logits"]
exec(open("crit1.py").read().split("base = run(")[0])

def make_th(cands):
    def f(tr, te):
        best = None
        cache = {}
        for a, s in cands:
            if s not in cache:
                cache[s] = np.stack([grey_closing(L[i], size=(s, s)) - L[i] for i in range(len(L))])
            Lp = L - a * cache[s]
            ts = np.unique(np.concatenate([np.quantile(Lp[tr], np.linspace(0.02, 0.98, 49)), [0.0]]))
            for t in ts:
                sc = np.mean([miou(Lp[i] > t, G[i]) for i in tr])
                if best is None or sc > best[0]: best = (sc, a, s, t)
        _, a, s, t = best
        print(f" a={a} s={s}", end="")
        Lp = L - a * cache[s] if a else L
        return [Lp[i] > t for i in te]
    return f

base = run(per_pop(base_thr), "base"); print()
grid = [(0.0, 9)] + [(a, s) for a in (0.5, 1.0, 2.0) for s in (5, 9, 17, 33)]
new = run(per_pop(make_th(grid)), "tophat"); print()
report(base, new, "E3 chapeau noir")
