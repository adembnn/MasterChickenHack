"""Calibration de qview : z des faux endpoints (chacun contre les autres faux de temperature voisine), par schema.
Donne le max et les quantiles de la loi nulle, et la p-valeur empirique (tous faux, tous schemas) de chaque guichet."""
import sys, json, statistics
R = json.load(open(sys.argv[1])); TS = R["TS"]; n = R["ntok"]; eps = R["eps"]
L0 = {e: max(v) for e, v in R["base"].items()}
Tst = {e: (TS[max(range(len(TS)), key=lambda i: R["base"][e][i])] if not e.startswith("fk-") else float(e.split("-")[1])) for e in R["base"]}
fk = [e for e in R["base"] if e.startswith("fk-")]
allz = []; perfake = {f: [] for f in fk}; zep = {}
for sc, P in R["fp8"].items():
    lam = {x: (max(P[x]) - L0[x]) / n[x] for x in P if x in L0}
    def z(e, excl=None):
        b = [f for f in fk if abs(Tst[f] - Tst[e]) <= 0.1001 and f in lam and f != excl]; vs = [lam[f] for f in b]
        med = statistics.median(vs); mad = statistics.median(abs(v - med) for v in vs) * 1.4826 or 1e-9
        return (lam[e] - med) / mad
    zs = [z(f, f) for f in fk if f in lam]
    for f in fk:
        if f in lam: perfake[f].append(z(f, f))
    allz += zs; zs.sort()
    print(f"{sc:12s} faux : n={len(zs)} q95={zs[int(.95 * len(zs))]:+.1f} q99={zs[int(.99 * len(zs))]:+.1f} max={zs[-1]:+.1f}")
    for e in eps: zep[(e, sc)] = z(e)
mx = sorted(max(v) for v in perfake.values() if v)
print(f"max sur les {len(R['fp8'])} schemas, par faux endpoint : q90={mx[int(.9 * len(mx))]:+.1f} q95={mx[int(.95 * len(mx))]:+.1f} q99={mx[int(.99 * len(mx))]:+.1f} max={mx[-1]:+.1f}")
for e in eps:
    best = max((zep[(e, sc)], sc) for sc in R["fp8"])
    p = (1 + sum(m >= best[0] for m in mx)) / (1 + len(mx))
    print(f"{e} meilleur schema {best[1]:12s} z={best[0]:+.1f}  p(max sur schemas, par guichet)={p:.3f}")
