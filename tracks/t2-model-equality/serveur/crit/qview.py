"""z de chaque schema de quantification par guichet, a 50 reponses, contre les faux endpoints de temperature voisine."""
import sys, json, statistics
R = json.load(open(sys.argv[1])); TS = R["TS"]; n = R["ntok"]; eps = R["eps"]
L0 = {e: max(v) for e, v in R["base"].items()}
Tst = {e: (TS[max(range(len(TS)), key=lambda i: R["base"][e][i])] if not e.startswith("fk-") else float(e.split("-")[1])) for e in R["base"]}
fk = [e for e in R["base"] if e.startswith("fk-")]
print("guichet  T*   " + "  ".join(f"{sc:>12s}" for sc in R["fp8"]))
for e in eps:
    row = []
    for sc, P in R["fp8"].items():
        lam = {x: (max(P[x]) - L0[x]) / n[x] for x in P if x in L0}
        b = [f for f in fk if abs(Tst[f] - Tst[e]) <= 0.1001 and f in lam]; vs = [lam[f] for f in b]
        med = statistics.median(vs); mad = statistics.median(abs(v - med) for v in vs) * 1.4826 or 1e-9
        row.append(f"{(lam[e] - med) / mad:+12.1f}")
    print(f"{e}  {Tst[e]:.2f} " + "  ".join(row))
