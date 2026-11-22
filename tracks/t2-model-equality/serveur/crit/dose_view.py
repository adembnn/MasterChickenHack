"""z du gain sous la direction d, dose par dose (au lieu du max sur la dose) : explique l'ecart entre deux implementations."""
import json, sys, statistics
R = json.load(open(sys.argv[1])); d = sys.argv[2]; TS = R["TS"]; n = R["ntok"]; eps = R["eps"]
L0 = {e: max(v) for e, v in R["base"].items()}
Tst = {e: (TS[max(range(len(TS)), key=lambda i: R["base"][e][i])] if not e.startswith("fk-") else float(e.split("-")[1])) for e in R["base"]}
for s in [str(x) for x in R["scales"]]:
    P = R["dir"][d][s]; lam = {t: (max(P[t]) - L0[t]) / n[t] for t in P}
    fk = [t for t in lam if t.startswith("fk-")]
    def z(x):
        b = [f for f in fk if abs(Tst[f] - Tst[x]) <= 0.1001]; vs = [lam[f] for f in b]; med = statistics.median(vs)
        mad = statistics.median(abs(v - med) for v in vs) * 1.4826 or 1e-9; return (lam[x] - med) / mad
    print(f"direction {d} dose {s}: " + " ".join(f"{t[-2:]}={z(t):+.1f}({lam[t]:+.3f})" for t in eps if t != d))
