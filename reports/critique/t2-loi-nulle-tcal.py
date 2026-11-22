"""Temperature et vraisemblance de chaque guichet face aux faux endpoints « copie » tires a T = 0,70 (50 reponses).
Une copie en clair doit avoir la temperature par defaut de la famille et une NLL par token dans la loi des copies. Usage : tcal.py fichiers self50_*.json"""
import sys, json, statistics
real, fk = {}, {}
for p in sys.argv[1:]:
    try: R = json.load(open(p))
    except Exception: continue
    TS = R["TS"]; n = R["ntok"]; d = TS[1] - TS[0]
    for e, v in R["base"].items():
        i = max(range(len(v)), key=lambda k: v[k]); T, l = TS[i], v[i]
        if 0 < i < len(v) - 1:
            x, y, z = v[i - 1], v[i], v[i + 1]; q = x - 2 * y + z
            if q < 0: s = 0.5 * (x - z) / q; T, l = TS[i] + s * d, y - 0.25 * (x - z) * s
        (fk if e.startswith("fk-") else real)[e] = (T, -l / n[e])
for T0 in ("0.70", "0.80"):
    f = [v for e, v in fk.items() if e.split("-")[1] == T0]
    if len(f) < 5: continue
    mT, sT = statistics.mean(x[0] for x in f), statistics.pstdev(x[0] for x in f); mN, sN = statistics.mean(x[1] for x in f), statistics.pstdev(x[1] for x in f)
    print(f"faux endpoints tires a T={T0} ({len(f)}) : T* lue {mT:.3f} +- {sT:.3f} ; NLL/token {mN:.3f} +- {sN:.3f}")
    if T0 == "0.70":
        for e, (T, N) in sorted(real.items(), key=lambda kv: kv[1][0]):
            print(f"  {e}  T*={T:.3f} (z {(T - mT) / sT:+5.1f})  NLL/token={N:.3f} (z {(N - mN) / sN:+5.1f})")
