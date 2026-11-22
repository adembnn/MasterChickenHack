"""Lecteur propre a 50 reponses : gain hors echantillon (2 plis) d'une LoRA apprise sur le guichet lui-meme, compare aux faux endpoints « copie ».
Base en clair = gain dans la loi nulle. Usage : self_view.py self50_F1_0.json self50_F1_1.json ..."""
import sys, json, statistics
rows, null = [], []
for p in sys.argv[1:]:
    try: R = json.load(open(p))
    except Exception: continue
    TS = R["TS"]; n = R["ntok"]
    for e, d in R.get("self", {}).items():
        l0 = max(R["base"][e]); g = max((max(v) - l0) / n[e] for v in d.values())
        T = TS[max(range(len(TS)), key=lambda i: R["base"][e][i])]
        (null if e.startswith("fk-") else rows).append((g, e, T, -l0 / n[e]))
gs = [g for g, *_ in null]
med = statistics.median(gs) if gs else 0.0; mad = (statistics.median(abs(g - med) for g in gs) * 1.4826 or 1e-9) if gs else 1.0
print(f"loi nulle ({len(gs)} faux endpoints) : mediane {med:+.4f}, max {max(gs) if gs else 0:+.4f}")
for g, e, T, nll in sorted(rows):
    p = (1 + sum(x >= g for x in gs)) / (1 + len(gs)) if gs else float("nan")
    print(f"  {e}  T*={T:.2f}  nll/token={nll:.3f}  gain propre={g:+.4f}  z={(g - med) / mad:+5.1f}  p={p:.2f}")
