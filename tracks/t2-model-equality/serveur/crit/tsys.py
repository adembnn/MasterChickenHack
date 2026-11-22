import json, sys
tag = sys.argv[1]
R = json.load(open(f"sys2_{tag}.json")); G = json.load(open(f"gl_{tag}.json")); TS = G["TS"]
def ts(v): return TS[max(range(len(TS)), key=lambda i: v[i])]
for e in sys.argv[2].split(","):
    row = ["sans prompt T*=%.2f" % ts(G["base"][e])]
    for c in R["sys"]:
        v = R["sys"][c].get(e)
        if isinstance(v, list): row.append("%s T*=%.2f" % (c[:6], ts(v)))
    print(e, " | ".join(row))
