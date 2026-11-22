#!/usr/bin/env python
"""Puissance de la règle fp8 « z_e4m3 + z_W8A8 >= 5 » à 25 réponses, sur jumeaux Qwen (fp8 simulés de vérité connue).
Entrée : sorties de ep_readers.py (res_*.json) + *_truth.json écrits par launch_fp8tw.sh. Usage : python np2_fp8power.py fp8tw/ q3 | q15
"""
import json, sys, glob, collections
import numpy as np
d, short = sys.argv[1], sys.argv[2]
rows = []
for res in sorted(x for x in glob.glob("%s/res_%s_*.json" % (d, short)) if not x.endswith("_labels.json")):
    tag = res.split("res_")[1][:-5]
    R = json.load(open(res)); truth = json.load(open("%s/%s_truth.json" % (d, tag)))
    o = list(R["families"].values())[0]
    for e in o["endpoints"]:
        z1 = o["z"]["e4m3_tensor"].get(e); z2 = o["z"]["w8a8_tensor"].get(e)
        if z1 is None or z2 is None: continue
        v, kind, T = truth[e]
        rows.append(dict(world=tag, gid=e, var=v, kind=kind, T=T, z1=z1, z2=z2, s=z1 + z2))
print(short, len(rows), "guichets", dict(collections.Counter(r["var"] if not r["var"].startswith("lora") else "lora" for r in rows)))
base = [r for r in rows if r["var"] == "base"]
print("copies base : somme z  médiane %.2f  p95 %.2f  max %.2f ; au-dessus de 5 : %d / %d" % (
    np.median([r["s"] for r in base]), np.percentile([r["s"] for r in base], 95), max(r["s"] for r in base),
    sum(r["s"] >= 5 for r in base), len(base)))
for v in ("fp8", "fp8wa", "fp8ch", "fp8e5", "fp8all", "int8", "nf4"):
    P = [r for r in rows if r["var"] == v]
    if P: print("  %-7s n=%d  somme z médiane %.2f  [min %.1f, max %.1f]  règle (>=5) : %d/%d   z_e4m3 médian %.2f  z_W8A8 médian %.2f" % (
        v, len(P), np.median([r["s"] for r in P]), min(r["s"] for r in P), max(r["s"] for r in P), sum(r["s"] >= 5 for r in P), len(P),
        np.median([r["z1"] for r in P]), np.median([r["z2"] for r in P])))
fp = [r for r in base if r["s"] >= 5]
print("faux positifs (copies isolées par la règle) :", [(r["gid"].split("|")[-1], r["kind"], round(r["T"], 2), round(r["s"], 1)) for r in fp])
lo = [r for r in rows if r["var"].startswith("lora")]
if lo: print("LoRA : somme z médiane %.2f, au-dessus de 5 : %d/%d" % (np.median([r["s"] for r in lo]), sum(r["s"] >= 5 for r in lo), len(lo)))
