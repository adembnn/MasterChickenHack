#!/usr/bin/env python
"""Régime échantillonné : test à deux échantillons sur les réponses libres, contre la base en clair.
Usage : python explore_S_text.py ../probes/probes_S.jsonl ../../worldS olmo,qwen"""
import sys, glob, os, json
from parse import load
from features import build
import sampled
probes, world, fams = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
P = [json.loads(l) for l in open(probes, encoding="utf-8") if l.strip()]
for fam in fams:
    kinds, data = load(sorted(glob.glob(os.path.join(world, f"{fam}_S_*.jsonl"))), probes)
    eps = build(data, kinds, prompts={p["id"]: p["prompt"] for p in P}, only={p["id"]: 8 for p in P})
    base = eps[f"{fam}-base-plain"]
    print(f"== {fam} : z du test sur textes libres (zt), z de longueur (zlen), contre la base en clair ; {sum(1 for p in P if p['kind'] in ('open','meta'))} sondes x 2")
    for n in sorted(eps):
        if n == base.name or not (n.split("-")[1] == "base" or n.endswith("plain")):
            continue
        c = sampled.compare_text(base.texts, eps[n].texts, kinds)
        print(f"   {n[len(fam) + 1:]:16s} zt={c['zt']:6.1f}  zlen={c['zlen']:5.1f}  (n={c['nt']})")
