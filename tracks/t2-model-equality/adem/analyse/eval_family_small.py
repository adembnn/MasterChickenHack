#!/usr/bin/env python
"""Mode dégradé : combien de réponses par endpoint faut-il pour retrouver les familles seules ?

Pour plusieurs petits plans, tire 33 endpoints parmi les familles locales, estime les familles (rules.family_partition)
et note le MCC famille. Sert à décider d'un premier envoi « familles seules » qui laisse le budget presque intact.
Usage : python eval_family_small.py --probes ../probes/probes_gelees.jsonl --world ../../world3 --regime G
"""
import argparse, glob, itertools, json, os, random
from collections import defaultdict
from parse import load
from features import build, pair_features
from learn import mcc
from rules import family_partition

ap = argparse.ArgumentParser()
ap.add_argument("--probes", required=True)
ap.add_argument("--world", required=True)
ap.add_argument("--regime", default="G")
ap.add_argument("--draws", type=int, default=40)
a = ap.parse_args()
probes = [json.loads(l) for l in open(a.probes, encoding="utf-8") if l.strip()]
by = defaultdict(list)
for p in probes:
    by[p["kind"]].append(p["id"])
kinds, data = load(sorted(glob.glob(os.path.join(a.world, f"*_{a.regime}_*.jsonl"))), a.probes)
kinds.update({p["id"]: p["kind"] for p in probes})
prompts = {p["id"]: p["prompt"] for p in probes}
fam = lambda e: e.split("-")[0]
truth = {e: fam(e) for e in data}
r0 = by["ab5"][:2] + by["q1"][:2]
plans = {
    "ronde 0 seule (4 sondes x 2 = 8 réponses)": {p: 2 for p in r0},
    "ronde 0 + 3 identité + 3 extraction (14)": {**{p: 2 for p in r0}, **{p: 1 for p in by["meta"][:6]}},
    "ronde 0 + 6 meta + 4 lots A/B + 4 lots chiffrés (22)": {**{p: 2 for p in r0}, **{p: 1 for p in by["meta"][:6] + by["ab5"][2:6] + by["nb5"][:4]}},
    "ronde 0 + 6 meta + 8 A/B + 8 chiffrés + 4 textes (34)": {**{p: 2 for p in r0}, **{p: 1 for p in by["meta"][:6] + by["ab5"][2:10] + by["nb5"][:8] + by["open"][:4]}},
}
rng = random.Random(0)
names = sorted(data)
subs = [rng.sample(names, min(33, len(names))) for _ in range(a.draws)]
print(f"régime {a.regime}, {len(names)} endpoints, {len(set(truth.values()))} familles, {a.draws} tirages de 33 endpoints")
print(f"{'plan':56s} {'MCC famille':>12s} {'min':>6s}  nombre de familles estimé")
for pname, only in plans.items():
    only = {p: n for p, n in only.items() if p in prompts}
    eps = build(data, kinds, only=only, prompts=prompts)
    F = {frozenset((x, y)): pair_features(eps[x], eps[y]) for x, y in itertools.combinations(names, 2)}
    for cut in (0.4, 0.5):
        res = []
        for sub in subs:
            lab = family_partition(sub, F, cut)
            res.append((mcc(truth, lab, sub), len(set(lab.values()))))
        print(f"{pname:56s} {sum(r[0] for r in res) / len(res):12.3f} {min(r[0] for r in res):6.3f}  "
              f"{min(r[1] for r in res)} à {max(r[1] for r in res)}  (seuil {cut})")
