#!/usr/bin/env python
"""Familles : pour chaque endpoint, meilleure proximité avec un endpoint de sa famille et avec un endpoint d'une autre.
Si la première dépasse toujours la seconde, un regroupement par lien simple (fermeture transitive) retrouve les familles.
Usage : python explore_family.py --probes ../probes/probes_gelees.jsonl --plan ../probes/plan.json --world ../../world3
"""
import argparse, glob, itertools, json, os
from parse import load
from features import build, pair_features
from rules import SIMS, nanmean

ap = argparse.ArgumentParser()
ap.add_argument("--probes", required=True); ap.add_argument("--plan", required=True); ap.add_argument("--world", required=True)
ap.add_argument("--regime", default="G")
a = ap.parse_args()
probes = [json.loads(l) for l in open(a.probes, encoding="utf-8") if l.strip()]
plan = json.load(open(a.plan))
det = {p: 1 for p in plan["deterministe"]}; det.update({p: 2 for p in plan["ronde0"]})
samp = dict(plan["echantillonne"])
def plan_fn(name, d):
    nondet = any(len(set(t.strip() for t in d[p][:2])) > 1 for p in plan["ronde0"] if p in d)
    return (samp if nondet else det), 4
kinds, data = load(sorted(glob.glob(os.path.join(a.world, f"*_{a.regime}_*.jsonl"))), a.probes)
kinds.update({p["id"]: p["kind"] for p in probes})
eps = build(data, kinds, plan=plan_fn, prompts={p["id"]: p["prompt"] for p in probes})
names = sorted(eps)
F = {frozenset((x, y)): pair_features(eps[x], eps[y]) for x, y in itertools.combinations(names, 2)}
stats = dict(SIMS)
stats["ident_open"] = lambda f: nanmean([f["ident_open"]])
stats["pref"] = lambda f: nanmean([f["pref"]])
stats["m_word"] = lambda f: nanmean([f["m_word"]])
stats["id_cos"] = lambda f: nanmean([f["id_cos"]])
fam = lambda e: e.split("-")[0]
print(f"{len(names)} endpoints, {len(set(map(fam, names)))} familles")
print(f"{'statistique':12s} {'min(meilleur même famille)':>27s} {'max(meilleur autre famille)':>28s} {'endpoints mal reliés':>21s}   max croisé global")
for n, sf in stats.items():
    lo, hi, bad, gmax = 9, -9, 0, -9
    for e in names:
        s_in = max(sf(F[frozenset((e, o))]) for o in names if o != e and fam(o) == fam(e))
        s_out = max(sf(F[frozenset((e, o))]) for o in names if fam(o) != fam(e))
        lo, hi, bad, gmax = min(lo, s_in), max(hi, s_out), bad + (s_out >= s_in), max(gmax, s_out)
    print(f"{n:12s} {lo:27.3f} {hi:28.3f} {bad:21d}   {gmax:.3f}")

# Graphe des k plus proches voisins sur le rang moyen de 4 statistiques ; familles = composantes connexes.
import random
from learn import mcc
SIM4 = [SIMS[n] for n in ("ident", "ident+pref", "fermées", "mixte")]


def knn_families(names, F, k):
    nb = {}
    for e in names:
        others = [o for o in names if o != e]
        rank = {o: 0.0 for o in others}
        for sf in SIM4:
            for r, o in enumerate(sorted(others, key=lambda o: -sf(F[frozenset((e, o))]))):
                rank[o] += r
        nb[e] = sorted(others, key=lambda o: rank[o])[:k]
    parent = {e: e for e in names}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for e in names:
        for o in nb[e]:
            parent[find(e)] = find(o)
    return {e: find(e) for e in names}


truth = {e: fam(e) for e in names}
rng = random.Random(0)
print("familles par graphe des k plus proches voisins (rang moyen de 4 statistiques) :")
for k in (1, 2, 3, 4):
    lab = knn_families(names, F, k)
    res = []
    for _ in range(100):
        sub = rng.sample(names, 33)
        l2 = knn_families(sub, F, k)
        res.append((mcc(truth, l2, sub), len(set(l2.values()))))
    print(f"  k={k} : 80 endpoints -> {len(set(lab.values()))} composantes, MCC famille {mcc(truth, lab, names):.3f} ; "
          f"100 tirages de 33 endpoints -> MCC famille moyen {sum(r[0] for r in res) / 100:.3f}, min {min(r[0] for r in res):.3f}, "
          f"composantes {min(r[1] for r in res)} à {max(r[1] for r in res)}")

# Lien moyen sur une proximité normalisée par endpoint : s(e,o) / meilleur s(e,.) , moyenne des deux sens,
# moyennée sur 4 statistiques. Sensibilité au seuil de coupe.
from rules import linkage


def norm_sim(names, F):
    out = {}
    best = {}
    for i, sf in enumerate(SIM4):
        for e in names:
            best[(i, e)] = max(sf(F[frozenset((e, o))]) for o in names if o != e) or 1e-9
    for x, y in itertools.combinations(names, 2):
        f = F[frozenset((x, y))]
        out[frozenset((x, y))] = sum(0.5 * (sf(f) / best[(i, x)] + sf(f) / best[(i, y)]) for i, sf in enumerate(SIM4)) / len(SIM4)
    return out


ratios = []
for e in names:
    r = []
    for sf in SIM4:
        s_in = max(sf(F[frozenset((e, o))]) for o in names if o != e and fam(o) == fam(e))
        s_out = max(sf(F[frozenset((e, o))]) for o in names if fam(o) != fam(e))
        r.append(s_out / max(s_in, 1e-9))
    ratios.append(sum(r) / len(r))
ratios.sort()
print(f"rapport (meilleur autre famille) / (meilleur même famille), moyenne de 4 statistiques : médiane {ratios[len(ratios) // 2]:.2f}, "
      f"p90 {ratios[int(0.9 * len(ratios))]:.2f}, max {ratios[-1]:.2f}")
subs = [rng.sample(names, 33) for _ in range(60)]
print("lien moyen sur proximité normalisée, MCC famille selon le seuil (60 tirages de 33 endpoints) :")
for cut in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
    res = []
    for sub in subs:
        lab = linkage(sub, norm_sim(sub, F), cut)
        res.append((mcc(truth, lab, sub), len(set(lab.values()))))
    print(f"  seuil {cut:.1f} : MCC famille moyen {sum(r[0] for r in res) / len(res):.3f}, min {min(r[0] for r in res):.3f}, "
          f"nombre de familles {min(r[1] for r in res)} à {max(r[1] for r in res)}")
