#!/usr/bin/env python
"""Régime échantillonné v2 (monde worldS3 : T 0,7 + top-p 0,9, prompts système des organisateurs, 10 tirages par sonde).

A. Par famille de sondes, z de chaque endpoint contre la base en clair (test sans biais sur les distributions de
   réponses, analyse/sampled.py) : une bonne sonde donne z bas pour les copies déguisées et z haut pour les poids
   différents.
B. MCC sur des mondes de structure corrigé (base en clair, 1 prompt système, persona, filigrane, T 1,0, 1 quantification,
   LoRA 0,3 / 0,7 / 1,5), plan de 10 sondes x 10 tirages = 100 réponses, pour trois décodeurs :
   consensus (isoler si z contre le reste >= zcut), paires (lien moyen sur z < zcut), et les deux bornes du barème.
Usage : python eval_S3.py --probes ../probes/probes_S3.jsonl --world ../../worldS3 --fams olmo,qwen
"""
import argparse, glob, itertools, json, os, random, statistics
from collections import defaultdict
import numpy as np
from parse import load
from features import build
from learn import mcc
import sampled
from eval_S2 import decode_consensus


def link_z(world, Z, zcut):
    cl = [[e] for e in world]
    while len(cl) > 1:
        best, bi = None, None
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                w = statistics.mean(Z[frozenset((a, b))] for a in cl[i] for b in cl[j])
                if best is None or w < best:
                    best, bi = w, (i, j)
        if best >= zcut:
            break
        i, j = bi
        cl[i] += cl[j]
        del cl[j]
    return {e: k for k, c in enumerate(cl) for e in c}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", required=True)
    ap.add_argument("--world", required=True)
    ap.add_argument("--fams", default="olmo,qwen")
    ap.add_argument("--worlds", type=int, default=100)
    a = ap.parse_args()
    probes = [json.loads(l) for l in open(a.probes, encoding="utf-8") if l.strip()]
    prompts = {p["id"]: p["prompt"] for p in probes}
    by = defaultdict(list)
    for p in probes:
        by[p["kind"]].append(p["id"])
    plans = {"4 ab10 + 4 eq10 + 2 nb10": {"ab10": 4, "eq10": 4, "nb10": 2}, "4 eq10 + 3 nb10 + 3 ab10": {"eq10": 4, "nb10": 3, "ab10": 3},
             "4 ab10 + 4 eq10 + 2 eq5": {"ab10": 4, "eq10": 4, "eq5": 2}, "4 ab10 + 3 nb10 + 3 ab5": {"ab10": 4, "nb10": 3, "ab5": 3},
             "4 eq10 + 2 eq5 + 4 lp": {"eq10": 4, "eq5": 2, "lp": 4}}
    for fam in a.fams.split(","):
        files = sorted(glob.glob(os.path.join(a.world, f"{fam}_S_*.jsonl")))
        if not files:
            continue
        kinds, data = load(files, a.probes)
        kinds.update({p["id"]: p["kind"] for p in probes})
        names = sorted(data)
        truth = {e: e.split("-")[1] for e in names}
        base = f"{fam}-base-plain"
        short = lambda e: e[len(fam) + 1:]
        print(f"\n================ {fam} : {len(names)} endpoints, T 0,7 + top-p 0,9, 10 tirages par sonde ================")
        # taux d'analyse et A
        print("A. z contre la base en clair, par famille de sondes (toutes les sondes de la famille, 10 tirages) :")
        ks = [k for k in ("ab10", "eq10", "nb10", "ab5", "eq5", "lp") if by[k]]
        print(f"   {'endpoint':18s}" + "".join(f"{k:>7s}" for k in ks) + "   items analysés / attendus (ab10, eq10, nb10)")
        per = {}
        for k in ks:
            eps = build(data, kinds, only={p: 10 for p in by[k]}, prompts=prompts)
            A = {e: sampled.answers(eps[e]) for e in names}
            per[k] = (A, {e: sampled.compare(A[base], A[e])["z"] for e in names if e != base})
        for e in names:
            if e == base:
                continue
            ok = " ".join(f"{sum(1 for v in per[k][0][e].values() for x in v if x != '?') / (10 * 10 * len(by[k])):.2f}" for k in ("ab10", "eq10", "nb10") if by[k])
            print(f"   {short(e):18s}" + "".join(f"{per[k][1][e]:7.1f}" for k in ks) + f"   {ok}")
        # B
        wr = random.Random(1)

        def ref_world():
            w = [base, f"{fam}-base-{wr.choice(['sysacme', 'syshelp', 'sysspace'])}", f"{fam}-base-persona", f"{fam}-base-wm",
                 f"{fam}-base-t1", f"{fam}-{wr.choice(['fp8', 'int8', 'nf4'])}-plain", f"{fam}-lora0.3-plain",
                 f"{fam}-lora0.7-plain", f"{fam}-lora1.5-plain"]
            return [e for e in w if e in data]
        worlds = [ref_world() for _ in range(a.worlds)]
        for pname, spec in plans.items():
            only = {p: 10 for k, n in spec.items() for p in by[k][:n]}
            eps = build(data, kinds, only=only, prompts=prompts)
            A = {e: sampled.answers(eps[e]) for e in names}
            Z = {frozenset((x, y)): sampled.compare(A[x], A[y])["z"] for x, y in itertools.combinations(names, 2)}
            print(f"B. plan {pname} ({sum(only.values())} réponses) ; z contre la base : " + "  ".join(
                f"{short(e)}={Z[frozenset((base, e))]:.1f}" for e in names if e != base))
            print(f"   {'décodeur':34s} {'MCC ref':>8s} {'[p10 ; p90]':>14s}")
            for vn, fn in [(f"consensus, z >= {z}", (lambda w, z=z: decode_consensus(w, A, z)[0])) for z in (2.0, 3.0, 4.0)] + \
                          [(f"paires, lien moyen z < {z}", (lambda w, z=z: link_z(w, Z, z))) for z in (2.0, 3.0, 4.0)]:
                ms = [mcc(truth, fn(w), w) for w in worlds]
                print(f"   {vn:34s} {np.mean(ms):8.3f} [{np.percentile(ms, 10):.2f} ; {np.percentile(ms, 90):.2f}]")


if __name__ == "__main__":
    main()
