#!/usr/bin/env python
"""Régime échantillonné, décodeur « écart au consensus de la famille » (aucun endpoint de référence à élire).

Pour chaque endpoint e d'un monde : D(e, reste) compare ses réponses fermées à celles de TOUS les autres endpoints du
monde mis en commun (estimateur sans biais de la distance entre distributions, analyse/sampled.py), z = D / erreur type.
Décision : tout le monde dans le même modèle, sauf les endpoints dont z dépasse zcut, isolés chacun dans son modèle.
C'est la règle « fusionner par défaut, n'isoler que sur preuve » : elle vise les premiers points du barème du corrigé
(copies regroupées avec fp8 + LoRA 0,3 + LoRA 0,7 fusionnés = 0,33 ; LoRA 0,7 sortie = 0,52 ; fp8 seul fusionné = 0,73).
Usage : python eval_S2.py --probes ../probes/probes_S.jsonl --world ../../worldS --fams olmo,qwen [--reps 8]
"""
import argparse, glob, json, os, random, statistics
from collections import defaultdict
import numpy as np
from parse import load
from features import build
from learn import mcc
from learn2 import ref_world
import sampled


def pooled_z(world, A):
    """z de chaque endpoint contre la mise en commun des réponses des autres."""
    out = {}
    for e in world:
        pool = defaultdict(list)
        for o in world:
            if o != e:
                for k, v in A[o].items():
                    pool[k] += v
        out[e] = sampled.compare(A[e], pool)["z"]
    return out


def decode_consensus(world, A, zcut):
    z = pooled_z(world, A)
    lab, nxt = {}, 1
    for e in world:
        if z[e] == z[e] and z[e] >= zcut:
            lab[e] = nxt
            nxt += 1
        else:
            lab[e] = 0
    return lab, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", required=True)
    ap.add_argument("--world", required=True)
    ap.add_argument("--fams", default="olmo,qwen")
    ap.add_argument("--regime", default="S")
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--worlds", type=int, default=200)
    ap.add_argument("--plans", default=None, help="json {nom: {kind: nombre de sondes}} ; défaut = plans du monde worldS")
    a = ap.parse_args()
    probes = [json.loads(l) for l in open(a.probes, encoding="utf-8") if l.strip()]
    prompts = {p["id"]: p["prompt"] for p in probes}
    by = defaultdict(list)
    for p in probes:
        by[p["kind"]].append(p["id"])
    plans = json.loads(a.plans) if a.plans else {
        "12 lots A/B": {"ab5": 8, "lp": 4}, "6 A/B + 6 chiffrés": {"ab5": 6, "nb5": 6},
        "4 A/B + 4 chiffrés + 4 étiquettes": {"ab5": 4, "nb5": 4, "lp": 4}, "8 étiquettes + 4 A/B": {"lp": 8, "ab5": 4},
        "6 chiffrés + 4 un mot + 2 A/B": {"nb5": 6, "word": 4, "ab5": 2}}
    dis = lambda e: "plain" if e.endswith("plain2") else e.split("-")[-1].rstrip("2b")
    for fam in a.fams.split(","):
        files = sorted(glob.glob(os.path.join(a.world, f"{fam}_{a.regime}_*.jsonl")))
        kinds, data = load(files, a.probes)
        kinds.update({p["id"]: p["kind"] for p in probes})
        truth = {}
        for f in glob.glob(os.path.join(a.world, f"{fam}_{a.regime}_*_truth.csv")):
            for l in open(f).read().splitlines():
                if l and not l.startswith("endpoint"):
                    ep, _, model, _ = l.split(",")
                    truth[ep] = model
        wr = random.Random(1)
        names_all = sorted(data)
        worlds = [[e for e in ref_world(fam, wr) if e in data] for _ in range(a.worlds)] if not a.plans else [names_all]
        mixes = [wr.sample(names_all, min(14, len(names_all))) for _ in range(a.worlds)]
        for pname, spec in plans.items():
            only = {p: a.reps for k, n in spec.items() for p in by[k][:n]}
            eps = build(data, kinds, only=only, prompts=prompts)
            A = {e: sampled.answers(eps[e]) for e in eps}
            n_items = statistics.mean(len(v) for v in A.values())
            print(f"\n======== {fam}, régime {a.regime}, plan : {pname} x {a.reps} tirages = {sum(only.values())} réponses, "
                  f"{n_items:.0f} items fermés par endpoint ========")
            # z contre le consensus dans le monde « tout » : par type d'endpoint
            zall = pooled_z(names_all, A)
            groups = defaultdict(list)
            for e in names_all:
                w, d = e.split("-")[1], dis(e)
                key = f"base+{d}" if w == "base" else (w if d == "plain" else f"{w}+déguisé")
                groups[key].append(zall[e])
            print("z contre le consensus des " + str(len(names_all)) + " endpoints : " + "  ".join(
                f"{k}={statistics.mean(v):.1f}" for k, v in sorted(groups.items())))
            print(f"   {'zcut':>5s} {'MCC ref':>8s} {'[p10 ; p90]':>14s} {'MCC mix':>8s}   isolés à raison / à tort (mondes ref)")
            for zc in (2.0, 2.5, 3.0, 4.0):
                mr, mm, good, bad = [], [], 0, 0
                for w in worlds:
                    lab, z = decode_consensus(w, A, zc)
                    mr.append(mcc(truth, lab, w))
                    for e in w:
                        if lab[e] != 0:
                            if e.split("-")[1] == "base":
                                bad += 1
                            else:
                                good += 1
                for w in mixes:
                    lab, _ = decode_consensus(w, A, zc)
                    mm.append(mcc(truth, lab, w))
                print(f"   {zc:5.1f} {np.mean(mr):8.3f} [{np.percentile(mr, 10):.2f} ; {np.percentile(mr, 90):.2f}] {np.mean(mm):8.3f}"
                      f"   {good / len(worlds):.2f} / {bad / len(worlds):.2f} par monde")


if __name__ == "__main__":
    main()
