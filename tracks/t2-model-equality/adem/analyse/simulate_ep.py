#!/usr/bin/env python
"""Fabrique un faux jeu de réponses « ep-01..ep-N » à partir des mondes locaux, au budget réel, avec sa vérité.

Chaque endpoint simulé reçoit exactement le plan de 100 réponses de probes/plan.json : ronde 0 (4 sondes x 2), puis
plan déterministe ou plan échantillonné selon ce que la ronde 0 montre.
Usage : python simulate_ep.py --world ../../world3 --regime G --fams olmo,qwen --n 33 --seed 0 \
            --plan ../probes/plan.json --out sim_answers.jsonl --truth sim_truth.csv [--structure ref|mix]
"""
import argparse, csv, glob, json, os, random
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True)
    ap.add_argument("--regime", default="G")
    ap.add_argument("--fams", default="olmo,qwen")
    ap.add_argument("--n", type=int, default=33)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--structure", choices=["ref", "mix"], default="mix")
    ap.add_argument("--prefix", default="ep")
    ap.add_argument("--out", required=True)
    ap.add_argument("--truth", required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    plan = json.load(open(a.plan, encoding="utf-8"))
    data, truth = defaultdict(lambda: defaultdict(dict)), {}
    for fam in a.fams.split(","):
        for f in glob.glob(os.path.join(a.world, f"{fam}_{a.regime}_*.jsonl")):
            for l in open(f, encoding="utf-8"):
                r = json.loads(l)
                data[r["endpoint"]][r["id"]][r["repeat"]] = r
        for f in glob.glob(os.path.join(a.world, f"{fam}_{a.regime}_*_truth.csv")):
            for l in open(f).read().splitlines():
                if l and not l.startswith("endpoint"):
                    ep, fa, model, kind = l.split(",")
                    truth[ep] = (fa, model, kind)
    if a.structure == "ref":
        chosen = []
        for fam in a.fams.split(","):
            q = rng.choice(["fp8", "int8", "nf4"])
            chosen += [f"{fam}-base-plain", f"{fam}-base-sys", f"{fam}-base-persona", f"{fam}-base-wm", f"{fam}-base-t1",
                       f"{fam}-{q}-plain", f"{fam}-lora0.3-plain", f"{fam}-lora0.7-plain", f"{fam}-lora1.5-plain"]
        chosen = [e for e in chosen if e in data]
    else:
        chosen = rng.sample(sorted(e for e in data if e in truth), min(a.n, len(data)))
    rng.shuffle(chosen)
    with open(a.out, "w", encoding="utf-8") as fo, open(a.truth, "w", newline="") as ft:
        w = csv.writer(ft)
        w.writerow(["endpoint", "family", "model", "kind", "source"])
        for i, src in enumerate(chosen):
            ep = f"{a.prefix}-{i + 1:02d}"
            d = data[src]
            nondet = any(d[p][0]["response"].strip() != d[p][1]["response"].strip() for p in plan["ronde0"] if p in d)
            todo = {p: 2 for p in plan["ronde0"]}
            if nondet:
                todo.update(plan["echantillonne"])
            else:
                todo.update({p: max(todo.get(p, 0), 1) for p in plan["deterministe"]})
            n = 0
            for p, reps in todo.items():
                for k in range(reps):
                    if p in d and k in d[p]:
                        r = dict(d[p][k])
                        r["endpoint"] = ep
                        fo.write(json.dumps(r, ensure_ascii=False) + "\n")
                        n += 1
            w.writerow([ep, truth[src][0], truth[src][1], truth[src][2], src])
            assert n <= 100, (src, n)
    print(len(chosen), "endpoints simulés ->", a.out)


if __name__ == "__main__":
    main()
