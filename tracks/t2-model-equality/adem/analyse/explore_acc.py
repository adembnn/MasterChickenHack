#!/usr/bin/env python
"""Profil de capacité : taux de bonnes réponses par type d'item et par endpoint (première répétition), plus longueurs.
Usage : python explore_acc.py --probes probes_v1.jsonl world/olmo_G_*.jsonl
"""
import argparse, json, statistics
from collections import defaultdict
from parse import parse, load
from truth import truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", required=True)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    kinds, data = load(a.files, a.probes)
    prompts = {json.loads(l)["id"]: json.loads(l)["prompt"] for l in open(a.probes, encoding="utf-8") if l.strip()}
    types = ["mul", "add", "sub", "div", "fsmul", "fsadd", "cmp", "alpha", "dict", "prime", "addab"]
    print(f"{'endpoint':24s}" + "".join(f"{t:>6s}" for t in types) + "  len_open  len_list  n_seq")
    for ep in sorted(data):
        ok = defaultdict(list)
        lo, ll, ns = [], [], []
        for pid, texts in data[ep].items():
            k = kinds[pid]
            tr = truth(k, prompts[pid])
            for rep in texts:
                ans = parse(k, rep)
                if k == "q1" and ans:  # réponse verbeuse : on garde le dernier entier
                    from parse import _ints
                    ans = {0: _ints(rep)[-1]}
                for key, (t, good) in tr.items():
                    ok[t].append(ans.get(key) == good)
                if k == "open":
                    lo.append(len(rep))
                if k == "list":
                    ll.append(len(parse(k, rep)))
                if k == "seq":
                    ns.append(len(parse(k, rep)))
        row = "".join(f"{(sum(ok[t]) / len(ok[t]) if ok[t] else float('nan')):6.2f}" for t in types)
        print(f"{ep:24s}{row}  {statistics.mean(lo):8.0f}  {statistics.mean(ll):8.1f}  {statistics.mean(ns):5.0f}")


if __name__ == "__main__":
    main()
