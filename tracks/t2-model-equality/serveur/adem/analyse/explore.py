#!/usr/bin/env python
"""Exploration : accord par famille de sondes entre un endpoint de référence et tous les autres.

Pour qu'une statistique serve, il faut : accord(base en clair, base déguisée) > accord(base en clair, autres poids).
Accord d'une sonde = part des items dont la réponse analysée est identique (items absents d'un côté = désaccord,
items absents des deux côtés ignorés). Pour list : Jaccard des ensembles. Pour seq : part des positions égales.
Usage : python explore.py --probes probes_v1.jsonl --ref olmo-base-plain world/olmo_G_*.jsonl
"""
import argparse, sys
from collections import defaultdict
from parse import parse, load, word_set


def agree(kind, ta, tb):
    if kind in ("open", "extract"):
        return None
    if kind == "word":
        a, b = word_set(ta), word_set(tb)
        return (1.0 if a & b else 0.0) if (a or b) else None
    a, b = parse(kind, ta), parse(kind, tb)
    if kind == "list":
        sa, sb = set(a.values()), set(b.values())
        return len(sa & sb) / len(sa | sb) if sa | sb else None
    keys = set(a) | set(b)
    if not keys:
        return None
    return sum(1 for k in keys if a.get(k) is not None and a.get(k) == b.get(k)) / len(keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    kinds, data = load(a.files, a.probes)
    order = ["q1", "ab5", "nb5", "fs", "word", "list", "seq"]
    ref = data[a.ref]
    print(f"{'endpoint':26s}" + "".join(f"{k:>7s}" for k in order) + "   taux d'analyse (q1 ab5 nb5 fs)")
    for ep in sorted(data):
        acc = defaultdict(list)
        ok = defaultdict(list)
        for pid, texts in data[ep].items():
            k = kinds[pid]
            if pid not in ref:
                continue
            reps_b = texts
            reps_a = ref[pid]
            vals = [agree(k, x, y) for i, x in enumerate(reps_a) for j, y in enumerate(reps_b)
                    if not (ep == a.ref and i == j)]
            vals = [v for v in vals if v is not None]
            if vals:
                acc[k].append(sum(vals) / len(vals))
            if k in ("q1", "ab5", "nb5", "fs"):
                n = 1 if k == "q1" else 5
                ok[k].append(len(parse(k, texts[0])) / n)
        row = "".join(f"{(sum(acc[k]) / len(acc[k]) if acc[k] else float('nan')):7.2f}" for k in order)
        pr = " ".join(f"{(sum(ok[k]) / len(ok[k]) if ok[k] else 0):.2f}" for k in ("q1", "ab5", "nb5", "fs"))
        print(f"{ep:26s}{row}   {pr}")


if __name__ == "__main__":
    sys.exit(main())
