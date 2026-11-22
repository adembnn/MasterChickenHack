#!/usr/bin/env python
"""Exploration v2 : pour chaque neutraliseur, compare chaque endpoint à l'endpoint de référence.

Mesures (texte seul) par neutraliseur :
  ident : part des sondes dont la réponse est identique caractère pour caractère (méthode v3 de Titouan)
  pref  : longueur médiane du préfixe commun, en caractères
  q1    : part des questions chiffrées dont le nombre extrait est identique
Usage : python explore_v2.py --probes probes_v2_olmo.jsonl --ref olmo-base-plain world/olmo2_G_*.jsonl
"""
import argparse, json, statistics
from collections import defaultdict
from parse import parse, load


def lcp(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    kinds, data = load(a.files, a.probes)
    neutral = {}
    for l in open(a.probes, encoding="utf-8"):
        p = json.loads(l)
        neutral[p["id"]] = p["neutral"]
    ns = sorted(set(neutral.values()))
    ref = data[a.ref]
    print(f"{'endpoint':24s}" + "".join(f"{n + ':id':>8s}{'pref':>5s}{'q1':>6s}" for n in ns))
    for ep in sorted(data):
        row = ""
        for n in ns:
            ident, pref, q1 = [], [], []
            for pid, texts in data[ep].items():
                if neutral[pid] != n or pid not in ref:
                    continue
                pairs = [(x, y) for i, x in enumerate(ref[pid]) for j, y in enumerate(texts) if not (ep == a.ref and i == j)]
                if not pairs:
                    pairs = [(ref[pid][0], texts[0])]
                ident.append(sum(x.strip() == y.strip() for x, y in pairs) / len(pairs))
                pref.append(statistics.mean(lcp(x, y) for x, y in pairs))
                if kinds[pid] == "q1":
                    q1.append(statistics.mean(parse("q1", x).get(0, "?") == parse("q1", y).get(0, "!") for x, y in pairs))
            row += f"{statistics.mean(ident):8.2f}{statistics.median(pref):5.0f}{statistics.mean(q1):6.2f}"
        print(f"{ep:24s}{row}")


if __name__ == "__main__":
    main()
