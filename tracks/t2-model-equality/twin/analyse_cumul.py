#!/usr/bin/env python
"""Analyse hors ligne de results_cumul_v2_*.json : t apparié à n=60 et n=240, t normalisé par le plancher de bruit
(Welch contre la paire « mêmes poids fp16 » de la même condition), nombre de prompts requis pour t = 3.

Usage : python analyse_cumul.py results_cumul_v2_h200.json [autre.json ...]
"""
import json, math, sys


def stats(g):
    g = [x for x in g if x == x]
    m = sum(g) / len(g)
    sd = math.sqrt(sum((x - m) ** 2 for x in g) / (len(g) - 1))
    return m, sd, len(g)


def sub(gaps, n_tot, k):
    return gaps[:k] + gaps[n_tot:n_tot + k]


def welch(a, b):
    ma, sa, na = stats(a)
    mb, sb, nb = stats(b)
    return (ma - mb) / math.sqrt(sa ** 2 / na + sb ** 2 / nb)


for path in sys.argv[1:]:
    r = json.load(open(path))
    n = r["config"]["n"]
    R = r["config"]["repeats"]
    res = r["results"]
    print(f"\n## {path}  (n={n}, repeats={R})")
    print("| variante | condition | gap moyen | sd | t (60 prompts) | t (n prompts) | Welch vs plancher (n) | prompts requis t=3 |")
    print("|---|---|---|---|---|---|---|---|")
    for cond in ["nu", "ctx", "blind"]:
        for kind, row in res.items():
            keys = ["nu"] if cond == "nu" else [f"{cond}_{i}" for i in range(R)]
            for k in keys:
                g = row[k]["gaps"]
                m, sd, N = stats(g)
                m60, sd60, N60 = stats(sub(g, n, 60))
                t60 = m60 / (sd60 / math.sqrt(N60)) if sd60 > 0 else float("nan")
                t = m / (sd / math.sqrt(N)) if sd > 0 else float("nan")
                floor = res.get("same_fp16", {}).get(k)
                w = welch(g, floor["gaps"]) if floor and kind != "same_fp16" else float("nan")
                req = math.ceil((3 * sd / m) ** 2 / 2) if m > 0 else None
                print(f"| {kind} | {k} | {m:.4f} | {sd:.3f} | {t60:.1f} | {t:.1f} | {w:.1f} | {req} |")
