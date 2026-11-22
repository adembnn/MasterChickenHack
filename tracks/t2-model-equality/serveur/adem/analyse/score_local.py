#!/usr/bin/env python
"""Score local au barème officiel : 0,1 x MCC famille (toutes les paires) + 0,9 x MCC modèle (paires d'une même
famille vraie). MCC négatif compté 0. Usage : python score_local.py soumission.csv verite.csv
"""
import csv, itertools, math, sys


def read(path):
    rows = [r for r in csv.reader(open(path, newline="", encoding="utf-8-sig")) if r and r[0] != "endpoint"]
    return {r[0]: (r[1], r[2]) for r in rows}


def mcc(pairs):
    tp = sum(t and g for t, g in pairs); fp = sum((not t) and g for t, g in pairs)
    fn = sum(t and not g for t, g in pairs); tn = sum((not t) and not g for t, g in pairs)
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (max(0.0, (tp * tn - fp * fn) / d) if d else 0.0), (tp, fp, fn, tn)


def score(mine, truth):
    eps = sorted(truth)
    fam = [(truth[a][0] == truth[b][0], mine[a][0] == mine[b][0]) for a, b in itertools.combinations(eps, 2)]
    mod = [(truth[a] == truth[b], mine[a] == mine[b]) for a, b in itertools.combinations(eps, 2)
           if truth[a][0] == truth[b][0]]
    mf, cf = mcc(fam)
    mm, cm = mcc(mod)
    return 0.1 * mf + 0.9 * mm, mf, mm, cf, cm


def main():
    mine, truth = read(sys.argv[1]), read(sys.argv[2])
    s, mf, mm, cf, cm = score(mine, truth)
    print(f"score {s:.3f} = 0,1 x MCC famille {mf:.3f} (VP,FP,FN,VN={cf}) + 0,9 x MCC modèle {mm:.3f} (VP,FP,FN,VN={cm})")


if __name__ == "__main__":
    main()
