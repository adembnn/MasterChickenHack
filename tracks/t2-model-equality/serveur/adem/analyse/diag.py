import pickle, sys, numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from features import NAMES
feats, truth = pickle.load(open(sys.argv[1], "rb"))
gross = len(sys.argv) > 2
kind = lambda e: "plain" if e.endswith("plain2") else e.split("-")[-1].rstrip("2b")
rows = defaultdict(lambda: ([], []))
for (fam, d), F in feats.items():
    for pr, f in F.items():
        x, y = sorted(pr)
        wx, wy = x.split("-")[1], y.split("-")[1]
        if gross and wx != wy and not ({wx, wy} & {"lora0.7", "lora1.5", "nf4"}): continue
        if gross and wx != wy and ("base" not in (wx, wy)): continue
        k = "/".join(sorted((kind(x), kind(y))))
        rows[k][0].append([f[n] for n in NAMES]); rows[k][1].append(int(truth[x] == truth[y]))
cols = [n for n in NAMES if n not in ("nondet_max","nondet_min","recur_max","recur_min")]
print("%-16s %5s %5s " % ("paire", "nmeme", "ndiff") + " ".join("%7s" % c[:7] for c in cols))
for k in sorted(rows):
    X, y = np.array(rows[k][0], float), np.array(rows[k][1])
    if y.sum() == 0 or y.sum() == len(y): continue
    out = []
    for c in cols:
        v = X[:, NAMES.index(c)]; ok = ~np.isnan(v)
        out.append("%7.2f" % roc_auc_score(y[ok], v[ok]) if ok.sum() > 5 and 0 < y[ok].sum() < ok.sum() else "    nan")
    print("%-16s %5d %5d " % (k, y.sum(), len(y) - y.sum()) + " ".join(out))
