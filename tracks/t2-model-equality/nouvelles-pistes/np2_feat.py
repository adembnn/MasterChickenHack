#!/usr/bin/env python
"""Tour 2 : banque de z ciblés par (position, entropie) sous la base au décodeur ajusté, + évaluation AUC / précision.
Chaque trait est un z ~N(0,1) sous H0 « copie de la base au décodeur (T, p) ajusté » :
  m@bin   résidu de score  Σ(log p(y) - E log p) / sqrt(Σ Var)   (sensible à un déplacement de vraisemblance)
  u@bin   (moyenne u - 0.5) / sqrt(1/12n)                        (sensible à un décalage de rang)
  v@bin   variance de u                                           (sensible à un aplatissement / étalement)
  r0@bin  excès de rang 0 (token argmax de la base)               (la LoRA change l'ORDRE des candidats)
  bins : position {0, 1-2, 3-7, 8-31, 32+} et entropie {<0.5, 0.5-1.5, >1.5} (bits non, nats)
Score d'un guichet = max |z| sur la banque (Bonferroni empirique : seuil appris sur les jumeaux).
"""
import math, pickle, glob, sys, collections
import numpy as np

POS = [(0, 1), (1, 3), (3, 8), (8, 32), (32, 10 ** 6)]
ENT = [(0, 0.5), (0.5, 1.5), (1.5, 99)]


def zs(d):
    pos, H, u, m, v, c, e0 = d["pos"], d["H"], d["u"], d["m"], d["v"], d["c"], d["e0"]
    out = {}
    for name, bins, key in (("pos", POS, pos), ("ent", ENT, H)):
        for lo, hi in bins:
            s = (key >= lo) & (key < hi)
            n = int(s.sum())
            if n < 25: continue
            t = f"{name}[{lo},{hi})"
            out["m@" + t] = m[s].sum() / math.sqrt(max(v[s].sum(), 1e-9))
            out["u@" + t] = (u[s].mean() - 0.5) / math.sqrt(1 / 12 / n)
            out["v@" + t] = (((u[s] - 0.5) ** 2).mean() - 1 / 12) / (0.0745 / math.sqrt(n))
            ex = e0[s]
            out["r0@" + t] = ((c[s] == 0).sum() - ex.sum()) / math.sqrt(max((ex * (1 - ex)).sum(), 1e-9))
    return out


def load(paths):
    rows = []
    for p in paths:
        w = pickle.load(open(p, "rb")); fam = p.split("/")[-1].split("_")[0]
        for gid, d in w.items():
            z = zs(d); z.update(gid=gid, fam=fam, world=p.split("/")[-1][:-4], T=d["T"], p=d["p"], **{"meta_" + k: v for k, v in d["meta"].items()})
            rows.append(z)
    return rows


def label(r):
    v = r.get("meta_variant", "?")
    if v == "base": return "neg"
    if v.startswith("lora:"): return "lora" + v.split("@")[1]
    return "quant"


def auc(pos, neg):
    if not len(pos) or not len(neg): return float("nan")
    a = np.array(pos)[:, None]; b = np.array(neg)[None, :]
    return float(((a > b).sum() + 0.5 * (a == b).sum()) / (len(pos) * len(neg)))


if __name__ == "__main__":
    rows = load(sorted(glob.glob(sys.argv[1])))
    keys = sorted({k for r in rows for k in r if "@" in k})
    print(len(rows), collections.Counter(label(r) for r in rows))
    neg = [r for r in rows if label(r) == "neg"]; lor = [r for r in rows if label(r).startswith("lora")]
    res = []
    for k in keys:
        f = lambda r: abs(r.get(k, 0.0))
        res.append((auc([f(r) for r in lor], [f(r) for r in neg]), k))
    for a, k in sorted(res, reverse=True)[:14]: print("%.3f %s" % (a, k))
    # fausses alertes par déguisement : moyenne du max |z| sur la banque
    mx = lambda r: max(abs(r[k]) for k in keys if k in r)
    by = collections.defaultdict(list)
    for r in neg: by[r["meta_kind"]].append(mx(r))
    print("max|z| copies par déguisement:", {k: round(float(np.mean(v)), 1) for k, v in sorted(by.items())})
    print("max|z| LoRA:", {s: round(float(np.mean([mx(r) for r in lor if label(r) == "lora" + s])), 1) for s in ("0.3", "0.7", "1.5")})
    print("AUC max|z| banque:", [round(auc([mx(r) for r in lor if label(r) == "lora" + s], [mx(r) for r in neg]), 3) for s in ("0.3", "0.7", "1.5")])
