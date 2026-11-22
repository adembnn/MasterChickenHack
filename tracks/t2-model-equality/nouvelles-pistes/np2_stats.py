#!/usr/bin/env python
"""Tour 2 : statistiques « LoRA sans adaptateur » calculées sur les sorties de np2_extract.py (0 requête, CPU).
Chaque statistique est un score ≥ 0 : plus il est grand, plus le guichet s'écarte de « la base lue au décodeur ajusté ».
  AD     Anderson-Darling du PIT u (référence : tour 1)
  POS    PIT cumulé par position : max sur 4 tranches de position de |z(moyenne u)|, |z(variance u)|
  RANK   signature des rangs : khi² sur les comptes de rang 0 / 1 / 2 / ≥3 contre les attendus du décodeur ajusté
  MART   résidu de score m = log p(y) - E[log p] : |z| total et max sur 3 tranches d'entropie
  HBIN   PIT par tranche d'entropie : max |z(moyenne u)| sur 3 tranches
Version « PEER » : chaque statistique est recentrée sur la médiane des autres guichets de la même famille (robuste MAD).
"""
import math, pickle, sys, json, glob, collections
import numpy as np


def ad(u):
    n = len(u)
    us = np.sort(np.clip(u, 1e-9, 1 - 1e-9)); i = np.arange(1, n + 1)
    return float(-n - np.sum((2 * i - 1) * (np.log(us) + np.log(1 - us[::-1]))) / n)


def zmean(u):
    return (u.mean() - 0.5) / math.sqrt(1 / 12 / len(u)) if len(u) > 5 else 0.0


def zvar(u):
    x = (u - 0.5) ** 2
    return (x.mean() - 1 / 12) / (0.0745 / math.sqrt(len(u))) if len(u) > 5 else 0.0


def stats(d):
    u, pos, m, v, H, c = d["u"], d["pos"], d["m"], d["v"], d["H"], d["c"]
    out = {"AD": ad(u)}
    b = [(0, 4), (4, 16), (16, 64), (64, 10 ** 6)]
    out["POS"] = max(max(abs(zmean(u[(pos >= lo) & (pos < hi)])), abs(zvar(u[(pos >= lo) & (pos < hi)]))) for lo, hi in b)
    e = [d["e0"], d["e1"], d["e2"]]
    chi = 0.0
    for k in range(3):
        o = (c == k).sum(); ex = e[k].sum(); va = (e[k] * (1 - e[k])).sum()
        chi += (o - ex) ** 2 / max(va, 1e-9)
    out["RANK"] = chi
    zt = m.sum() / math.sqrt(max(v.sum(), 1e-9))
    hb = [(0, 0.5), (0.5, 1.5), (1.5, 99)]
    zb = [abs(m[(H >= lo) & (H < hi)].sum() / math.sqrt(max(v[(H >= lo) & (H < hi)].sum(), 1e-9))) for lo, hi in hb if ((H >= lo) & (H < hi)).sum() > 30]
    out["MART"] = max([abs(zt)] + zb)
    out["MART_tot"] = abs(zt)
    out["HBIN"] = max([abs(zmean(u[(H >= lo) & (H < hi)])) for lo, hi in hb if ((H >= lo) & (H < hi)).sum() > 30] or [0])
    out["T"] = d["T"]; out["p"] = d["p"]
    return out


def load(paths):
    rows = []
    for p in paths:
        w = pickle.load(open(p, "rb")); fam = p.split("/")[-1].split("_")[0]
        for gid, d in w.items():
            s = stats(d); s.update(gid=gid, fam=fam, world=p.split("/")[-1][:-4], **{"meta_" + k: v for k, v in d["meta"].items()})
            rows.append(s)
    return rows


def peer(rows, keys):
    """recentre chaque statistique sur la médiane / MAD des autres guichets du même monde (hors cibles connues ? non : aveugle)."""
    by = collections.defaultdict(list)
    for r in rows: by[r["world"]].append(r)
    for w, L in by.items():
        for k in keys:
            x = np.array([r[k] for r in L])
            for r in L:
                o = np.delete(x, L.index(r))
                med = np.median(o); mad = np.median(np.abs(o - med)) * 1.4826 + 1e-6
                r["P_" + k] = (r[k] - med) / mad


def auc(pos, neg):
    if not len(pos) or not len(neg): return float("nan")
    a = np.array(pos)[:, None]; b = np.array(neg)[None, :]
    return float(((a > b).sum() + 0.5 * (a == b).sum()) / (len(pos) * len(neg)))


def label(r):
    v = r["meta_variant"]
    if v == "base": return "neg"
    if v.startswith("lora:"): return "lora" + v.split("@")[1]
    return "quant"


if __name__ == "__main__":
    paths = sorted(glob.glob(sys.argv[1]))
    rows = load(paths)
    keys = ["AD", "POS", "RANK", "MART", "MART_tot", "HBIN"]
    peer(rows, keys)
    allk = keys + ["P_" + k for k in keys]
    print(len(rows), "guichets", collections.Counter(label(r) for r in rows))
    print("%-10s" % "stat", " ".join("%8s" % c for c in ["lora0.3", "lora0.7", "lora1.5", "all"]))
    for k in allk:
        neg = [r[k] for r in rows if label(r) == "neg"]
        a = [auc([r[k] for r in rows if label(r) == "lora" + s], neg) for s in ("0.3", "0.7", "1.5")]
        al = auc([r[k] for r in rows if label(r).startswith("lora")], neg)
        print("%-10s" % k, " ".join("%8.3f" % x for x in a + [al]))
