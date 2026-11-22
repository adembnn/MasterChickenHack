#!/usr/bin/env python
"""Piste A (RUT corrigé de la température) : test de rang par vraisemblance exacte, 0 requête.
Par réponse i d'un guichet : s_i = log p_base(r_i|x)/N_i et entropie moyenne H_i de la base aux tokens générés, tous deux lus
AU DÉCODEUR AJUSTÉ du guichet (T, top-p par maximum de vraisemblance sur ses réponses). Le résidu ε_i = s_i + H_i est centré sous
« copie de la base à (T,p) ». On le standardise par sa variance exacte : z_i = Σm / sqrt(Σv) ~ N(0,1) (martingale), puis
p_i = Φ(z_i) est uniforme sous H0. Test d'uniformité : Cramér-von Mises sur n=25 réponses, loi nulle calibrée sur les guichets
« base » des jumeaux (échantillons de 25 tirés dans les 100 réponses, 200 répétitions).
Usage : python np2_rank.py "o/k100_*.pkl" [--n 25 --reps 200]
"""
import sys, glob, math, pickle, collections
import numpy as np


def phi(x):
    return 0.5 * (1 + np.vectorize(math.erf)(np.asarray(x) / math.sqrt(2)))


def per_response(d):
    r = d["r"]; out = []
    for k in np.unique(r):
        s = r == k
        v = float(d["v"][s].sum()); m = float(d["m"][s].sum())
        out.append((m / math.sqrt(max(v, 1e-9)), float(d["m"][s].mean()), float(d["H"][s].mean())))
    return np.array(out)   # colonnes : z, ε moyen par token (= s+H), H moyen


def cvm(u):
    u = np.sort(np.clip(u, 1e-9, 1 - 1e-9)); n = len(u)
    return float(1 / (12 * n) + np.sum((u - (2 * np.arange(1, n + 1) - 1) / (2 * n)) ** 2))


def auc(pos, neg):
    if not len(pos) or not len(neg): return float("nan")
    a = np.array(pos)[:, None]; b = np.array(neg)[None, :]
    return float(((a > b).sum() + 0.5 * (a == b).sum()) / (len(pos) * len(neg)))


def label(meta):
    v = meta["variant"]
    return "base" if v == "base" else ("lora" + v.split("@")[1] if v.startswith("lora:") else "fp8")


def stats_sub(z, rng, n, reps):
    """statistiques sur `reps` sous-échantillons de n réponses : CvM de Φ(z), |mean z|·√n, var-1."""
    out = []
    for _ in range(reps):
        zz = z[rng.choice(len(z), n, replace=False)]
        out.append((cvm(phi(zz)), abs(zz.mean()) * math.sqrt(n), abs(zz.var(ddof=1) - 1)))
    return np.array(out)


if __name__ == "__main__":
    pat = sys.argv[1]; n = 25; reps = 200
    if "--n" in sys.argv: n = int(sys.argv[sys.argv.index("--n") + 1])
    if "--reps" in sys.argv: reps = int(sys.argv[sys.argv.index("--reps") + 1])
    rng = np.random.default_rng(0)
    G = []
    for p in sorted(glob.glob(pat)):
        world = p.split("/")[-1][:-4]
        for gid, d in pickle.load(open(p, "rb")).items():
            z = per_response(d)[:, 0]
            if len(z) < n + 5: continue
            G.append(dict(world=world, gid=gid, lab=label(d["meta"]), kind=d["meta"]["kind"], T=d["T"], p=d["p"], z=z, S=stats_sub(z, rng, n, reps)))
    print(len(G), "guichets", collections.Counter(g["lab"] for g in G), "n=%d reps=%d" % (n, reps))
    names = ["CvM", "|mean z|", "|var-1|"]
    train = [g for g in G if "train" in g["world"]]; val = [g for g in G if "val" in g["world"]]
    for j, nm in enumerate(names):
        # seuil à 5 % de faux positifs : calé sur les copies « base » du train, appliqué à val
        nulls = np.concatenate([g["S"][:, j] for g in train if g["lab"] == "base"])
        thr = float(np.percentile(nulls, 95))
        print("== %s  seuil(95%% des copies train)=%.3f" % (nm, thr))
        for split, GG in (("train", train), ("val", val)):
            base = [g for g in GG if g["lab"] == "base"]
            for lab in ("lora0.3", "fp8"):
                pos = [g for g in GG if g["lab"] == lab]
                if not pos: continue
                a = auc(np.concatenate([g["S"][:, j] for g in pos]), np.concatenate([g["S"][:, j] for g in base]))
                tp = np.mean([np.mean(g["S"][:, j] > thr) for g in pos]); fpr = np.mean([np.mean(g["S"][:, j] > thr) for g in base])
                prec = tp * len(pos) / max(tp * len(pos) + fpr * len(base), 1e-9)
                print("  %-5s %-8s AUC %.3f  puissance %.2f  faux positifs %.2f  précision(prévalence du monde) %.2f  (n_pos=%d n_base=%d)" % (split, lab, a, tp, fpr, prec, len(pos), len(base)))
    # faux positifs par déguisement / température (statistique CvM)
    thr = float(np.percentile(np.concatenate([g["S"][:, 0] for g in train if g["lab"] == "base"]), 95))
    by = collections.defaultdict(list)
    for g in G:
        if g["lab"] == "base": by[(g["kind"], round(g["T"], 2))].append(np.mean(g["S"][:, 0] > thr))
    print("faux positifs CvM par (déguisement, T ajustée):", {k: round(float(np.mean(v)), 2) for k, v in sorted(by.items())})
