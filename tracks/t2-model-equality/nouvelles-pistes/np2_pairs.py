#!/usr/bin/env python
"""Tour 2, idée « entre guichets » : tous les guichets reçoivent les MÊMES prompts. Une LoRA déplace les réponses dans une
direction cohérente d'un prompt à l'autre ; une copie (déguisement de décodage) non. Pour chaque guichet on forme, par prompt,
un vecteur de traits lus sous la BASE (au décodeur ajusté du guichet), recentré sur la médiane des autres guichets de la
famille (retire l'effet du prompt), puis on corrèle les guichets deux à deux.
Traits par réponse : zm (résidu de score standardisé), zu (PIT moyen), z0 (excès de rang 0), ln (log longueur), hm (entropie moyenne).
Usage : python np2_pairs.py "o/*_val_s101.pkl" [--cols zm,zu,z0,ln,hm]
"""
import pickle, glob, sys, math, collections, itertools
import numpy as np

COLS = ["zm", "zu", "z0", "ln", "hm"]


def per_prompt(d):
    r = d["r"]; out = {c: [] for c in COLS}
    for k in np.unique(r):
        s = r == k
        v = d["v"][s]; m = d["m"][s]; u = d["u"][s]
        out["zm"].append(m.sum() / math.sqrt(max(v.sum(), 1e-9)))
        out["zu"].append((u.mean() - 0.5) / math.sqrt(1 / 12 / s.sum()))
        e0 = d["e0"][s]; o0 = (d["c"][s] == 0).sum()
        out["z0"].append((o0 - e0.sum()) / math.sqrt(max((e0 * (1 - e0)).sum(), 1e-9)))
        out["ln"].append(math.log(s.sum()))
        out["hm"].append(float(d["H"][s].mean()))
    return {c: np.array(x) for c, x in out.items()}


def world_features(w, cols):
    gids = list(w); F = {g: per_prompt(w[g]) for g in gids}
    for c in cols:
        X = np.stack([F[g][c] for g in gids])              # guichets × prompts
        med = np.median(X, 0, keepdims=True)
        X = X - med                                         # retire l'effet du prompt
        sd = X.std(1, keepdims=True) + 1e-9                # et l'échelle propre au guichet
        for i, g in enumerate(gids): F[g][c] = X[i] / sd[i]
    return gids, F


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (math.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))


def same_group(m1, m2):
    """vérité : même adaptateur LoRA (quelle que soit la force) ou même variante de poids non LoRA."""
    v1, v2 = m1["variant"], m2["variant"]
    if v1.startswith("lora:") and v2.startswith("lora:"):
        return v1.split("@")[0] == v2.split("@")[0]
    return False


def auc(pos, neg):
    if not len(pos) or not len(neg): return float("nan")
    a = np.array(pos)[:, None]; b = np.array(neg)[None, :]
    return float(((a > b).sum() + 0.5 * (a == b).sum()) / (len(pos) * len(neg)))


if __name__ == "__main__":
    paths = sorted(glob.glob(sys.argv[1]))
    cols = COLS
    if "--cols" in sys.argv: cols = sys.argv[sys.argv.index("--cols") + 1].split(",")
    res = collections.defaultdict(lambda: ([], []))   # col -> (pos corr, neg corr)
    comb = ([], [])
    for p in paths:
        w = pickle.load(open(p, "rb")); gids, F = world_features(w, cols)
        for g1, g2 in itertools.combinations(gids, 2):
            m1, m2 = w[g1]["meta"], w[g2]["meta"]
            v1, v2 = m1["variant"], m2["variant"]
            if v1 not in ("base",) and not v1.startswith("lora:") or v2 not in ("base",) and not v2.startswith("lora:"):
                continue   # quantifications exclues de ce test
            cs = {c: corr(F[g1][c], F[g2][c]) for c in cols}
            is_pos = same_group(m1, m2)
            is_neg = (v1 == "base" and v2 == "base") or (v1 == "base") != (v2 == "base") or (v1.startswith("lora:") and v2.startswith("lora:") and not is_pos)
            for c in cols: (res[c][0] if is_pos else res[c][1] if is_neg else []).append(cs[c]) if (is_pos or is_neg) else None
            if is_pos or is_neg: (comb[0] if is_pos else comb[1]).append(sum(cs.values()) / len(cs))
    print("mondes", len(paths))
    for c in cols:
        pos, neg = res[c]
        print("%-3s AUC=%.3f  n_pos=%d n_neg=%d  corr pos med %.2f / neg med %.2f p95 %.2f" % (
            c, auc(pos, neg), len(pos), len(neg), np.median(pos) if pos else float('nan'), np.median(neg), np.percentile(neg, 95)))
    print("moyenne des traits : AUC=%.3f" % auc(*comb))
