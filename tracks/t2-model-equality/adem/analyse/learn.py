#!/usr/bin/env python
"""Banc de mesure : statistique x MCC sur le monde local déguisé, au budget réel (100 réponses par endpoint).

Pour chaque tirage : un plan de 100 réponses par endpoint est tiré parmi les sondes du banc (test de déterminisme
sur 4 sondes x 2 ; endpoint déterministe -> 92 autres sondes distinctes ; endpoint échantillonné -> 32 sondes fermées
x 3 répétitions), puis les statistiques par paire sont calculées. Un classifieur (régression logistique) appris sur une
famille est appliqué à l'autre. Le MCC « modèle » est calculé comme score_practice.py sur des mondes tirés :
  ref : structure exacte du corrigé (base en clair + 4 copies déguisées, 4 variantes de poids en clair)
  mix : 14 endpoints tirés au hasard (variantes de poids elles-mêmes déguisées)
Usage : python learn.py --probes probes_v1.jsonl --world ../../world --regime G --fams olmo,qwen --draws 20
"""
import argparse, glob, itertools, json, math, pickle, random, statistics, os
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from parse import load
from features import build, pair_features, NAMES

DET_KINDS = {"ab5": 16, "q1": 22, "nb5": 14, "word": 14, "open": 8, "list": 8, "seq": 8, "fs": 6}
SAMP_KINDS = {"ab5": 12, "q1": 8, "nb5": 6, "word": 6}


def make_plan(rng, by_kind):
    det = {k: rng.sample(by_kind[k], min(n, len(by_kind[k]))) for k, n in DET_KINDS.items() if k in by_kind}
    test4 = det["ab5"][:2] + det["q1"][:2]
    samp = set(test4)
    for k, n in SAMP_KINDS.items():
        samp |= set(det[k][:n])
    det_set = set(x for v in det.values() for x in v)

    def plan(name, d):
        nondet = any(len(set(t.strip() for t in d[p][:2])) > 1 for p in test4 if p in d)
        return (samp, 3) if nondet else (det_set, 1)
    return plan


def mcc(truth, pred, eps):
    tp = fp = fn = tn = 0
    for a, b in itertools.combinations(eps, 2):
        t, g = truth[a] == truth[b], pred[a] == pred[b]
        tp += t and g; fp += (not t) and g; fn += t and not g; tn += (not t) and not g
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return max(0.0, (tp * tn - fp * fn) / d) if d else 0.0


def cluster(eps, P, cut=0.5):
    """Regroupement hiérarchique à lien moyen sur P(même modèle), arrêt quand le meilleur lien passe sous `cut`."""
    cl = [[e] for e in eps]
    while len(cl) > 1:
        best, bi = -1, None
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                s = statistics.mean(P[frozenset((a, b))] for a in cl[i] for b in cl[j])
                if s > best:
                    best, bi = s, (i, j)
        if best < cut:
            break
        i, j = bi
        cl[i] += cl[j]
        del cl[j]
    return {e: k for k, c in enumerate(cl) for e in c}


def worlds(fam, rng, n_mix=3):
    pick = lambda a, b: rng.choice([a, b])
    q = rng.choice(["fp8", "fp8", "int8", "nf4"])
    ref = [f"{fam}-base-plain", f"{fam}-base-{pick('sys', 'sys2')}", f"{fam}-base-{pick('persona', 'persona2')}",
           f"{fam}-base-{pick('wm', 'wm2')}", f"{fam}-base-{pick('t1', 't1b')}", f"{fam}-{q}-plain",
           f"{fam}-lora0.3-plain", f"{fam}-lora0.7-plain", f"{fam}-lora1.5-plain"]
    return [("ref", ref)], n_mix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", required=True)
    ap.add_argument("--world", required=True)
    ap.add_argument("--regime", default="G")
    ap.add_argument("--fams", default="olmo,qwen")
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--save_model", default=None)
    a = ap.parse_args()
    fams = a.fams.split(",")
    rng = random.Random(0)
    feats = {}
    cache = a.cache or f"feats_{a.regime}_{'_'.join(fams)}_{a.draws}.pkl"
    truth = {}
    if os.path.exists(cache):
        feats, truth = pickle.load(open(cache, "rb"))
    else:
        for fam in fams:
            files = sorted(glob.glob(os.path.join(a.world, f"{fam}_{a.regime}_*.jsonl")))
            kinds, data = load(files, a.probes)
            for f in glob.glob(os.path.join(a.world, f"{fam}_{a.regime}_*_truth.csv")):
                for l in open(f).read().splitlines()[1:]:
                    ep, _, model, kind = l.split(",")
                    truth[ep] = model
            by_kind = defaultdict(list)
            for pid, k in kinds.items():
                by_kind[k].append(pid)
            for k in by_kind:
                by_kind[k].sort()
            for d in range(a.draws):
                eps = build(data, kinds, plan=make_plan(rng, by_kind))
                feats[(fam, d)] = {frozenset((x, y)): pair_features(eps[x], eps[y])
                                   for x, y in itertools.combinations(sorted(eps), 2)}
                print(f"{fam} tirage {d} : {len(eps)} endpoints", flush=True)
        pickle.dump((feats, truth), open(cache, "wb"))

    def xy(fam, draws):
        X, y, keys = [], [], []
        for d in draws:
            for pr, f in feats[(fam, d)].items():
                p, q = tuple(pr)
                X.append([f[n] for n in NAMES]); y.append(int(truth[p] == truth[q])); keys.append((d, pr))
        return np.array(X, float), np.array(y), keys

    def kind_of(ep):
        return ep.split("-")[-1].rstrip("2b") if not ep.endswith("plain2") else "plain"

    results = {}
    for test in fams:
        train = [f for f in fams if f != test] or [test]
        Xtr, ytr, _ = map(lambda *z: z[0], *[xy(f, range(a.draws)) for f in train]) if False else xy(train[0], range(a.draws))
        Xte, yte, kte = xy(test, range(a.draws))
        mu = np.nanmean(Xtr, 0)
        fill = lambda X: np.where(np.isnan(X), mu, X)
        sd = np.nanstd(Xtr, 0) + 1e-6
        norm = lambda X: (fill(X) - mu) / sd
        print(f"\n===== appris sur {train[0]}, testé sur {test} (régime {a.regime}, {a.draws} tirages de budget) =====")
        # 1) chaque statistique seule : AUC + MCC ref avec seuil appris sur la famille d'entraînement
        models = {}
        for i, n in enumerate(NAMES):
            if np.isnan(Xtr[:, i]).mean() > 0.5:
                continue
            lr = LogisticRegression(C=10.0, max_iter=2000, class_weight="balanced").fit(norm(Xtr)[:, [i]], ytr)
            models[n] = (lr, [i])
        sets = {
            "RL toutes": list(range(len(NAMES))),
            "RL texte (ident,pref,bow,lenr)": [NAMES.index(n) for n in ("ident", "pref", "ident_open", "bow", "lenr", "ident_closed")],
            "RL fermées (modes)": [NAMES.index(n) for n in ("m_q1", "m_ab5", "m_nb5", "m_word", "asym", "dA", "ddig")],
            "RL sans par-endpoint": list(range(NAMES.index("nondet_max"))),
        }
        for n, idx in sets.items():
            lr = LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced").fit(norm(Xtr)[:, idx], ytr)
            models[n] = (lr, idx)
        from sklearn.ensemble import GradientBoostingClassifier
        gb = GradientBoostingClassifier(n_estimators=150, max_depth=3, subsample=0.8, random_state=0).fit(fill(Xtr), ytr)
        print(f"{'statistique':34s} {'AUC':>6s} {'MCC ref':>8s} {'[p10 ; p90]':>15s} {'MCC mix':>8s} {'[p10 ; p90]':>15s}")
        wr = random.Random(1)
        for n in list(models) + ["GB toutes"]:
            if n == "GB toutes":
                p = gb.predict_proba(fill(Xte))[:, 1]
            else:
                lr, idx = models[n]
                p = lr.predict_proba(norm(Xte)[:, idx])[:, 1]
            auc = roc_auc_score(yte, p)
            P = defaultdict(dict)
            for (d, pr), v in zip(kte, p):
                P[d][pr] = v
            mref, mmix = [], []
            for d in range(a.draws):
                all_eps = sorted(set(e for pr in P[d] for e in pr))
                for _ in range(3):
                    (_, ref), _ = worlds(test, wr)[0][0], None
                    ref = [e for e in ref if e in all_eps]
                    mref.append(mcc(truth, cluster(ref, P[d]), ref))
                    mix = wr.sample(all_eps, min(14, len(all_eps)))
                    mmix.append(mcc(truth, cluster(mix, P[d]), mix))
            q = lambda v: f"[{np.percentile(v, 10):.2f} ; {np.percentile(v, 90):.2f}]"
            print(f"{n:34s} {auc:6.3f} {np.mean(mref):8.3f} {q(mref):>15s} {np.mean(mmix):8.3f} {q(mmix):>15s}")
            results[(test, n)] = (auc, np.mean(mref), np.mean(mmix))
        # 2) diagnostic : P(même) moyenne par type de paire, pour la RL complète
        lr, idx = models["RL toutes"]
        p = lr.predict_proba(norm(Xte)[:, idx])[:, 1]
        by = defaultdict(list)
        for (d, pr), v in zip(kte, p):
            x, y = sorted(pr)
            wx, wy = x.split("-")[1], y.split("-")[1]
            key = ("même " if wx == wy else "diff ") + "/".join(sorted((kind_of(x), kind_of(y))))
            by[key].append(v)
        print("P(même) moyenne de la RL complète par type de paire (déguisements) :")
        for k in sorted(by):
            print(f"   {k:28s} n={len(by[k]):5d}  P={np.mean(by[k]):.2f}")
        coef = sorted(zip(lr.coef_[0], [NAMES[i] for i in idx]), key=lambda t: -abs(t[0]))
        print("coefficients RL :", " ".join(f"{n}={c:+.2f}" for c, n in coef[:12]))
        if a.save_model:
            json.dump({"names": NAMES, "mu": mu.tolist(), "sd": sd.tolist(), "coef": lr.coef_[0].tolist(),
                       "intercept": float(lr.intercept_[0]), "train": train[0], "regime": a.regime},
                      open(f"{a.save_model}_{train[0]}_{a.regime}.json", "w"))


if __name__ == "__main__":
    main()
