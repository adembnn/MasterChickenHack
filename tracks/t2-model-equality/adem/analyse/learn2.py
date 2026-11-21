#!/usr/bin/env python
"""Banc de mesure v4 : méthode stratifiée par type d'endpoint détecté, au budget réel de 100 réponses par endpoint.

Types détectés sur les réponses seules : T = échantillonné (deux réponses différentes au même message),
S = prompt système ou persona (fuite par les sondes d'extraction), P = le reste (en clair, ou filigrane non détecté).
Classifieurs comparés (appris sur une famille, testés sur l'autre) :
  ident seul            : la statistique de la v3 (part de réponses identiques), seuil appris
  RL toutes             : régression logistique sur toutes les statistiques, sans le type
  GB toutes             : arbres boostés, sans le type
  GB + type             : arbres boostés avec le type de la paire (PP, PS, SS, PT, ST, TT)
Décodeurs : lien moyen sur P(même) ; « rattachement » = hypothèse du corrigé (un endpoint déguisé sert le modèle de
base) : les P sont regroupés entre eux, puis S et T sont rattachés au groupe P central.
Mondes : ref (structure exacte du corrigé), mix (14 endpoints au hasard, variantes de poids déguisées elles aussi).
Usage : python learn2.py --probes ../probes/probes_v3.jsonl --world ../../world3 --fams olmo,qwen --draws 12
"""
import argparse, glob, itertools, json, os, pickle, random, statistics
from collections import defaultdict
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from parse import load
from features import build, pair_features, NAMES
from learn import mcc, cluster

DET = {"lp": 16, "meta": 12, "ab5": 16, "q1": 20, "nb5": 12, "word": 12, "open": 8}   # 96 + 4 répétitions de test = 100
SAMP1 = {"lp": 16, "meta": 12}                                                        # 28 réponses uniques
SAMP3 = {"ab5": 10, "q1": 8, "nb5": 3, "word": 3}                                     # 24 sondes x 3 = 72
SEL = ["ident", "pref", "bow", "lenr", "ident_closed", "m_q1", "m_ab5", "m_nb5", "m_word", "m_lp", "dlp", "dlen",
       "same_sys", "id_cos"]
STRATA = ["PP", "PS", "SS", "PT", "ST", "TT"]


def make_plan(rng, by_kind):
    det = {k: rng.sample(by_kind[k], min(n, len(by_kind[k]))) for k, n in DET.items()}
    test4 = det["ab5"][:2] + det["q1"][:2]
    plan_det = {p: (2 if p in test4 else 1) for v in det.values() for p in v}
    plan_samp = {p: 1 for k in SAMP1 for p in det[k]}
    for k, n in SAMP3.items():
        for p in det[k][:n]:
            plan_samp[p] = 3

    def plan(name, d):
        nondet = any(len(set(t.strip() for t in d[p][:2])) > 1 for p in test4 if p in d)
        return (plan_samp if nondet else plan_det), 3
    return plan


def kind(e):
    return "T" if e.nondet > 0 else ("S" if e.sys else "P")


def ref_world(fam, rng):
    pick = lambda a, b: rng.choice([a, b])
    q = rng.choice(["fp8", "fp8", "int8", "nf4"])
    return [f"{fam}-base-plain", f"{fam}-base-{pick('sys', 'sys2')}", f"{fam}-base-{pick('persona', 'persona2')}",
            f"{fam}-base-{pick('wm', 'wm2')}", f"{fam}-base-{pick('t1', 't1b')}", f"{fam}-{q}-plain",
            f"{fam}-lora0.3-plain", f"{fam}-lora0.7-plain", f"{fam}-lora1.5-plain"]


def attach(eps, P, kinds, ident, cut=0.5, bonus=0.1):
    """Décodeur « rattachement » (hypothèse lue dans le corrigé : un endpoint déguisé sert un modèle qui existe aussi
    en clair, le plus souvent le modèle de base). Les endpoints P sont regroupés entre eux ; chaque endpoint S ou T est
    rattaché au groupe P dont la P(même) moyenne est la plus haute, le groupe central recevant un bonus a priori."""
    ps = [e for e in eps if kinds[e] == "P"]
    if not ps:
        return {e: 0 for e in eps}
    lab = cluster(ps, P, cut)
    hub = max(ps, key=lambda e: statistics.mean([ident[frozenset((e, o))] for o in ps if o != e] or [0]))
    out = dict(lab)
    groups = {}
    for e, k in lab.items():
        groups.setdefault(k, []).append(e)
    for e in eps:
        if kinds[e] != "P":
            score = {k: statistics.mean(P[frozenset((e, o))] for o in g) + (bonus if k == lab[hub] else 0.0)
                     for k, g in groups.items()}
            out[e] = max(score, key=score.get)
    return out


def fit_strata(Xn, y, nf, C=0.3):
    """Une régression logistique par strate (type de paire) sur les statistiques SEL ; repli global si strate pauvre."""
    idx = [NAMES.index(n) for n in SEL]
    out = {"global": LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(Xn[:, idx], y)}
    for k, st in enumerate(STRATA):
        m = Xn[:, nf + k] > 0  # colonnes one-hot normalisées : positif = dans la strate
        if m.sum() >= 30 and 5 <= y[m].sum() <= m.sum() - 5:
            out[st] = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(Xn[m][:, idx], y[m])
    return out


def predict_strata(sm, Xn, nf):
    idx = [NAMES.index(n) for n in SEL]
    p = sm["global"].predict_proba(Xn[:, idx])[:, 1]
    for k, st in enumerate(STRATA):
        m = Xn[:, nf + k] > 0
        if st in sm and m.any():
            p[m] = sm[st].predict_proba(Xn[m][:, idx])[:, 1]
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", required=True)
    ap.add_argument("--world", required=True)
    ap.add_argument("--regime", default="G")
    ap.add_argument("--fams", default="olmo,qwen")
    ap.add_argument("--draws", type=int, default=12)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--save", default=None, help="enregistre le classifieur final (appris sur toutes les familles)")
    a = ap.parse_args()
    fams = a.fams.split(",")
    rng = random.Random(0)
    cache = a.cache or f"/tmp/adem_feats2_{a.regime}_{'_'.join(fams)}_{a.draws}.pkl"
    probes = [json.loads(l) for f in a.probes.split(",") for l in open(f, encoding="utf-8") if l.strip()]
    prompts = {p["id"]: p["prompt"] for p in probes}
    if os.path.exists(cache):
        feats, kinds_d, truth = pickle.load(open(cache, "rb"))
    else:
        feats, kinds_d, truth = {}, {}, {}
        for fam in fams:
            files = sorted(f for f in glob.glob(os.path.join(a.world, f"{fam}_{a.regime}_*.jsonl")))
            kinds, data = load(files, a.probes.split(",")[0])
            kinds.update({p["id"]: p["kind"] for p in probes})
            for f in glob.glob(os.path.join(a.world, f"{fam}_{a.regime}_*_truth.csv")):
                for l in open(f).read().splitlines():
                    if l and not l.startswith("endpoint"):
                        ep, _, model, _ = l.split(",")
                        truth[ep] = model
            by_kind = defaultdict(list)
            for pid, k in sorted(kinds.items()):
                by_kind[k].append(pid)
            for d in range(a.draws):
                eps = build(data, kinds, plan=make_plan(rng, by_kind), prompts=prompts)
                kinds_d[(fam, d)] = {n: kind(e) for n, e in eps.items()}
                feats[(fam, d)] = {frozenset((x, y)): pair_features(eps[x], eps[y])
                                   for x, y in itertools.combinations(sorted(eps), 2)}
                print(f"{fam} tirage {d} : {len(eps)} endpoints, types {dict((k, list(kinds_d[(fam, d)].values()).count(k)) for k in 'PST')}", flush=True)
        pickle.dump((feats, kinds_d, truth), open(cache, "wb"))

    # exactitude de la détection de type
    for fam in fams:
        ok = defaultdict(lambda: [0, 0])
        for d in range(a.draws):
            for ep, k in kinds_d[(fam, d)].items():
                dis = ep.split("-")[-1].rstrip("2b")
                want = "T" if dis == "t1" else ("S" if dis in ("sys", "persona") else "P")
                ok[dis][0] += k == want
                ok[dis][1] += 1
        print(f"détection du type, {fam} : " + "  ".join(f"{k} {v[0]}/{v[1]}" for k, v in sorted(ok.items())))

    def xy(fam_list, with_kind):
        X, y, keys = [], [], []
        for fam in fam_list:
            for d in range(a.draws):
                kd = kinds_d[(fam, d)]
                for pr, f in feats[(fam, d)].items():
                    p, q = sorted(pr)
                    st = "".join(sorted(kd[p] + kd[q]))
                    row = [f[n] for n in NAMES] + ([float(st == s) for s in STRATA] if with_kind else [])
                    X.append(row); y.append(int(truth[p] == truth[q])); keys.append((fam, d, pr))
        return np.array(X, float), np.array(y), keys

    wr = random.Random(1)
    table = []
    for test in fams:
        train = [f for f in fams if f != test] or [test]
        print(f"\n===== appris sur {','.join(train)}, testé sur {test} (régime {a.regime}, {a.draws} tirages au budget 100) =====")
        Xtr, ytr, _ = xy(train, True)
        Xte, yte, kte = xy([test], True)
        nf = len(NAMES)
        mu = np.nan_to_num(np.nanmean(Xtr, 0))
        fill = lambda X: np.where(np.isnan(X), mu, X)
        sd = np.nan_to_num(np.nanstd(Xtr, 0)) + 1e-6
        norm = lambda X: (fill(X) - mu) / sd
        i_id = NAMES.index("ident")
        preds = {}
        lr = LogisticRegression(C=10, max_iter=3000, class_weight="balanced").fit(norm(Xtr)[:, [i_id]], ytr)
        preds["ident seul (v3)"] = lr.predict_proba(norm(Xte)[:, [i_id]])[:, 1]
        lr = LogisticRegression(C=1, max_iter=5000, class_weight="balanced").fit(norm(Xtr)[:, :nf], ytr)
        preds["RL toutes"] = lr.predict_proba(norm(Xte)[:, :nf])[:, 1]
        gb = GradientBoostingClassifier(n_estimators=200, max_depth=3, subsample=0.8, random_state=0).fit(fill(Xtr)[:, :nf], ytr)
        preds["GB toutes"] = gb.predict_proba(fill(Xte)[:, :nf])[:, 1]
        gbk = GradientBoostingClassifier(n_estimators=200, max_depth=3, subsample=0.8, random_state=0).fit(fill(Xtr), ytr)
        preds["GB + type"] = gbk.predict_proba(fill(Xte))[:, 1]
        sm = fit_strata(norm(Xtr), ytr, nf)
        preds["RL par strate"] = predict_strata(sm, norm(Xte), nf)
        print(f"{'classifieur':20s} {'décodeur':15s} {'AUC':>6s} {'MCC ref':>8s} {'[p10 ; p90]':>14s} {'MCC mix':>8s} {'[p10 ; p90]':>14s}")
        for name, p in preds.items():
            auc = roc_auc_score(yte, p)
            P, I = defaultdict(dict), defaultdict(dict)
            for (fam, d, pr), v, row in zip(kte, p, Xte):
                P[d][pr] = v
                I[d][pr] = row[i_id]
            for dec in ("lien moyen", "rattachement", "rattach. centre"):
                mref, mmix = [], []
                for d in range(a.draws):
                    kd = kinds_d[(test, d)]
                    all_eps = sorted(kd)
                    for _ in range(4):
                        for world, acc in ((ref_world(test, wr), mref), (wr.sample(all_eps, min(14, len(all_eps))), mmix)):
                            world = [e for e in world if e in kd]
                            lab = cluster(world, P[d]) if dec == "lien moyen" else attach(
                                world, P[d], kd, I[d], bonus=0.1 if dec == "rattachement" else 9.0)
                            acc.append(mcc(truth, lab, world))
                q = lambda v: f"[{np.percentile(v, 10):.2f} ; {np.percentile(v, 90):.2f}]"
                print(f"{name:20s} {dec:15s} {auc:6.3f} {np.mean(mref):8.3f} {q(mref):>14s} {np.mean(mmix):8.3f} {q(mmix):>14s}")
                table.append((test, name, dec, auc, np.mean(mref), np.mean(mmix)))
        # diagnostic par strate
        p = preds["GB + type"]
        by = defaultdict(lambda: ([], []))
        for (fam, d, pr), v, yy in zip(kte, p, yte):
            x, y_ = sorted(pr)
            st = "".join(sorted(kinds_d[(fam, d)][x] + kinds_d[(fam, d)][y_]))
            by[st][0].append(v); by[st][1].append(yy)
        print("GB + type, par strate : " + "  ".join(
            f"{st} n={len(v[1])} AUC={roc_auc_score(v[1], v[0]):.2f}" for st, v in sorted(by.items()) if 0 < sum(v[1]) < len(v[1])))
        imp = sorted(zip(gbk.feature_importances_, NAMES + STRATA), reverse=True)[:10]
        print("importances : " + " ".join(f"{n}={v:.2f}" for v, n in imp))
    if a.save:  # classifieur final, appris sur toutes les familles, en JSON (évaluable sans scikit-learn)
        X, y, _ = xy(fams, True)
        nf = len(NAMES)
        mu = np.nan_to_num(np.nanmean(X, 0))
        sd = np.nan_to_num(np.nanstd(X, 0)) + 1e-6
        Xn = (np.where(np.isnan(X), mu, X) - mu) / sd
        sm = fit_strata(Xn, y, nf)
        idx = [NAMES.index(n) for n in SEL]
        js = {"names": SEL, "mu": mu[idx].tolist(), "sd": sd[idx].tolist(), "regime": a.regime, "fams": fams,
              "strata": {k: {"coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0])} for k, m in sm.items()}}
        json.dump(js, open(a.save, "w"), indent=1)
        print("classifieur enregistré :", a.save, "strates :", sorted(sm))


if __name__ == "__main__":
    main()
