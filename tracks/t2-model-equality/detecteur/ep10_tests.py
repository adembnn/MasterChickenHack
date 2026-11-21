"""Tests de stabilité d'un guichet ep (ex. ep-10) : 5 graines x 5 sous-échantillons de jumeaux (CPU seulement).
Règle fixée d'avance (demande de Titouan) : (1) le guichet sort « variante » dans >= 4/5 sous-échantillons ;
(2) contrôle nul : les copies simples des 9 ref (ref-01/03/05/06/08) et les tirages frais ne sortent jamais ;
(3) attribution : contributions SHAP par groupe de caractéristiques pour la paire (cible, plus proche voisin base).
python ep10_tests.py --target ep-10 --fam llama1b=feats_ep/ep_bloc0_llama1b,ep_bloc0_llama1b.jsonl
"""
import argparse, itertools, json, random
import numpy as np, lightgbm as lgb
import pairs as P, train_eval as T

NULL_COPIES = {"ref-01", "ref-03", "ref-05", "ref-06", "ref-08"}
TDROP = "tstar,z_,lp_,H_,zb,zresp".split(",")


def group(n):
    for pre, g in (("absdiff_", "|écart|"), ("mean_", "moy"), ("min_", "min"), ("max_", "max")):
        if n.startswith(pre):
            n = n[len(pre):]; break
    else:
        return "croisées(vocab/longueur/z)"
    if n.startswith("lam_") or n.startswith("zq_"):
        return "lecteurs typés (λ, zq)"
    if n in ("top1", "rank10", "lowp", "tok_per_resp", "frac_length", "words_per_resp"):
        return "NLL/rangs/longueur"
    if n.startswith(("c_", "w_", "ttr", "rep3")):
        return "style de texte"
    return "autres"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True); ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--fam", required=True); ap.add_argument("--target", required=True)
    ap.add_argument("--ref", default="feats_ref/ref_bloc0,ref_bloc0.jsonl")
    ap.add_argument("--extra_null", action="append", default=[], help="npz,jsonl de tirages frais de ref")
    ap.add_argument("--drop2", default="", help="préfixes supplémentaires à retirer (caractéristiques sensibles à T)")
    ap.add_argument("--k", type=int, default=25); ap.add_argument("--jdir", default="worlds"); ap.add_argument("--out", default="")
    a = ap.parse_args()
    T.DROP.extend(TDROP); T.DROP.extend([d for d in a.drop2.split(',') if d])
    worlds = []
    names = None
    for st in a.train:
        w = P.load_world(*T.world_files(st, a.jdir)); F, A, names = T.build(w, a.k, names)
        X, y, meta = T.pairs_of(w, F, A, names); worlds.append((w, F, A, X, y, meta))
    pn = P.pair_names(names)
    vworlds = []
    for st in a.val:
        w = P.load_world(*T.world_files(st, a.jdir)); F, A, _ = T.build(w, a.k, names); vworlds.append((w, F, A))
    def load(spec):
        stem, jl = spec.split(","); w = P.load_world(stem + ".npz", jl); F, A, _ = T.build(w, a.k, names); return w, F, A
    fam, spec = a.fam.split("="); ep = load(spec); refs = [("ref_bloc0", load(a.ref))] + [(s.split(",")[0], load(s)) for s in a.extra_null]
    def pmat(model, W):
        w, F, A = W; n = len(w); M = np.eye(n)
        for i, j in itertools.combinations(range(n), 2):
            M[i, j] = M[j, i] = model.predict(P.pair_features(F[i], F[j], A[i], A[j], names)[None, :])[0]
        return M
    runs = []
    for seed in range(5):
        for sub in range(5):
            r = random.Random(1000 * seed + sub)
            Xs, ys = [], []
            for (w, F, A, X, y, meta) in worlds:   # sous-échantillon : 80 % des guichets
                keep = set(r.sample(range(len(w)), int(0.8 * len(w))))
                m = np.array([(i in keep and j in keep) for i, j in meta]); Xs.append(X[m]); ys.append(y[m])
            Xs, ys = np.concatenate(Xs), np.concatenate(ys)
            model = lgb.train({"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 40, "feature_fraction": 0.7,
                               "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "scale_pos_weight": (len(ys) - ys.sum()) / max(ys.sum(), 1),
                               "verbose": -1, "seed": seed + 1, "num_threads": 6}, lgb.Dataset(Xs, ys, feature_name=pn), num_boost_round=300)
            sw = {t: float(np.mean([T.score_sets(w, F, A, names, model, random.Random(1), n_sets=40, thr=t, wid=f"v{vi}_{seed}_{sub}")
                                    for vi, (w, F, A) in enumerate(vworlds)])) for t in (0.1, 0.2, 0.3)}
            thr = max(sw, key=sw.get)
            M = pmat(model, ep); lab = P.complete_linkage(M, thr); gids = [g["gid"] for g in ep[0]]
            t = gids.index(a.target)
            # groupe base = plus grand groupe
            sizes = {c: lab.count(c) for c in set(lab)}; bc = max(sizes, key=sizes.get)
            bidx = [i for i in range(len(lab)) if lab[i] == bc]
            conf = 1 - max([M[t][j] for j in bidx if j != t] or [0])
            res = {"seed": seed, "sub": sub, "thr": thr, "variant": lab[t] != bc, "conf": round(float(conf), 2), "base_size": len(bidx),
                   "isolated": sorted(g for g, l in zip(gids, lab) if l != bc)}
            nulls = {}
            for name, W in refs:
                Mr = pmat(model, W); lr = P.complete_linkage(Mr, thr); rg = [g["gid"] for g in W[0]]
                # copies simples : sortent-elles du groupe qui contient la majorité des copies simples ?
                cs = [i for i, g in enumerate(rg) if g in NULL_COPIES]
                cl = max(set(lr[i] for i in cs), key=lambda c: sum(lr[i] == c for i in cs))
                nulls[name] = [rg[i] for i in cs if lr[i] != cl]
            res["null_flagged"] = nulls
            runs.append(res)
            print(json.dumps(res, ensure_ascii=False), flush=True)
            if seed == 0 and sub == 0:   # attribution SHAP sur le premier modèle
                F, A = ep[1], ep[2]
                j = max([k for k in bidx if k != t], key=lambda k: M[t][k])
                x = P.pair_features(F[t], F[j], A[t], A[j], names)[None, :]
                c = model.predict(x, pred_contrib=True)[0]
                grp = {}
                for n_, v in zip(pn, c[:-1]):
                    grp[group(n_)] = grp.get(group(n_), 0.0) + v
                top = sorted(zip(pn, c[:-1]), key=lambda t_: -abs(t_[1]))[:12]
                print("ATTRIB voisin_base=", gids[j], "P=", round(float(M[t][j]), 2), "par groupe:", {k: round(v, 2) for k, v in sorted(grp.items(), key=lambda t_: t_[1])},
                      "\n  top:", [(n_, round(float(v), 2)) for n_, v in top], flush=True)
                med = {n_: float(np.median([F[i][n_] for i in bidx if i != t])) for n_ in names}
                sd = {n_: float(np.std([F[i][n_] for i in bidx if i != t])) + 1e-9 for n_ in names}
                zz = sorted(((n_, (F[t][n_] - med[n_]) / sd[n_]) for n_ in names), key=lambda t_: -abs(t_[1]))[:10]
                print("  cible vs médiane du groupe base (écarts en σ):", [(n_, round(v, 1), round(F[t][n_], 3)) for n_, v in zz], flush=True)
    nv = sum(r["variant"] for r in runs)
    nn = sum(len(v) > 0 for r in runs for v in r["null_flagged"].values())
    print(f"VERDICT {a.target}: variante dans {nv}/{len(runs)} entraînements ; contrôle nul (copies simples ref sorties) : {nn} cas sur {len(runs) * len(refs)}")
    if a.out:
        json.dump(runs, open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
