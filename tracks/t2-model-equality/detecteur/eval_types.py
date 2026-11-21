"""Rappel/précision par type de variante, à k réponses, sur des mondes de validation (graines et LoRA neuves).

Pour chaque lot tiré (type ep : 4-6 copies de la base dont des copies CHAUDES T>=0,8 + variantes distinctes) :
partition par lien complet ; une copie base est « bien classée » si elle est dans le plus grand groupe de copies base ;
une variante est « détectée » si elle est séparée de toutes les copies base. On rapporte par type :
  variante : rappel = part séparée de la base ; base (par déguisement) : part rattachée au groupe base.
Précision « variante » = parmi les guichets déclarés variante, part qui en sont vraiment.

python eval_types.py --train A B --val C D --k 25 [--drop pref,pref] [--thr 0.2|auto]
"""
import argparse, itertools, json, random
from collections import defaultdict
import numpy as np
import lightgbm as lgb
import pairs as P
import train_eval as T


def vt(v):
    return "lora@" + v.split("@")[1] if v.startswith("lora:") else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--drop", default="")
    ap.add_argument("--thr", default="auto")
    ap.add_argument("--jdir", default="worlds")
    ap.add_argument("--n_sets", type=int, default=150)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    T.DROP.extend([d for d in a.drop.split(",") if d])
    rnd = random.Random(0); names = None; Xtr, ytr = [], []
    for st in a.train:
        w = P.load_world(*T.world_files(st, a.jdir))
        F, A, names = T.build(w, a.k, names)
        X, y, _ = T.pairs_of(w, F, A, names); Xtr.append(X); ytr.append(y)
    Xtr, ytr = np.concatenate(Xtr), np.concatenate(ytr)
    pn = P.pair_names(names)
    model = lgb.train({"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 40, "feature_fraction": 0.7,
                       "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
                       "scale_pos_weight": (len(ytr) - ytr.sum()) / max(ytr.sum(), 1), "verbose": -1, "seed": 1, "num_threads": 6},
                      lgb.Dataset(Xtr, ytr, feature_name=pn), num_boost_round=400)
    V = []
    for st in a.val:
        w = P.load_world(*T.world_files(st, a.jdir)); F, A, _ = T.build(w, a.k, names); V.append((st, w, F, A))
    def run(thr):
        det = defaultdict(lambda: [0, 0]); dec_var = [0, 0]; mccs = []
        for vi, (st, w, F, A) in enumerate(V):
            r = random.Random(7)
            for _ in range(a.n_sets):
                pick = T.random_set(w, r)
                X, keys = T.set_pairs(w, f"v{vi}", F, A, names, pick)
                M = np.eye(len(pick))
                for (i, j), v in zip(keys, model.predict(X)):
                    M[i, j] = M[j, i] = v
                lab = P.complete_linkage(M, thr)
                truth = [w[i]["variant"] for i in pick]
                mccs.append(P.mcc_pairs(truth, lab))
                bidx = [n for n, i in enumerate(pick) if w[i]["variant"] == "base"]
                cl = max(set(lab[n] for n in bidx), key=lambda c: sum(lab[n] == c for n in bidx))
                for n, i in enumerate(pick):
                    g = w[i]; in_base = lab[n] == cl
                    if g["variant"] == "base":
                        key = "base:" + g["kind"]; det[key][0] += in_base; det[key][1] += 1
                    else:
                        key = vt(g["variant"]); det[key][0] += (not in_base); det[key][1] += 1
                    declared_var = not in_base
                    if declared_var:
                        dec_var[1] += 1; dec_var[0] += g["variant"] != "base"
        return float(np.mean(mccs)), det, dec_var
    if a.thr == "auto":
        sw = {t: run(t)[0] for t in (0.1, 0.2, 0.3, 0.5)}
        thr = max(sw, key=sw.get); print("balayage seuil (sur validation)", {k: round(v, 3) for k, v in sw.items()})
    else:
        thr = float(a.thr)
    m, det, dv = run(thr)
    print(f"k={a.k} drop={a.drop or '-'} seuil={thr} MCC ep-like={m:.3f} ; précision 'variante' = {dv[0]}/{dv[1]} = {dv[0] / max(dv[1], 1):.3f}")
    for key in sorted(det):
        ok, n = det[key]
        lab = "part rattachée à la base" if key.startswith("base:") else "rappel (séparée de la base)"
        print(f"  {key:22s} {ok / n:.2f}  ({lab}, n={n})")
    if a.out:
        json.dump({"k": a.k, "drop": a.drop, "thr": thr, "mcc": m, "prec_variant": dv[0] / max(dv[1], 1),
                   "by_type": {k: v[0] / v[1] for k, v in det.items()}}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
