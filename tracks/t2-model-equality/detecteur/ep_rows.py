"""P(même modèle) d'un ou plusieurs guichets ep avec chaque autre guichet de la famille, sur 3 graines (classifieur complet, sans T)."""
import argparse, itertools, json
import numpy as np, lightgbm as lgb
import pairs as P, train_eval as T

ap = argparse.ArgumentParser()
ap.add_argument("--train", nargs="+", required=True); ap.add_argument("--fam", required=True)
ap.add_argument("--targets", required=True); ap.add_argument("--k", type=int, default=25)
ap.add_argument("--keepT", action="store_true"); ap.add_argument("--seeds", type=int, default=3)
a = ap.parse_args()
if not a.keepT:
    T.DROP.extend("tstar,z_,lp_,H_,zb,zresp".split(","))
names = None; Xs, ys = [], []
for st in a.train:
    w = P.load_world(*T.world_files(st, "worlds")); F, A, names = T.build(w, a.k, names)
    X, y, _ = T.pairs_of(w, F, A, names); Xs.append(X); ys.append(y)
X, y = np.concatenate(Xs), np.concatenate(ys); pn = P.pair_names(names)
stem, jl = a.fam.split(","); w = P.load_world(stem + ".npz", jl); F, A, _ = T.build(w, a.k, names)
gids = [g["gid"] for g in w]; n = len(w)
Ms = []
for s in range(a.seeds):
    m = lgb.train({"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 40, "feature_fraction": 0.7,
                   "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "scale_pos_weight": (len(y) - y.sum()) / max(y.sum(), 1),
                   "verbose": -1, "seed": s + 1, "num_threads": 6}, lgb.Dataset(X, y, feature_name=pn), num_boost_round=400)
    M = np.eye(n)
    for i, j in itertools.combinations(range(n), 2):
        M[i, j] = M[j, i] = m.predict(P.pair_features(F[i], F[j], A[i], A[j], names)[None, :])[0]
    Ms.append(M)
Ms = np.array(Ms)
print("colonnes:", " ".join(g[3:] for g in gids))
for t in a.targets.split(","):
    i = gids.index(t)
    for s in range(a.seeds):
        row = Ms[s][i]; oth = [(gids[j], row[j]) for j in range(n) if j != i]
        print(f"{t} graine{s+1}: " + " ".join(f"{g[3:]}={p:.2f}" for g, p in oth), f"| max={max(p for _, p in oth):.2f} -> conf_variante(1-max)={1-max(p for _, p in oth):.2f}")
    mean = Ms.mean(0)[i]; oth = sorted([(mean[j], gids[j]) for j in range(n) if j != i], reverse=True)
    print(f"{t} MOYENNE 3 graines: plus proches {[(g, round(p, 2)) for p, g in oth[:4]]} ; conf_variante moyenne = {1 - oth[0][0]:.2f}")
