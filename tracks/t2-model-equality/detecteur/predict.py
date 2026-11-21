"""Applique une règle FIGÉE (modèle LightGBM + seuil + noms de caractéristiques, sortis par train_eval.py) à des guichets
réels, famille par famille, et écrit le CSV endpoint,family,model.

python predict.py --res out_r4/res_k25_x.json --model out_r4/model_k25_x.txt --k 25 \
    --fam qwen3=feats_ep/ep_bloc0_qwen3,ep_bloc0_qwen3.jsonl --fam qwen1.5=... --out submission_ep.csv
"""
import argparse, itertools, json
import numpy as np
import lightgbm as lgb
import pairs as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--fam", action="append", required=True, help="nom=stem_npz,fichier_jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--thr", type=float, default=None)
    a = ap.parse_args()
    res = json.load(open(a.res))
    names, thr = res["feature_names"], (a.thr if a.thr is not None else res["thr"])
    model = lgb.Booster(model_file=a.model)
    rows, report = [], {}
    for fi, spec in enumerate(a.fam):
        fam, rest = spec.split("=")
        stem, jl = rest.split(",")
        w = P.load_world(stem + ".npz", jl)
        F = [P.guichet_features(g, a.k) for g in w]
        A = [P.guichet_aux(g, a.k) for g in w]
        n = len(w); M = np.eye(n)
        Fs, nm = (P.set_relative(F, list(range(n)), names), names + ["rel_" + x for x in names]) if res.get("setrel") else (F, names)
        for i, j in itertools.combinations(range(n), 2):
            M[i, j] = M[j, i] = model.predict(P.pair_features(Fs[i], Fs[j], A[i], A[j], nm)[None, :])[0]
        lab = P.complete_linkage(M, thr)
        for g, l in zip(w, lab):
            rows.append((g["gid"], f"F{fi}", f"F{fi}M{l}"))
        report[fam] = {"gids": [g["gid"] for g in w], "labels": lab, "P": [[round(float(x), 2) for x in r] for r in M]}
        sizes = sorted(np.bincount(lab).tolist(), reverse=True)
        print(fam, "groupes :", sizes)
        for g, r in zip(w, M):
            print("  ", g["gid"], " ".join(f"{x:.2f}" for x in r))
    with open(a.out, "w") as f:
        f.write("endpoint,family,model\n")
        for r in sorted(rows):
            f.write(",".join(r) + "\n")
    json.dump(report, open(a.out.replace(".csv", ".json"), "w"), indent=1)
    print("écrit", a.out, "seuil", thr)


if __name__ == "__main__":
    main()
