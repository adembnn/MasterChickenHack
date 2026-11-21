"""Applique un classifieur figé aux ep d'une famille et liste, par guichet, la confiance « variante » = 1 - max P(même modèle)
vers les autres guichets du groupe base (plus grand groupe), pour --conf > 0,8 par défaut.
python ep_variants.py --res R.json --model M.txt --k 25 --fam llama1b=feats_ep/ep_bloc0_llama1b,ep_bloc0_llama1b.jsonl
"""
import argparse, itertools, json
import numpy as np, lightgbm as lgb
import pairs as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--k", type=int, default=25); ap.add_argument("--fam", action="append", required=True)
    ap.add_argument("--conf", type=float, default=0.8); ap.add_argument("--out", default="")
    a = ap.parse_args()
    res = json.load(open(a.res)); names = res["feature_names"]; thr = res["thr"]
    model = lgb.Booster(model_file=a.model); out = {}
    for spec in a.fam:
        fam, rest = spec.split("="); stem, jl = rest.split(",")
        w = P.load_world(stem + ".npz", jl)
        F = [P.guichet_features(g, a.k) for g in w]; A = [P.guichet_aux(g, a.k) for g in w]
        n = len(w); M = np.eye(n)
        for i, j in itertools.combinations(range(n), 2):
            M[i, j] = M[j, i] = model.predict(P.pair_features(F[i], F[j], A[i], A[j], names)[None, :])[0]
        lab = P.complete_linkage(M, thr)
        # groupe base = groupe qui contient le plus de guichets à forte similarité moyenne
        score = [np.mean(np.sort(M[i][np.arange(n) != i])[-4:]) for i in range(n)]   # voisinage serré
        base_c = max(set(lab), key=lambda c: (sum(l == c for l in lab), sum(s for s, l in zip(score, lab) if l == c)))
        bidx = [i for i in range(n) if lab[i] == base_c]
        rows = []
        for i in range(n):
            others = [M[i][j] for j in bidx if j != i]
            conf = 1 - max(others) if others else 1.0
            rows.append((w[i]["gid"], lab[i] == base_c, round(float(conf), 2)))
        out[fam] = {"base_group": [w[i]["gid"] for i in bidx], "thr": thr, "rows": rows}
        print(f"{fam}: groupe base ({len(bidx)}/{n}) = {[w[i]['gid'] for i in bidx]}")
        print("   variantes avec confiance > %.1f : %s" % (a.conf, [(g, c) for g, b, c in rows if not b and c > a.conf]))
        print("   variantes douteuses (conf<=%.1f) : %s" % (a.conf, [(g, c) for g, b, c in rows if not b and c <= a.conf]))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
