"""z du lecteur fp8 SANS corrige : chaque endpoint contre la moyenne de tous les autres de sa famille, apparie par question.
Entree : le JSON ecrit par score_ref.py de Titouan (champ S par lecteur et par reponse). Aucune requete."""
import sys, json, math, collections

res = json.load(open(sys.argv[1]))
schemes = sys.argv[2].split(",") if len(sys.argv) > 2 else ["e4m3_tensor", "w8a8_tensor"]
eps = sorted(res["prompts"])
out = {}
for sc in schemes:
    if sc not in res["S"]:
        continue
    per = {}
    for e in eps:
        d = collections.defaultdict(list)
        for p, s in zip(res["prompts"][e], res["S"][sc][e]):
            d[p].append(s)
        per[e] = {p: sum(v) / len(v) for p, v in d.items()}
    common = sorted(set.intersection(*[set(per[e]) for e in eps]))
    z = {}
    for e in eps:
        pool = [q for q in eps if q != e]
        d = [per[e][p] - sum(per[q][p] for q in pool) / len(pool) for p in common]
        mu = sum(d) / len(d)
        sd = math.sqrt(sum((x - mu) ** 2 for x in d) / (len(d) - 1))
        z[e] = round(mu / (sd / math.sqrt(len(d))), 2) if sd > 0 else 0.0
    out[sc] = z
    print(f"{sc:12s} z contre les autres ({len(common)} questions) : " + " ".join(f"{e[-2:]}={z[e]:+.1f}" for e in eps))
json.dump(out, open(sys.argv[1].replace(".json", "_zpool.json"), "w"), indent=1)
