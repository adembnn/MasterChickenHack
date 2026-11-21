#!/usr/bin/env python
"""Expérience 3b : choisir les prompts stables face au bruit pour mieux détecter int8 (CPU seul, relit les *_raw.json d'exp3).

Sélection : prompts dont TOUTES les paires bruit du fichier --select sont identiques (option : exclure les paires
contenant --exclude, gardées pour l'évaluation). Évaluation sur --eval : identical_frac des paires bruit et int8
sur tous les prompts vs sur les prompts stables, puis bootstrap : seuil = 5e centile de la pire paire bruit,
puissance = part des tirages int8 sous le seuil. Si --select et --eval sont deux modèles différents, on mesure
le transfert de la sélection d'un modèle à l'autre (le cas réel : on choisit les prompts sur un jumeau local).
Usage : python exp3b_stable_prompts.py --select results_exp3_olmo_raw.json --exclude a100 --eval results_exp3_olmo_raw.json
"""
import argparse, json, random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--exclude", default="")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    S = json.load(open(a.select))["pairs"]
    E = json.load(open(a.eval))["pairs"]
    is_noise = lambda k: k.startswith("fp_") and "|fp_" in k
    has = lambda k: False
    if a.exclude and not a.exclude.startswith("fp_"):
        has = lambda k: a.exclude in k  # sous-chaîne libre (ex. « a100 »)
    elif a.exclude:
        has = lambda k: a.exclude in k.split("|")  # nom de configuration exact (ex. « fp_bs1 », pas « fp_bs16 »)
    sel = [k for k in S if is_noise(k) and not has(k)]
    N = len(S[sel[0]]["ident"])
    stable = [i for i in range(N) if all(S[k]["ident"][i] == 1 for k in sel)]
    ev_noise = [k for k in E if is_noise(k) and (has(k) if (a.exclude and a.select == a.eval) else True)]
    int8 = [k for k in E if k.startswith("int8_") and not k.startswith("int8_w8a8")]
    rate = lambda k, idx: sum(E[k]["ident"][i] for i in idx) / len(idx)
    res = {"n_stable": len(stable), "n": N, "select_pairs": sel, "eval_noise_pairs": ev_noise,
           "rates": {k: {"all": round(rate(k, range(N)), 4), "stable": round(rate(k, stable), 4)} for k in ev_noise + int8}}
    print(f"prompts stables : {len(stable)}/{N} (sélection sur {len(sel)} paires bruit)")
    for k in ev_noise + int8:
        print(f"  {k}: tous {res['rates'][k]['all']:.3f} | stables {res['rates'][k]['stable']:.3f}")
    rnd = random.Random(0)
    res["power"] = {}
    for subset_name, idx in [("all", list(range(N))), ("stable", stable)]:
        for n in [50, 100]:
            draw = lambda k: [sum(E[k]["ident"][i] for i in rnd.choices(idx, k=n)) / n for _ in range(a.B)]
            thr = min(sorted(draw(k))[int(0.05 * a.B)] for k in ev_noise)
            pw = {}
            for k in int8:
                v = draw(k)
                pw[k] = sum(1 for x in v if x < thr) / a.B
            agg = {}
            for k, v in pw.items():
                agg.setdefault(k.split("|")[0], []).append(v)
            summ = {q: {"min": round(min(v), 3), "mean": round(sum(v) / len(v), 3)} for q, v in agg.items()}
            res["power"][f"{subset_name}_n{n}"] = {"threshold": thr, "per_variant": summ}
            print(f"  [{subset_name} n={n}] seuil {thr:.3f} -> {summ}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
