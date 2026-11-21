#!/usr/bin/env python
"""Expérience 2b : puissance de détection d'int8 au vrai budget (100 réponses par endpoint).

Regénère la base (bs=16) et deux configurations « bruit » (bs=8, bs=32) en sauvant les sorties par prompt,
recharge les sorties int8 d'exp2 (mêmes prompts, même bs=16 -> reproductibles à 100 %), puis :
  - matrice des paires : identical_frac et token_match_frac entre toutes les configurations (y compris bruit vs bruit,
    c.-à-d. deux endpoints du MÊME modèle servis avec des lots différents, et int8 vs bruit) ;
  - bootstrap par sous-échantillons de n prompts (n = 50, 100, 200) : seuil = 5e centile du bruit (bs8 vs bs32),
    puissance = part des tirages int8 sous ce seuil ; séparation = P(stat_int8 < stat_bruit) sur tirages indépendants.
Aucune requête API. Usage : python exp2b_power.py --exp2_outputs results_exp2_a100_outputs.json --out results_exp2b.json
"""
import argparse, json, os, random, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp1_divergence import MODEL, build_prompts, generate, first_div, load  # noqa: E402
from exp2_int8_detection import compare2  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def stats(a_list, b_list, idx):
    same, matched, total = 0, 0, 0
    for i in idx:
        a, b = a_list[i], b_list[i]
        d = first_div(a, b)
        total += len(a)
        if d is None:
            same += 1
            matched += len(a)
        else:
            matched += d
    return same / len(idx), matched / total


def bootstrap(base, noise_a, noise_b, variants, n, B, seed=0):
    rnd = random.Random(seed)
    N = len(base)
    noise_id, noise_tok = [], []
    var_id = {k: [] for k in variants}
    var_tok = {k: [] for k in variants}
    for _ in range(B):
        idx = rnd.sample(range(N), n)
        s_id, s_tok = stats(noise_a, noise_b, idx)
        noise_id.append(s_id)
        noise_tok.append(s_tok)
        idx2 = rnd.sample(range(N), n)
        for k, o in variants.items():
            # endpoint int8 (bs16) contre endpoint fp servi avec un autre lot (bs32) : la paire réaliste
            v_id, v_tok = stats(o, noise_b, idx2)
            var_id[k].append(v_id)
            var_tok[k].append(v_tok)
    out = {}
    for metric, nz, vz in [("identical_frac", noise_id, var_id), ("token_match_frac", noise_tok, var_tok)]:
        thr = sorted(nz)[int(0.05 * B)]
        m = {"noise_mean": sum(nz) / B, "noise_p05": thr, "noise_p50": sorted(nz)[B // 2]}
        for k in variants:
            v = vz[k]
            m[k] = {"mean": sum(v) / B, "power_at_fpr5": sum(1 for x in v if x < thr) / B,
                    "p_below_noise": sum(1 for x, y in zip(v, nz) if x < y) / B}
        out[metric] = m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--max_new", type=int, default=96)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--exp2_outputs", default="results_exp2_a100_outputs.json")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--out", default="results_exp2b.json")
    a = ap.parse_args()
    dtype = getattr(torch, a.dtype)
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompts = build_prompts(a.n, seed=0)
    t0 = time.time()
    prev = json.load(open(a.exp2_outputs))

    base_m = load(dtype)
    cfgs = {"base_bs16": generate(base_m, tok, prompts, a.bs, a.max_new)}
    assert compare2(prev["base"], cfgs["base_bs16"])["identical_frac"] == 1.0, "base non reproductible"
    cfgs["noise_bs32"] = generate(base_m, tok, prompts, 32, a.max_new)
    cfgs["noise_bs8"] = generate(base_m, tok, prompts, 8, a.max_new)
    print(f"[{time.time()-t0:.0f}s] générations prêtes", flush=True)
    variants = {k: prev[k] for k in ["int8_w_channel", "int8_w_group128", "int8_w_group32", "int8_w8a8", "int8_w_tensor"]}
    cfgs.update(variants)

    names = list(cfgs)
    pair = {}
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            c = compare2(cfgs[x], cfgs[y])
            pair[f"{x} vs {y}"] = {"identical_frac": round(c["identical_frac"], 4), "token_match_frac": round(c["token_match_frac"], 4)}
            print(f"{x} vs {y}: {pair[f'{x} vs {y}']}", flush=True)

    boot = {}
    for n in [50, 100, 200]:
        boot[f"n{n}"] = bootstrap(cfgs["base_bs16"], cfgs["noise_bs8"], cfgs["noise_bs32"], variants, n, a.B)
        print(f"[{time.time()-t0:.0f}s] bootstrap n={n}: {json.dumps(boot[f'n{n}'])}", flush=True)

    json.dump({"config": vars(a), "pairs": pair, "bootstrap": boot}, open(a.out, "w"), indent=1)
    json.dump({k: cfgs[k] for k in ["base_bs16", "noise_bs32", "noise_bs8"]}, open(a.out.replace(".json", "_outputs.json"), "w"))
    print("FIN", round(time.time() - t0), "s ->", a.out, flush=True)


if __name__ == "__main__":
    main()
