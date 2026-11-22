#!/usr/bin/env python
"""Suite de fp8_llr.py : le format fp8 distant est inconnu, donc on lit chaque endpoint avec une BANQUE de lecteurs
fp8 locaux et on prend le meilleur score standardisé. Le seuil est calibré sur le MAXIMUM sur la banque calculé pour
les endpoints nuls (correction automatique des tests multiples). 100 % local, aucune requête API hackathon.
Usage : python fp8_bank.py --n 200 --n_numeric 80 --out results_fp8_bank.json
"""
import argparse, json, random, time
import torch
import fp8_llr as F

SCHEMES = {  # nom -> (format, granularité des poids, activations fp8 dynamiques)
    "e4m3_ch": ("e4m3", "channel", False),
    "e4m3_tensor": ("e4m3", "tensor", False),
    "w8a8_tensor": ("e4m3", "tensor", True),
    "w8a8_ch": ("e4m3", "channel", True),
    "e5m2_ch": ("e5m2", "channel", False),
}


def make(scheme, dtype=torch.bfloat16):
    fmt, per, act = SCHEMES[scheme]
    m = F.quant_weights_(F.load(dtype), fmt, per=per)
    return F.add_fp8_act_hooks(m, fmt) if act else m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n_numeric", type=int, default=80)
    ap.add_argument("--max_new", type=int, default=64)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--delta", type=float, default=4.0)
    ap.add_argument("--B", type=int, default=3000)
    ap.add_argument("--model", default=F.MODEL)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--out", default="results_fp8_bank.json")
    a = ap.parse_args()
    F.MODEL = a.model
    tok = F.AutoTokenizer.from_pretrained(F.MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    P = F.build_prompts(a.n, 0) + F.build_numeric(a.n_numeric, 0)
    t0 = time.time()
    log = lambda *x: print(f"[{time.time()-t0:.0f}s]", *x, flush=True)

    base = F.load(torch.bfloat16)
    V = base.get_output_embeddings().weight.shape[0]
    gens = {}
    wm = lambda key, m: F.KGW(V, 0.25, a.delta, key, m.device)
    T = a.temperature
    seed = iter(range(10, 10000))

    def endpoints(m, tag, bs):
        gens[("plain", tag)] = F.generate(m, tok, P, bs, a.max_new, temperature=T, seed=next(seed))
        gens[("sys_wm", tag)] = F.generate(m, tok, P, bs, a.max_new, system=True, wm=wm(next(seed), m),
                                           temperature=T, seed=next(seed))
        gens[("t1", tag)] = F.generate(m, tok, P, bs, a.max_new, temperature=1.0, seed=next(seed))

    endpoints(base, "null_bs16", a.bs)
    endpoints(base, "null_bs5", 5)
    m = F.load(torch.float16)
    endpoints(m, "null_fp16", a.bs)
    del m
    log("nuls générés")
    for s in SCHEMES:
        m = make(s)
        endpoints(m, f"fp8_{s}", a.bs)
        del m
        torch.cuda.empty_cache()
        log("généré", s)
    json.dump({f"{k[0]}/{k[1]}": v for k, v in gens.items()}, open(a.out.replace(".json", "_gens.json"), "w"))

    lp_base = {k: F.token_logprobs(base, tok, P, g, a.bs) for k, g in gens.items()}
    S = {}  # (endpoint, lecteur) -> s par réponse
    for r in SCHEMES:
        m = make(r)
        for k, g in gens.items():
            f = F.token_logprobs(m, tok, P, g, a.bs)
            S[(k, r)] = [sum(x) - sum(y) for x, y in zip(f, lp_base[k])]
        del m
        torch.cuda.empty_cache()
        log("lu par", r)

    rnd = random.Random(0)
    res = {"config": vars(a), "matrix_mean_s": {}, "power_bank": {}, "power_matched": {}}
    for k in gens:
        res["matrix_mean_s"][f"{k[0]}/{k[1]}"] = {r: round(sum(S[(k, r)]) / len(P), 3) for r in SCHEMES}
    for cls in ["plain", "sys_wm", "t1"]:
        nulls = [k for k in gens if k[0] == cls and k[1].startswith("null")]
        poss = [k for k in gens if k[0] == cls and k[1].startswith("fp8")]
        # standardisation par lecteur : moyenne et écart-type des s sur les réponses nulles de la classe
        mu, sd = {}, {}
        for r in SCHEMES:
            xs = [x for k in nulls for x in S[(k, r)]]
            mu[r] = sum(xs) / len(xs)
            sd[r] = (sum((x - mu[r]) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5

        def stat(k, idx, readers):
            return max(sum((S[(k, r)][i] - mu[r]) / sd[r] for i in idx) / len(idx) ** 0.5 for r in readers)

        for mode, readers_of in [("bank_max", lambda k: list(SCHEMES)),
                                 ("matched_only", lambda k: [k[1][4:]])]:
            out = {}
            for n in [n for n in [10, 25, 50, 100] if n <= len(P)]:
                # seuil : 95e centile du max sur la banque pour des endpoints nuls (même tirage de prompts)
                null_stats = sorted(stat(rnd.choice(nulls), rnd.sample(range(len(P)), n), list(SCHEMES)) if mode == "bank_max"
                                    else max(stat(rnd.choice(nulls), rnd.sample(range(len(P)), n), [r]) for r in [rnd.choice(list(SCHEMES))])
                                    for _ in range(a.B))
                thr = null_stats[int(0.95 * a.B)]
                row = {"thr": round(thr, 3)}
                for k in poss:
                    row[k[1]] = round(sum(1 for _ in range(a.B) if stat(k, rnd.sample(range(len(P)), n), readers_of(k)) > thr) / a.B, 3)
                out[f"n{n}"] = row
            res["power_bank" if mode == "bank_max" else "power_matched"][cls] = out
            log(cls, mode, json.dumps(out.get("n25")), json.dumps(out.get("n50")), json.dumps(out.get("n100")))
    json.dump(res, open(a.out, "w"), indent=1)
    log("FIN ->", a.out)


if __name__ == "__main__":
    main()
