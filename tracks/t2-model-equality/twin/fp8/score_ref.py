#!/usr/bin/env python
"""Test fp8 sur de VRAIES réponses déjà collectées (aucune requête faite par ce script).
Entrée : JSONL {"endpoint","prompt","response"}. Chaque réponse est relue (contexte = message user seul) par la base locale
et par une banque de lecteurs fp8 ; s = log p_fp8(y) - log p_base(y), sommé sur les tokens.
Sortie : par endpoint et par lecteur, moyenne de s et z apparié contre un endpoint de référence (même prompts).
Usage : python score_ref.py --data ref_responses.jsonl --model HuggingFaceTB/SmolLM2-360M-Instruct --ref ref-01
"""
import argparse, json, math, collections
import torch
import fp8_llr as F
from fp8_bank import SCHEMES, make


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    ap.add_argument("--ref", default="ref-01")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--read_temp", type=float, default=1.0, help="tempérer les logits du lecteur (T du serveur)")
    ap.add_argument("--out", default="results_score_ref.json")
    a = ap.parse_args()
    F.MODEL = a.model
    tok = F.AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rows = [json.loads(l) for l in open(a.data) if l.strip()]
    by_ep = collections.defaultdict(list)
    for r in rows:
        by_ep[r["endpoint"]].append(r)
    eps = sorted(by_ep)

    def read(model):
        out = {}
        for ep in eps:
            P = [r["prompt"] for r in by_ep[ep]]
            G = [tok(r["response"], add_special_tokens=False)["input_ids"] or [tok.eos_token_id] for r in by_ep[ep]]
            if a.read_temp != 1.0:
                orig = model.forward

                def fwd(*x, **k):
                    o = orig(*x, **k)
                    o.logits = o.logits / a.read_temp
                    return o
                model.forward = fwd
            out[ep] = [sum(t) for t in F.token_logprobs(model, tok, P, G, a.bs)]
            if a.read_temp != 1.0:
                model.forward = orig
        return out

    base = read(F.load(torch.bfloat16))
    S = {}
    for r in SCHEMES:
        m = make(r)
        lp = read(m)
        S[r] = {ep: [x - y for x, y in zip(lp[ep], base[ep])] for ep in eps}
        del m
        torch.cuda.empty_cache()
        print("lu par", r, flush=True)

    res = {"config": vars(a), "n_per_ep": {ep: len(by_ep[ep]) for ep in eps}, "mean_s": {}, "z_vs_ref": {}}
    ref_prompts = {r["prompt"]: i for i, r in enumerate(by_ep[a.ref])}
    for r in SCHEMES:
        res["mean_s"][r] = {ep: round(sum(S[r][ep]) / len(S[r][ep]), 4) for ep in eps}
        res["z_vs_ref"][r] = {}
        for ep in eps:
            pairs = [(x, S[r][a.ref][ref_prompts[q["prompt"]]]) for x, q in zip(S[r][ep], by_ep[ep]) if q["prompt"] in ref_prompts]
            if len(pairs) > 2 and ep != a.ref:
                d = [x - y for x, y in pairs]
                mu = sum(d) / len(d)
                sd = math.sqrt(sum((x - mu) ** 2 for x in d) / (len(d) - 1))
                res["z_vs_ref"][r][ep] = round(mu / (sd / math.sqrt(len(d))), 2) if sd > 0 else None
    res["S"] = S
    res["prompts"] = {ep: [r["prompt"] for r in by_ep[ep]] for ep in eps}
    res["base_lp"] = base
    print(json.dumps({k: res[k] for k in ["mean_s", "z_vs_ref"]}, indent=1))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
