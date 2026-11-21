#!/usr/bin/env python
"""Lecteurs fp8 per-tensor + LoRA distillée, famille par famille, sur des réponses DÉJÀ collectées (aucune requête).
Entrée : JSONL {"endpoint","prompt","response"} + une famille par endpoint (--families "ep-01:qwen3b,..." ou fichier JSON).
Pour chaque famille (base locale connue) :
  1. s_fp8 = log p_fp8(y) - log p_base(y) avec deux lecteurs per-tensor (e4m3 poids seuls, W8A8 type vLLM).
  2. LoRA distillée : corpus = l'endpoint « LoRA forte » de la famille (--lora_train, ou auto = NLL/token la plus haute sous
     la base parmi les endpoints non exclus). Validation croisée par PROMPT (plis disjoints) et l'endpoint d'entraînement
     n'est jamais noté. s_lora(k) = log p_{base+LoRA*k}(y) - log p_base(y).
  3. Pour chaque lecteur : z apparié par prompt de chaque endpoint contre une référence ROBUSTE = médiane par prompt des autres
     endpoints de la famille, après retrait itératif des endpoints déjà hors norme (|z| > --z_out) et des endpoints --exclude_ref
     (ex. guichets T>=1,0).
Usage : python ep_readers.py --data ep.jsonl --families fam.json --out results_ep_readers.json
"""
import argparse, collections, json, math, statistics
import torch
import fp8_llr as F
from fp8_bank import make
from lora_distill import train_lora, set_scale, fold_of, seq_logprob as _seq_logprob


def seq_logprob(m, tok, items):
    return _seq_logprob(m, tok, items, bs=4)  # grand vocabulaire (Qwen 151k) sur GPU partagé

BASES = {"qwen3b": "Qwen/Qwen2.5-3B-Instruct", "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
         "llama1b": "unsloth/Llama-3.2-1B-Instruct", "smol360m": "HuggingFaceTB/SmolLM2-360M-Instruct"}


def robust_z(S, eps, exclude_ref, z_out, iters=3):
    """S[ep][prompt] -> s. Référence = médiane par prompt des endpoints de référence (hors endpoint testé)."""
    ref = [e for e in eps if e not in exclude_ref]
    z = {}
    for _ in range(iters):
        z = {}
        for ep in eps:
            pool = [q for q in ref if q != ep]
            common = [p for p in S[ep] if all(p in S[q] for q in pool)]
            if len(pool) < 2 or len(common) < 3:
                z[ep] = None
                continue
            d = [S[ep][p] - statistics.median(S[q][p] for q in pool) for p in common]
            mu = sum(d) / len(d)
            sd = math.sqrt(sum((x - mu) ** 2 for x in d) / (len(d) - 1))
            z[ep] = round(mu / (sd / math.sqrt(len(d))), 2) if sd > 0 else None
        new_ref = [e for e in eps if e not in exclude_ref and (z[e] is None or abs(z[e]) <= z_out)]
        if new_ref == ref or len(new_ref) < 3:
            break
        ref = new_ref
    return z, ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--families", required=True, help="fichier JSON {endpoint: famille} ou 'ep-01:qwen3b,...'")
    ap.add_argument("--only", default="", help="familles à traiter (virgules)")
    ap.add_argument("--exclude_ref", default="", help="endpoints jamais pris comme référence (ex. guichets T>=1)")
    ap.add_argument("--lora_train", default="", help="'famille:ep-XX,...' ; sinon auto (NLL/token max)")
    ap.add_argument("--scales", default="0.3,0.6,1.0")
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--z_out", type=float, default=3.0)
    ap.add_argument("--no_lora", action="store_true")
    ap.add_argument("--out", default="results_ep_readers.json")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.data) if l.strip()]
    fams = (json.load(open(a.families)) if a.families.endswith(".json")
            else dict(x.split(":") for x in a.families.split(",")))
    exclude_ref = set(filter(None, a.exclude_ref.split(",")))
    lora_train = dict(x.split(":") for x in a.lora_train.split(",")) if a.lora_train else {}
    scales = [float(x) for x in a.scales.split(",")]
    res = {"config": vars(a), "families": {}}

    for fam in sorted(set(fams.values())):
        if a.only and fam not in a.only.split(","):
            continue
        F.MODEL = BASES.get(fam, fam)
        tok = F.AutoTokenizer.from_pretrained(F.MODEL)
        tok.padding_side = "right"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        by_ep = collections.defaultdict(list)
        for r in rows:
            if fams.get(r["endpoint"]) == fam and r.get("response"):
                by_ep[r["endpoint"]].append((r["prompt"], r["response"]))
        eps = sorted(by_ep)
        print(f"== famille {fam} ({F.MODEL}) : {len(eps)} endpoints", flush=True)
        out = {"base": F.MODEL, "endpoints": eps, "z": {}, "ref_used": {}}

        base = F.load(torch.bfloat16)
        lp_base = {ep: seq_logprob(base, tok, by_ep[ep]) for ep in eps}
        ntok = {ep: [max(1, len(tok(r, add_special_tokens=False)["input_ids"][:160])) for _, r in by_ep[ep]] for ep in eps}
        out["nll_per_token_base"] = {ep: round(-sum(lp_base[ep]) / sum(ntok[ep]), 4) for ep in eps}
        print("NLL/token sous la base:", out["nll_per_token_base"], flush=True)
        del base
        torch.cuda.empty_cache()

        # 1. lecteurs fp8 per-tensor
        for rd in ["e4m3_tensor", "w8a8_tensor"]:
            m = make(rd)
            S = {ep: {p: x - y for (p, _), x, y in zip(by_ep[ep], seq_logprob(m, tok, by_ep[ep]), lp_base[ep])} for ep in eps}
            del m
            torch.cuda.empty_cache()
            out["z"][rd], out["ref_used"][rd] = robust_z(S, eps, exclude_ref, a.z_out)
            print(rd, out["z"][rd], flush=True)

        # 2. LoRA distillée, validation croisée par prompt, endpoint d'entraînement jamais noté
        if not a.no_lora:
            fp8_flag = {e for e in eps if max((out["z"][r].get(e) or 0) for r in ["e4m3_tensor", "w8a8_tensor"]) > a.z_out}
            cands = [e for e in eps if e not in exclude_ref and e not in fp8_flag] or [e for e in eps if e not in exclude_ref]
            out["lora_candidates_nll"] = sorted(((out["nll_per_token_base"][e], e) for e in cands), reverse=True)[:3]
            tr = lora_train.get(fam) or max(cands, key=lambda e: out["nll_per_token_base"][e])
            out["lora_train_endpoint"] = tr
            test_eps = [e for e in eps if e != tr]
            Sk = {k: {ep: {} for ep in test_eps} for k in scales}
            for f in range(a.folds):
                base = F.load(torch.bfloat16)
                train = [(p, r, 1.0) for p, r in by_ep[tr] if fold_of(p, a.folds) != f]
                pm, store = train_lora(a, tok, base, train, f)
                for k in scales:
                    set_scale(pm, k, store)
                    for ep in test_eps:
                        items = [(p, r) for p, r in by_ep[ep] if fold_of(p, a.folds) == f]
                        idx = [i for i, (p, _) in enumerate(by_ep[ep]) if fold_of(p, a.folds) == f]
                        for (p, _), x, i in zip(items, seq_logprob(pm, tok, items), idx):
                            Sk[k][ep][p] = x - lp_base[ep][i]
                del pm, base
                torch.cuda.empty_cache()
            for k in scales:
                key = f"lora_x{k}"
                out["z"][key], out["ref_used"][key] = robust_z(Sk[k], test_eps, exclude_ref, a.z_out)
                print(key, "(entraîné sur", tr, ")", out["z"][key], flush=True)
        res["families"][fam] = out
        json.dump(res, open(a.out, "w"), indent=1)
    print("FIN ->", a.out, flush=True)


if __name__ == "__main__":
    main()
