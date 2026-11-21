#!/usr/bin/env python
"""Variante « température égale » de ep_readers.py (réponses déjà collectées, aucune requête).
Chaque endpoint est relu à SA température T* (base, lecteurs fp8 et LoRA, à l'identique). La LoRA distillée est apprise comme
(base+LoRA)/T*_corpus, elle n'absorbe donc pas la température du corpus. Le corpus est l'endpoint dont la NLL à son T* (surprise que
la température n'explique pas) est maximale, hors T>=1 et hors fp8. Décision LoRA sur le z résiduel après régression z ~ NLL(T*).
fp8 : règle de concordance, les deux lecteurs per-tensor doivent être d'accord (min > t_min et max > t_eff*sqrt(n)).
Usage : python ep_readers_t.py --data ep.jsonl --families fam.json --tstar tstar.json --out results_ep_readers_t.json
"""
import argparse, collections, json, math
import torch
import fp8_llr as F
from fp8_bank import make
from lora_distill import train_lora, set_scale, fold_of
from ep_readers import BASES, robust_z, seq_logprob


def ols_resid(x, y):
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sxx = sum((a - mx) ** 2 for a in x)
    b = sum((a - mx) * (c - my) for a, c in zip(x, y)) / sxx if sxx > 0 else 0.0
    sy = math.sqrt(sum((c - my) ** 2 for c in y)) or 1.0
    corr = b * math.sqrt(sxx) / sy
    return [c - (my + b * (a - mx)) for a, c in zip(x, y)], b, corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--families", required=True)
    ap.add_argument("--tstar", required=True, help="JSON {endpoint: T*}")
    ap.add_argument("--only", default="")
    ap.add_argument("--scales", default="0.3,0.6,1.0")
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--z_out", type=float, default=3.0)
    ap.add_argument("--t_eff", type=float, default=0.51)
    ap.add_argument("--t_min", type=float, default=2.0)
    ap.add_argument("--corpus_exclude", default="", help="endpoints interdits comme corpus LoRA (déguisés)")
    ap.add_argument("--corpus", default="", help="forcer le corpus LoRA (un endpoint)")
    ap.add_argument("--out", default="results_ep_readers_t.json")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.data) if l.strip()]
    fams = json.load(open(a.families))
    fams = fams.get("families", fams)
    TS = {k: float(v) for k, v in json.load(open(a.tstar)).items()}
    hot = {e for e, t in TS.items() if t >= 1.0}
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
        excl = hot & set(eps)
        out = {"base": F.MODEL, "endpoints": eps, "tstar": {e: TS.get(e, 1.0) for e in eps}, "z": {}}
        print(f"== {fam} : {len(eps)} endpoints, exclus de la référence {sorted(excl)}", flush=True)

        base = F.load(torch.bfloat16)
        lp_base = {e: seq_logprob(base, tok, by_ep[e], temp=TS.get(e, 1.0), rep=1.0) for e in eps}
        ntok = {e: sum(max(1, len(tok(r, add_special_tokens=False)["input_ids"][:160])) for _, r in by_ep[e]) for e in eps}
        nllT = {e: -sum(lp_base[e]) / ntok[e] for e in eps}
        out["nll_per_token_at_tstar"] = {e: round(v, 4) for e, v in nllT.items()}
        print("NLL/token à T*:", out["nll_per_token_at_tstar"], flush=True)
        del base
        torch.cuda.empty_cache()

        for rd in ["e4m3_tensor", "w8a8_tensor"]:
            m = make(rd)
            S = {e: {p: x - y for (p, _), x, y in zip(by_ep[e], seq_logprob(m, tok, by_ep[e], temp=TS.get(e, 1.0), rep=1.0),
                                                      lp_base[e])} for e in eps}
            del m
            torch.cuda.empty_cache()
            out["z"][rd], _ = robust_z(S, eps, excl, a.z_out)
            print(rd, "(T*)", out["z"][rd], flush=True)
        n_e = {e: len({p for p, _ in by_ep[e]}) for e in eps}
        zf = {e: [out["z"][r].get(e) or -99 for r in ["e4m3_tensor", "w8a8_tensor"]] for e in eps}
        fp8 = {e for e in eps if min(zf[e]) > a.t_min and max(zf[e]) > a.t_eff * math.sqrt(n_e[e])}
        out["fp8"] = sorted(fp8)

        bad = set(filter(None, a.corpus_exclude.split(",")))
        cands = [e for e in eps if e not in excl and e not in fp8 and e not in bad]
        tr = a.corpus if a.corpus in eps else max(cands, key=lambda e: nllT[e])
        out["lora_train_endpoint"], out["lora_candidates"] = tr, sorted(((round(nllT[e], 4), e) for e in cands), reverse=True)[:3]
        test = [e for e in eps if e != tr]
        Sk = {k: {e: {} for e in test} for k in scales}
        for f in range(a.folds):
            base = F.load(torch.bfloat16)
            train = [(p, r, 1.0) for p, r in by_ep[tr] if fold_of(p, a.folds) != f]
            pm, store = train_lora(a, tok, base, train, f, temp=TS.get(tr, 1.0))
            for k in scales:
                set_scale(pm, k, store)
                for e in test:
                    idx = [i for i, (p, _) in enumerate(by_ep[e]) if fold_of(p, a.folds) == f]
                    items = [by_ep[e][i] for i in idx]
                    for i, x in zip(idx, seq_logprob(pm, tok, items, temp=TS.get(e, 1.0), rep=1.0)):
                        Sk[k][e][by_ep[e][i][0]] = x - lp_base[e][i]
            del pm, base
            torch.cuda.empty_cache()
        out["lora_resid"], out["lora_corr_with_nll"] = {}, {}
        for k in scales:
            key = f"lora_x{k}"
            out["z"][key], _ = robust_z(Sk[k], test, excl, a.z_out)
            ok = [e for e in test if out["z"][key].get(e) is not None and e not in excl]
            resid, slope, corr = ols_resid([nllT[e] for e in ok], [out["z"][key][e] for e in ok])
            out["lora_resid"][key] = {e: round(v, 2) for e, v in zip(ok, resid)}
            out["lora_corr_with_nll"][key] = round(corr, 2)
            print(key, "(T*, corpus", tr, ") z", out["z"][key], "| corr(z,NLL)", round(corr, 2), "| résidus",
                  out["lora_resid"][key], flush=True)

        labels = {}
        for e in eps:
            rl = max([out["lora_resid"][k].get(e, -99) for k in out["lora_resid"]] or [-99])
            if e in fp8:
                labels[e] = f"{fam}_fp8"
            elif e != tr and rl > a.t_eff * math.sqrt(n_e[e]):
                labels[e] = f"{fam}_lora_{e}"
            else:
                labels[e] = f"{fam}_base"
        out["labels"] = labels
        print("étiquettes:", json.dumps(labels), flush=True)
        res["families"][fam] = out
        json.dump(res, open(a.out, "w"), indent=1)
        json.dump({e: l for o in res["families"].values() for e, l in o["labels"].items()},
                  open(a.out.replace(".json", "_labels.json"), "w"), indent=1)
    print("FIN ->", a.out, flush=True)


if __name__ == "__main__":
    main()
