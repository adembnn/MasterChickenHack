#!/usr/bin/env python
"""Nouvelles pistes T2 (agent recherche) — 0 requête API, réponses déjà collectées.
Test A : banque de lecteurs de quantification AUTRES que fp8 (NF4 bnb émulé, int8 W8A16, int8 W8A8 dynamique,
         int4 RTN g128) ; s = log p_lecteur - log p_base par réponse, z robuste par famille (ep_readers.robust_z).
Test B : PIT randomisé (Rank-based Uniformity, RUT appliqué token par token avec loi nulle EXACTE) sous la base,
         après ajustement MLE du décodeur (T, top_p, top_k). Copie déguisée en décodage => u ~ U(0,1) ;
         poids différents => u non uniforme même au meilleur décodeur.
Usage : python np_tests.py --data X.jsonl --families fam.json --out res.json [--only fam]
"""
import sys, os, json, math, argparse, collections, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import fp8_llr as F
from fp8_bank import make
from ep_readers import seq_logprob, robust_z, BASES, READ
from lora_distill import batch_tensors

NF4 = torch.tensor([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
                    -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
                    0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
                    0.7229568362236023, 1.0])


@torch.no_grad()
def quantize_(model, scheme):
    for name, m in model.named_modules():
        if not isinstance(m, torch.nn.Linear) or "lm_head" in name:
            continue
        w = m.weight.data.float()
        if scheme == "nf4":  # bitsandbytes NF4, blocs de 64 sur le tenseur aplati (ordre ligne), absmax par bloc
            flat = w.reshape(-1, 64)
            amax = flat.abs().amax(1, keepdim=True).clamp(min=1e-8)
            code = NF4.to(w.device)
            q = code[(flat / amax).unsqueeze(-1).sub(code).abs().argmin(-1)]
            w2 = (q * amax).reshape(w.shape)
        elif scheme in ("int8_ch", "w8a8_int8"):
            s = w.abs().amax(1, keepdim=True).clamp(min=1e-8) / 127
            w2 = (w / s).round().clamp(-127, 127) * s
        elif scheme == "int4_g128":  # RTN asymétrique g128 (proxy grossier de GPTQ/AWQ)
            g = w.reshape(w.shape[0], -1, 128)
            mn, mx = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
            s = ((mx - mn) / 15).clamp(min=1e-8)
            zp = (-mn / s).round()
            w2 = (((g / s).round() + zp).clamp(0, 15) - zp).mul(s).reshape(w.shape)
        m.weight.data = w2.to(m.weight.dtype)
    if scheme == "w8a8_int8":  # activations int8 dynamiques par token (vLLM W8A8 int8)
        def hook(mod, inp):
            x = inp[0]
            s = x.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 127
            return ((x / s).round().clamp(-127, 127) * s).to(x.dtype),
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.Linear) and "lm_head" not in name:
                m.register_forward_pre_hook(hook)
    return model


TS = [round(0.45 + 0.05 * i, 2) for i in range(18)]  # 0,45 .. 1,30
PS = [1.0, 0.95, 0.9, 0.8]
KS = [0, 50]
CFG = [(t, p, k) for t in TS for p in PS for k in KS]


@torch.no_grad()
def pit_pass(model, tok, items, gen, bs=4):
    """Pour chaque config de décodeur : loglik totale (plancher 1e-6 hors support) et u randomisés par token.
    Accumulateurs sur GPU, une seule synchronisation par endpoint."""
    dev = model.device
    ll = torch.zeros(len(CFG), dtype=torch.float64, device=dev)
    nout = torch.zeros(len(CFG), device=dev)
    U = [[] for _ in CFG]
    for i in range(0, len(items), bs):
        ids, att, lab, plens, lens = batch_tensors(tok, items[i:i + bs], dev)
        logits = model(input_ids=ids, attention_mask=att).logits
        for j in range(len(lens)):
            a, b = plens[j] - 1, lens[j] - 2
            lg = logits[j, a:b].float()
            tgt = ids[j, a + 1:b + 1]
            M = 4096  # tête triée ; la queue est résumée par sa masse (tokens de rang >= M : rares)
            srt, order = lg.topk(M, -1)
            tl = lg.gather(-1, tgt[:, None]).squeeze(-1)
            inhead = (order == tgt[:, None])
            rank = torch.where(inhead.any(-1), inhead.int().argmax(-1), torch.full_like(tgt, M - 1))[:, None]
            tail_tok = ~inhead.any(-1)
            r = torch.rand(len(tgt), generator=gen).to(dev)
            c = 0
            for t in TS:
                lse = torch.logsumexp(lg / t, -1, keepdim=True)
                pr0 = torch.exp(srt / t - lse)  # probas exactes de la tête (normalisées sur tout le vocabulaire)
                ptgt = torch.exp(tl / t - lse.squeeze(-1))
                for p in PS:
                    for k in KS:
                        pr = pr0
                        if k:
                            pr = pr.clone(); pr[:, k:] = 0
                        cs = pr.cumsum(-1)
                        full = torch.ones_like(cs[:, -1]) if not k else cs[:, -1]
                        if p < 1.0:
                            pr = pr * ((cs - pr) < p * full[:, None])
                            cs = pr.cumsum(-1)
                            full = cs[:, -1]
                        Z = full
                        ok_tail = (p >= 1.0) & (k == 0)
                        px = torch.where(tail_tok, ptgt if ok_tail else torch.zeros_like(ptgt),
                                         pr.gather(-1, rank).squeeze(-1)) / Z
                        before = torch.where(tail_tok, cs[:, -1] / Z, cs.gather(-1, rank).squeeze(-1) / Z - px)
                        out = px <= 0
                        nout[c] += out.sum()
                        ll[c] += torch.log(px.clamp(min=1e-6)).double().sum()
                        U[c].append(torch.where(out, torch.ones_like(px), torch.where(tail_tok, before + r * (1 - before), before + r * px)))
                        c += 1
    return ll.cpu(), nout.cpu(), [torch.cat(u).cpu() for u in U]


def pit_stats(u):
    """Écarts à U(0,1) : moyenne, 2e moment, part dans les déciles extrêmes, AD normalisé."""
    n = len(u)
    us = u.double().clamp(1e-9, 1 - 1e-9).sort().values
    i = torch.arange(1, n + 1, dtype=torch.float64)
    ad = -n - ((2 * i - 1) * (us.log() + (1 - us.flip(0)).log())).sum().item() / n
    z_mean = (us.mean().item() - 0.5) / math.sqrt(1 / 12 / n)
    hi = (us > 0.9).double().mean().item(); lo = (us < 0.1).double().mean().item()
    z_hi = (hi - 0.1) / math.sqrt(0.09 / n); z_lo = (lo - 0.1) / math.sqrt(0.09 / n)
    return {"n": n, "AD": round(ad, 2), "z_mean": round(z_mean, 2), "z_top10": round(z_hi, 2), "z_bot10": round(z_lo, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--families", required=True)
    ap.add_argument("--only", default=""); ap.add_argument("--out", required=True)
    ap.add_argument("--readers", default="e4m3_tensor,nf4,int8_ch,w8a8_int8,int4_g128")
    ap.add_argument("--no_pit", action="store_true")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.data) if l.strip()]
    fams = json.load(open(a.families)); excl = set()
    if "families" in fams:
        excl = set(fams.get("T_star_ge_1", [])); fams = fams["families"]
    res = {}
    for fam in sorted(set(fams.values())):
        if a.only and fam not in a.only.split(","):
            continue
        F.MODEL = BASES[fam]
        tok = F.AutoTokenizer.from_pretrained(F.MODEL); tok.padding_side = "right"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        by = collections.defaultdict(list)
        for r in rows:
            if fams.get(r["endpoint"]) == fam and r.get("response"):
                by[r["endpoint"]].append((r["prompt"], r["response"]))
        eps = sorted(by); out = {"z": {}, "pit": {}}
        READ["rep"], READ["temp"] = 1.0, 1.0
        base = F.load(torch.bfloat16)
        lpb = {e: seq_logprob(base, tok, by[e]) for e in eps}
        if not a.no_pit:
            gen = torch.Generator().manual_seed(0)
            for e in eps:
                ll, nout, U = pit_pass(base, tok, by[e], gen)
                c = int(ll.argmax()); t, p, k = CFG[c]
                c_t = max((i for i, cf in enumerate(CFG) if cf[1] == 1.0 and cf[2] == 0), key=lambda i: ll[i])
                st = pit_stats(U[c]); st_t = pit_stats(U[c_t])
                out["pit"][e] = {"best": [t, p, k], "ll_best": round(ll[c].item(), 1), "n_out": int(nout[c]),
                                 "stats_best": st, "T_only": CFG[c_t][0], "ll_T_only": round(ll[c_t].item(), 1),
                                 "dll_trunc": round((ll[c] - ll[c_t]).item(), 1), "stats_T_only": st_t}
                print(fam, e, json.dumps(out["pit"][e]), flush=True)
        del base; torch.cuda.empty_cache()
        for rd in a.readers.split(","):
            m = make(rd) if rd in ("e4m3_tensor", "w8a8_tensor") else quantize_(F.load(torch.bfloat16), rd)
            S = {e: {p: x - y for (p, _), x, y in zip(by[e], seq_logprob(m, tok, by[e]), lpb[e])} for e in eps}
            del m; torch.cuda.empty_cache()
            z, ref = robust_z(S, eps, excl, 3.0)
            out["z"][rd] = z
            out.setdefault("mean_s", {})[rd] = {e: round(sum(S[e].values()) / len(S[e]), 3) for e in eps}
            print(fam, rd, json.dumps(z), flush=True)
        res[fam] = out
        json.dump(res, open(a.out, "w"), indent=1)
    print("FIN", flush=True)


if __name__ == "__main__":
    main()
