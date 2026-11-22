#!/usr/bin/env python
"""Levier 5 : la fp8 d'un 3B (famille qwen3b) est-elle détectable à ~24 réponses, et ep-12 en est-il une ?
Zéro requête : jumeau local Qwen2.5-3B, mêmes 24 prompts que les vrais ep, tirage par la generation_config du modèle
(T=0,7, top_p 0,8, top_k 20, pénalité de répétition), comme un serveur vLLM par défaut.
  - nulls : la base, plusieurs graines ; alternatives : fp8 e4m3 par tenseur (poids) et W8A8 par tenseur (poids +
    activations dynamiques), plusieurs graines ;
  - lecteurs : e4m3_tensor, w8a8_tensor ; par token d = log p_lecteur - log p_base et KL(base || lecteur) ;
  - agrégations par réponse : somme (all), tokens du top 20 % de KL (top), pondération par la KL (wkl) ;
  - z apparié par prompt contre la moyenne des autres nulls (laisser-un-dehors) ; « both » = z_e4m3 + z_w8a8 ;
  - grille de seuils de concordance (z_e4m3 >= a ET z_w8a8 >= b) : précision × rappel ;
  - puis la meilleure agrégation sur les vrais ep qwen3b (référence = autres qwen3b de T* < 1,0).
Usage : python lever5.py --raw ep_bloc0/raw_Qwen2.5-3B-Instruct.json --out lever5.json"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "fp8"))
import fp8_llr as F  # noqa: E402


def chat(tok, prompts):
    return [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts]


@torch.no_grad()
def sample(model, tok, prompts, seed, max_new):
    torch.manual_seed(seed)
    enc = tok(chat(tok, prompts), return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    gen = model.generate(**enc, max_new_tokens=max_new, pad_token_id=tok.pad_token_id)  # generation_config du modèle
    out = []
    for row in gen[:, enc["input_ids"].shape[1]:]:
        ids = row.tolist()
        for stop in ([tok.eos_token_id] if isinstance(tok.eos_token_id, int) else []) + [tok.convert_tokens_to_ids("<|im_end|>")]:
            if stop in ids:
                ids = ids[:ids.index(stop) + 1]
        out.append([t for t in ids if t != tok.pad_token_id])
    return out


@torch.no_grad()
def token_stats(base, readers, tok, prompts, gens, bs=8):
    """Par réponse : liste de (d_r, kl_r) par token pour chaque lecteur r."""
    texts = chat(tok, prompts)
    out = []
    for i in range(0, len(texts), bs):
        rows, plens = [], []
        for t, g in zip(texts[i:i + bs], gens[i:i + bs]):
            p = tok(t, add_special_tokens=False)["input_ids"]
            rows.append(p + list(g))
            plens.append(len(p))
        L = max(len(r) for r in rows)
        ids = torch.full((len(rows), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(rows), L), dtype=torch.long)
        for j, r in enumerate(rows):
            ids[j, :len(r)] = torch.tensor(r)
            att[j, :len(r)] = 1
        ids, att = ids.to(base.device), att.to(base.device)
        lb = torch.log_softmax(base(input_ids=ids, attention_mask=att).logits.float(), -1)
        lr = {n: torch.log_softmax(m(input_ids=ids, attention_mask=att).logits.float(), -1) for n, m in readers.items()}
        for j, r in enumerate(rows):
            s, e = plens[j] - 1, len(r) - 1
            tgt = ids[j, plens[j]:len(r)][:, None]
            b = lb[j, s:e]
            rec = {}
            for n in readers:
                x = lr[n][j, s:e]
                d = (x.gather(-1, tgt) - b.gather(-1, tgt)).squeeze(-1)
                kl = (b.exp() * (b - x)).sum(-1)
                rec[n] = (d.tolist(), kl.tolist())
            out.append(rec)
    return out


def agg(rec, n, how):
    d, kl = rec[n]
    if not d:
        return 0.0
    if how == "all":
        return sum(d)
    if how == "top":
        thr = sorted(kl)[int(0.8 * (len(kl) - 1))]
        return sum(x for x, k in zip(d, kl) if k >= thr)
    if how == "wkl":
        m = sum(kl) / len(kl) or 1e-9
        return sum(x * k for x, k in zip(d, kl)) / m
    raise ValueError(how)


def paired_z(x, refs):
    d = [xi - sum(r[i] for r in refs) / len(refs) for i, xi in enumerate(x)]
    mu = sum(d) / len(d)
    sd = math.sqrt(sum((v - mu) ** 2 for v in d) / (len(d) - 1))
    return mu / (sd / math.sqrt(len(d))) if sd > 0 else 0.0


def auc(pos, neg):
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / max(len(pos) * len(neg), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="raw_Qwen2.5-3B-Instruct.json (from_journal) : prompts + vraies réponses")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n_null", type=int, default=8)
    ap.add_argument("--n_alt", type=int, default=4, help="graines par schéma fp8 distant")
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--exclude", default="ep-33", help="endpoints réels hors référence (T* >= 1,0)")
    ap.add_argument("--out", default="lever5.json")
    a = ap.parse_args()
    t0 = time.time()
    log = lambda *x: print(f"[{time.time()-t0:.0f}s]", *x, flush=True)
    raw = json.load(open(a.raw))
    prompts = raw["rd_prompts"]
    F.MODEL = a.model
    tok = F.AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    base = F.load(torch.bfloat16)
    e4m3 = F.quant_weights_(F.load(torch.bfloat16), "e4m3", per="tensor")
    w8a8 = F.add_fp8_act_hooks(F.quant_weights_(F.load(torch.bfloat16), "e4m3", per="tensor"), "e4m3")
    readers = {"e4m3": e4m3, "w8a8": w8a8}
    log(f"modèles prêts ; {len(prompts)} prompts")

    sim = {}
    for k in range(a.n_null):
        sim[f"null{k}"] = ("null", sample(base, tok, prompts, 100 + k, a.max_new))
    for scheme, m in readers.items():
        for k in range(a.n_alt):
            sim[f"{scheme}_alt{k}"] = (scheme, sample(m, tok, prompts, 200 + k, a.max_new))
    log("tirages faits :", len(sim), "endpoints simulés")
    st = {n: token_stats(base, readers, tok, prompts, g) for n, (_, g) in sim.items()}
    log("lectures simulées faites")

    hows = ["all", "top", "wkl"]
    nulls = [n for n in sim if sim[n][0] == "null"]
    z = {}
    for n in sim:
        refs_names = [m for m in nulls if m != n]
        z[n] = {}
        for r in readers:
            for h in hows:
                x = [agg(rec, r, h) for rec in st[n]]
                refs = [[agg(rec, r, h) for rec in st[m]] for m in refs_names]
                z[n][f"{r}_{h}"] = paired_z(x, refs)
        for h in hows:
            z[n][f"both_{h}"] = z[n][f"e4m3_{h}"] + z[n][f"w8a8_{h}"]
    alts = [n for n in sim if n not in nulls]
    res = {"sim_z": z, "auc": {}, "power_at_3": {}, "fpr_at_3": {}}
    for key in z[nulls[0]]:
        pos = [z[n][key] for n in alts]
        neg = [z[n][key] for n in nulls]
        thr = 3.0 * (math.sqrt(2) if key.startswith("both") else 1.0)
        res["auc"][key] = round(auc(pos, neg), 3)
        res["power_at_3"][key] = round(sum(p >= thr for p in pos) / len(pos), 3)
        res["fpr_at_3"][key] = round(sum(q >= thr for q in neg) / len(neg), 3)
    log("AUC :", res["auc"])
    log("puissance à z>=3 :", res["power_at_3"])
    log("faux positifs à z>=3 :", res["fpr_at_3"])

    best = max((k for k in res["auc"] if not k.startswith("both")), key=lambda k: res["auc"][k])
    h = best.split("_")[1]
    grid = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    conc = []
    for ta in grid:
        for tb in grid:
            pred_pos = [n for n in sim if z[n][f"e4m3_{h}"] >= ta and z[n][f"w8a8_{h}"] >= tb]
            tp = sum(n in alts for n in pred_pos)
            prec = tp / len(pred_pos) if pred_pos else 0.0
            rec = tp / len(alts)
            conc.append({"a": ta, "b": tb, "precision": round(prec, 3), "recall": round(rec, 3), "pxr": round(prec * rec, 3)})
    conc.sort(key=lambda c: -c["pxr"])
    res["concordance_best"] = conc[:5]
    res["best_aggregation"] = h
    log(f"meilleure agrégation : {h} ; concordance (top 3) :", conc[:3])

    E = raw["endpoints"]
    real_st = {e: token_stats(base, readers, tok, prompts, d["rd_ids"]) for e, d in E.items()}
    excl = set(a.exclude.split(","))
    real = {}
    for e in E:
        refs_names = [m for m in E if m != e and m not in excl]
        real[e] = {}
        for r in readers:
            for hh in hows:
                x = [agg(rec, r, hh) for rec in real_st[e]]
                refs = [[agg(rec, r, hh) for rec in real_st[m]] for m in refs_names]
                real[e][f"{r}_{hh}"] = round(paired_z(x, refs), 2)
        for hh in hows:
            real[e][f"both_{hh}"] = round(real[e][f"e4m3_{hh}"] + real[e][f"w8a8_{hh}"], 2)
    res["real_z"] = real
    log("vrais ep qwen3b (z e4m3 / w8a8, agrégation retenue) :")
    for e in sorted(real):
        log(f"  {e}: all {real[e]['e4m3_all']:+.1f}/{real[e]['w8a8_all']:+.1f} | {h} {real[e][f'e4m3_{h}']:+.1f}/{real[e][f'w8a8_{h}']:+.1f}")
    json.dump(res, open(a.out, "w"), indent=1)
    log("FIN ->", a.out)


if __name__ == "__main__":
    main()
