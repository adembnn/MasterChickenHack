#!/usr/bin/env python
"""Piste B : test MMD façon MET (Gao, Liang, Guestrin, arXiv 2410.20247) sur des jumeaux masqués. Zéro requête API.

Principe. Pour chaque prompt i, une grande banque locale de N >= 500 réponses de la BASE (ses poids, notre copie) à
température tau ; un endpoint donne UNE réponse x_i par prompt (25 réponses au total, comme au bloc 0). Noyaux :
  H : Hamming sur les L premiers tokens (comme MET, rembourrage -1 après l'arrêt) ;
  F : gaussien (largeur 1) sur (log p_base moyen par token, entropie moyenne de la base), standardisés par prompt ;
  C : (H + F) / 2.
Statistique : T = somme_i (c_i(x_i) - kappa_i) / sigma_i, avec c_i(x) = moyenne_j k(x, y_j) et kappa_i = similarité
moyenne de la banque avec elle-même (leave-one-out). Test PAR PERMUTATION : sous H0, x_i est échangeable avec un
échantillon de la banque ; l'échange prompt par prompt donne la loi nulle exacte (B = 200 000 tirages), p bilatérale.
Température inconnue de l'endpoint : on teste contre la banque à chaque tau de la grille et on garde le MAX des p
(conservateur : une copie n'est isolée que si aucune température ne l'explique). Multiplicité : alpha / 33 (Bonferroni).

CRITÈRE FIXÉ D'AVANCE (écrit avant toute exécution) : pour un noyau donné, le test est « retenu » si
  (a) précision de l'appel « variante » >= 0,75 sur l'ensemble des réplicats (variantes = fp8 e4m3, W8A8, LoRA 0,3 / 0,7 /
      1,5 ; copies = base à T 0,5 / 0,7 / 0,75 / 1,0, base fp16, base + prompt système, base + persona, base + filigrane KGW),
  (b) au moins une variante détectée (TP > 0),
  (c) AUCUNE copie « isolée » : un type de copie est isolé si au moins 10 % de ses réplicats sont rejetés à alpha / 33.
Scénario principal = banque SANS le prompt caché. Scénarios secondaires : banque avec le vrai prompt caché (borne
supérieure si la sonde d'extraction réussissait) ; copies à prompt caché exclues.

Usage : bank | twins | eval (voir main)."""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "fp8"))

L = 48
GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ALPHA = 0.05
N_ENDPOINTS = 33
KERNELS = ["H", "F", "C"]
COPY_TWINS = ["T0.5", "T0.7", "T0.75", "T1.0", "fp16", "sys", "persona", "watermark"]
VARIANT_TWINS = ["fp8_e4m3", "w8a8", "lora0.3", "lora0.7", "lora1.5"]
HIDDEN = ["sys", "persona"]


def get_prompts(n):
    from probes_v4 import plan
    seen, out = set(), []
    for s in plan():
        if s["kind"] == "extract" or s["prompt"] in seen:
            continue
        seen.add(s["prompt"])
        out.append(s["prompt"])
        if len(out) == n:
            break
    return out


def ctx_text(tok, prompt, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def stop_ids(model, tok):
    s = set()
    g = model.generation_config.eos_token_id
    if g is not None:
        s |= set(g if isinstance(g, (list, tuple)) else [g])
    if tok.eos_token_id is not None:
        s.add(tok.eos_token_id)
    for t in ("<|im_end|>", "<|eot_id|>", "<|endoftext|>"):
        i = tok.convert_tokens_to_ids(t)
        if i is not None and i != tok.unk_token_id:
            s.add(i)
    return s


@torch.no_grad()
def sample_batch(model, tok, ctx, n, T, seed, stops, wm=None, top_p=1.0, top_k=0, rep=1.0):
    torch.manual_seed(seed)
    enc = tok([ctx], return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(**enc, do_sample=True, temperature=T, top_p=top_p, top_k=top_k, repetition_penalty=rep,
                         max_new_tokens=L, num_return_sequences=n, pad_token_id=tok.pad_token_id,
                         logits_processor=wm.processor() if wm is not None else None)
    gen = out[:, enc["input_ids"].shape[1]:].cpu()
    if gen.shape[1] < L:
        gen = torch.cat([gen, torch.full((n, L - gen.shape[1]), -1, dtype=gen.dtype)], 1)
    mask = torch.zeros_like(gen, dtype=torch.bool)
    for s in stops:
        mask |= gen == s
    first = torch.where(mask.any(1), mask.float().argmax(1), torch.full((n,), L))
    return torch.where(torch.arange(L)[None] >= first[:, None], torch.full_like(gen, -1), gen)


@torch.no_grad()
def features(model, tok, ctx, samples, bs=32):
    """(log p moyen par token, entropie moyenne) de la base à T=1, sur les tokens générés (hors rembourrage)."""
    p = tok(ctx, add_special_tokens=False)["input_ids"]
    P = len(p)
    pre = torch.tensor(p)[None]
    out = []
    for i in range(0, len(samples), bs):
        g = samples[i:i + bs]
        valid = g >= 0
        ids = torch.cat([pre.expand(len(g), -1), torch.where(valid, g, torch.full_like(g, tok.pad_token_id))], 1).to(model.device)
        att = torch.cat([torch.ones(len(g), P, dtype=torch.long), valid.long()], 1).to(model.device)
        lp = torch.log_softmax(model(input_ids=ids, attention_mask=att).logits[:, P - 1:P + L - 1].float(), -1)
        tlp = lp.gather(-1, ids[:, P:, None]).squeeze(-1)
        ent = -(lp.exp() * lp).sum(-1)
        v = valid.to(model.device).float()
        n = v.sum(1).clamp(min=1)
        out.append(torch.stack([(tlp * v).sum(1) / n, (ent * v).sum(1) / n], 1).cpu())
        del lp, tlp, ent
    return torch.cat(out)


def set_models(name):
    import exp1_divergence
    import watermark_robust
    import fp8_llr as F
    exp1_divergence.MODEL = watermark_robust.MODEL = F.MODEL = name


def load_tok(name):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# ----------------------------------------------------------------------------- banque
def cmd_bank(a):
    from method_v4 import SYSTEM_PROMPT, PERSONA_PROMPT
    import fp8_llr as F
    set_models(a.model)
    tok = load_tok(a.model)
    base = F.load(torch.bfloat16)
    stops = stop_ids(base, tok)
    prompts = get_prompts(a.n_prompts)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    jobs = [("plain", None, T, {}) for T in GRID]
    jobs += [("sys", SYSTEM_PROMPT, 0.7, {}), ("persona", PERSONA_PROMPT, 0.7, {})]
    if a.gencfg:  # configuration de génération propre au modèle (top_p, top_k, pénalité de répétition), grille de T
        gc = base.generation_config
        kw = dict(top_p=gc.top_p or 1.0, top_k=gc.top_k or 0, rep=gc.repetition_penalty or 1.0)
        jobs += [("gencfg", None, T, kw) for T in GRID]
    for name, system, T, kw in jobs:
        f = out / f"bank_{name}_T{T}.pt"
        if f.exists():
            continue
        toks, feats = [], []
        for i, pr in enumerate(prompts):
            c = ctx_text(tok, pr, system)
            s = sample_batch(base, tok, c, a.N, T, 1000 * i + int(T * 100), stops, **kw)
            toks.append(s)
            feats.append(features(base, tok, c, s))
        torch.save({"tok": torch.stack(toks), "feat": torch.stack(feats), "prompts": prompts, "T": T, "kw": kw}, f)
        print(f"[{time.time()-t0:.0f}s] banque {name} T={T} prête", flush=True)
    print("FIN banque", flush=True)


# ----------------------------------------------------------------------------- jumeaux masqués
def cmd_twins(a):
    from method_v4 import SYSTEM_PROMPT, PERSONA_PROMPT
    from exp1_divergence import build_prompts, train_lora, scale_lora
    from watermark_robust import KGW
    import fp8_llr as F
    set_models(a.model)
    tok = load_tok(a.model)
    base = F.load(torch.bfloat16)
    stops = stop_ids(base, tok)
    prompts = get_prompts(a.n_prompts)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    K = a.K

    def run(name, model, T, system=None, wm=None, off=0):
        f = out / f"twin_{name}.pt"
        if f.exists():
            return
        toks, feats, feats_ctx = [], [], []
        for i, pr in enumerate(prompts):
            c = ctx_text(tok, pr, system)
            s = sample_batch(model, tok, c, K, T, 500000 + 1000 * i + off, stops, wm=wm)
            toks.append(s)
            feats.append(features(base, tok, ctx_text(tok, pr), s))
            if system:
                feats_ctx.append(features(base, tok, c, s))
        d = {"tok": torch.stack(toks), "feat": torch.stack(feats), "T": T}
        if system:
            d["feat_ctx"] = torch.stack(feats_ctx)
        torch.save(d, f)
        print(f"[{time.time()-t0:.0f}s] jumeau {name} prêt", flush=True)

    for name, T in [("T0.5", 0.5), ("T0.7", 0.7), ("T0.75", 0.75), ("T1.0", 1.0)]:
        run(name, base, T, off=int(T * 100))
    run("sys", base, 0.7, system=SYSTEM_PROMPT, off=11)
    run("persona", base, 0.7, system=PERSONA_PROMPT, off=12)
    V = base.get_output_embeddings().weight.shape[0]
    run("watermark", base, 0.7, wm=KGW(V, gamma=0.25, delta=6.0, key=424242, device=base.device), off=13)
    m = F.load(torch.float16)
    run("fp16", m, 0.7, off=14)
    del m
    torch.cuda.empty_cache()
    m = F.quant_weights_(F.load(torch.bfloat16), "e4m3", per="tensor")
    run("fp8_e4m3", m, 0.7, off=15)
    del m
    torch.cuda.empty_cache()
    m = F.add_fp8_act_hooks(F.quant_weights_(F.load(torch.bfloat16), "e4m3", per="tensor"), "e4m3")
    run("w8a8", m, 0.7, off=16)
    del m
    torch.cuda.empty_cache()
    if not all((out / f"twin_lora{s}.pt").exists() for s in (0.3, 0.7, 1.5)):
        prompts_tr = build_prompts(120, seed=123)
        ctxs = [ctx_text(tok, p) for p in prompts_tr]
        outs = []
        with torch.no_grad():
            for i in range(0, len(ctxs), 40):
                enc = tok(ctxs[i:i + 40], return_tensors="pt", padding=True, add_special_tokens=False).to(base.device)
                g = base.generate(**enc, do_sample=False, max_new_tokens=40, pad_token_id=tok.pad_token_id)
                outs += [tok.decode(r, skip_special_tokens=True) for r in g[:, enc["input_ids"].shape[1]:]]
        pm = train_lora(tok, torch.bfloat16, 150, 4, [(c, " " + o + tok.eos_token) for c, o in zip(ctxs, outs)])
        store = {}
        for s in (0.3, 0.7, 1.5):
            scale_lora(pm, s, store)
            run(f"lora{s}", pm, 0.7, off=20 + int(s * 10))
    print("FIN jumeaux", flush=True)


# ----------------------------------------------------------------------------- test MMD
class Ref:
    """Banque de référence d'un (contexte, tau) : moyennes leave-one-out, loi nulle par permutation."""

    def __init__(self, tok_bank, feat_bank, B=200000):
        P, N, _ = tok_bank.shape
        self.tok = tok_bank
        self.mu = feat_bank.mean(1, keepdim=True)
        self.sd = feat_bank.std(1, keepdim=True).clamp(min=1e-6)
        self.fz = (feat_bank - self.mu) / self.sd
        mH, mF = [], []
        for i in range(P):
            KH = (tok_bank[i][:, None, :] == tok_bank[i][None, :, :]).float().mean(-1)
            KF = torch.exp(-torch.cdist(self.fz[i], self.fz[i]).pow(2) / 2)
            mH.append((KH.sum(1) - KH.diagonal()) / (N - 1))
            mF.append((KF.sum(1) - KF.diagonal()) / (N - 1))
        m = {"H": torch.stack(mH), "F": torch.stack(mF)}
        m["C"] = (m["H"] + m["F"]) / 2
        self.kappa, self.sigma, Sz = {}, {}, {}
        for k, v in m.items():
            kap = v.mean(1)
            S = v - kap[:, None]
            sig = S.std(1).clamp(min=1e-9)
            self.kappa[k], self.sigma[k], Sz[k] = kap, sig, S / sig[:, None]
        J = torch.randint(0, N, (B, P), generator=torch.Generator().manual_seed(0))
        ar = torch.arange(P)[None, :]
        self.null = {k: Sz[k][ar, J].sum(1).abs().sort().values for k in m}
        self.B = B

    def pvalues(self, x_tok, x_feat):
        """x_tok [P,K,L], x_feat [P,K,2] -> {kernel: (T [K], p [K])}."""
        P, K, _ = x_tok.shape
        cH = torch.stack([(x_tok[i][:, None, :] == self.tok[i][None]).float().mean(-1).mean(1) for i in range(P)])
        xz = (x_feat - self.mu) / self.sd
        cF = torch.stack([torch.exp(-torch.cdist(xz[i], self.fz[i]).pow(2) / 2).mean(1) for i in range(P)])
        c = {"H": cH, "F": cF, "C": (cH + cF) / 2}
        out = {}
        for k in c:
            T = ((c[k] - self.kappa[k][:, None]) / self.sigma[k][:, None]).sum(0)
            idx = torch.searchsorted(self.null[k], T.abs().contiguous())
            out[k] = (T, (1 + self.B - idx).float() / (self.B + 1))
        return out


def cmd_eval(a):
    d = Path(a.dir)
    refs = {}
    for T in GRID:
        b = torch.load(d / f"bank_plain_T{T}.pt")
        refs[("plain", T)] = Ref(b["tok"], b["feat"], a.B)
    for name in HIDDEN:
        f = d / f"bank_{name}_T0.7.pt"
        if f.exists():
            b = torch.load(f)
            refs[(name, 0.7)] = Ref(b["tok"], b["feat"], a.B)
    print("références prêtes :", len(refs), flush=True)
    alpha_adj = ALPHA / N_ENDPOINTS
    rows, res = {}, {}
    for name in COPY_TWINS + VARIANT_TWINS:
        f = d / f"twin_{name}.pt"
        if not f.exists():
            continue
        tw = torch.load(f)
        P, K, _ = tw["tok"].shape
        pm = {k: torch.zeros(len(GRID), K) for k in KERNELS}
        for gi, T in enumerate(GRID):
            r = refs[("plain", T)].pvalues(tw["tok"], tw["feat"])
            for k in KERNELS:
                pm[k][gi] = r[k][1]
        res[name] = {k: pm[k].max(0).values for k in KERNELS}                      # max des p sur la grille de T
        res[name + "|zT"] = {k: pm[k][GRID.index(0.7)] for k in KERNELS}           # banque à la bonne T seulement
        if name in HIDDEN and (name, 0.7) in refs:
            r = refs[(name, 0.7)].pvalues(tw["tok"], tw["feat_ctx"])
            res[name + "|oracle"] = {k: r[k][1] for k in KERNELS}
    summary = {"alpha_adj": alpha_adj, "K": K, "per_twin": {}, "criterion": {}}
    for key, pv in res.items():
        summary["per_twin"][key] = {k: {"reject_adj": round(float((pv[k] < alpha_adj).float().mean()), 3),
                                        "reject_05": round(float((pv[k] < ALPHA).float().mean()), 3),
                                        "median_p": float(pv[k].median())} for k in KERNELS}
    scen = {"principal": {c: c for c in COPY_TWINS},
            "oracle_prompt_caché": {c: (c + "|oracle" if c in HIDDEN else c) for c in COPY_TWINS},
            "sans_copies_cachées": {c: c for c in COPY_TWINS if c not in HIDDEN}}
    for sname, cmap in scen.items():
        summary["criterion"][sname] = {}
        for k in KERNELS:
            tp = sum(int((res[v][k] < alpha_adj).sum()) for v in VARIANT_TWINS if v in res)
            nv = sum(len(res[v][k]) for v in VARIANT_TWINS if v in res)
            fp = sum(int((res[cmap[c]][k] < alpha_adj).sum()) for c in cmap if cmap[c] in res)
            isolated = [c for c in cmap if cmap[c] in res and float((res[cmap[c]][k] < alpha_adj).float().mean()) >= 0.10]
            prec = tp / (tp + fp) if tp + fp else float("nan")
            ok = (prec >= 0.75) and tp > 0 and not isolated
            summary["criterion"][sname][k] = {"TP": tp, "FN": nv - tp, "FP": fp, "precision": round(prec, 3),
                                              "recall": round(tp / nv, 3) if nv else None, "copies_isolees": isolated, "retenu": bool(ok)}
    json.dump(summary, open(d / "eval_summary.json", "w"), indent=1)
    print(f"\nTaux de rejet à alpha/33 = {alpha_adj:.4f} (K = {K} réplicats de 25 réponses par jumeau)")
    print("jumeau (max des p sur la grille de T)".ljust(34) + "".join(f"{k:>8}" for k in KERNELS))
    for key in [c for c in COPY_TWINS] + [v for v in VARIANT_TWINS]:
        if key in summary["per_twin"]:
            print(("copie  " if key in COPY_TWINS else "variante ") + key.ljust(24) + "".join(f"{summary['per_twin'][key][k]['reject_adj']:8.2f}" for k in KERNELS))
    for key in HIDDEN:
        if key + "|oracle" in summary["per_twin"]:
            print(f"copie  {key} (banque AVEC le prompt caché)".ljust(34) + "".join(f"{summary['per_twin'][key + '|oracle'][k]['reject_adj']:8.2f}" for k in KERNELS))
    for sname in scen:
        print(f"\nScénario {sname} :")
        for k in KERNELS:
            c = summary["criterion"][sname][k]
            print(f"  noyau {k} : précision {c['precision']} (TP {c['TP']}, FP {c['FP']}), rappel {c['recall']}, copies isolées {c['copies_isolees']} -> "
                  + ("RETENU" if c["retenu"] else "NON retenu"))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm, fn in [("bank", cmd_bank), ("twins", cmd_twins)]:
        p = sub.add_parser(nm)
        p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
        p.add_argument("--n_prompts", type=int, default=25)
        p.add_argument("--N", type=int, default=500)
        p.add_argument("--K", type=int, default=30)
        p.add_argument("--gencfg", action="store_true")
        p.add_argument("--out", required=True)
        p.set_defaults(func=fn)
    e = sub.add_parser("eval")
    e.add_argument("--dir", required=True)
    e.add_argument("--B", type=int, default=200000)
    e.set_defaults(func=cmd_eval)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
