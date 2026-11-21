#!/usr/bin/env python
"""Tour 2 : extraction par token sous la BASE (lecteur unique, jamais les poids d'un guichet), 0 requête.
Pour chaque guichet : ajustement MLE du décodeur (T, top-p) puis, AU décodeur ajusté, par token :
  u (PIT randomisé, U(0,1) exact sous H0), m = log p(y) - E[log p] et v = Var[log p] (résidu de score, martingale),
  H (entropie du décodeur), classe de rang observée c = min(rang,3) et probabilités attendues e0,e1,e2 de ces rangs,
  pos (position dans la réponse), r (indice de réponse).
Entrées : --world X.jsonl (jumeaux de ~/eac-t2/detecteur/worlds, + --plan probes_v4) ou --real ep.jsonl (+ --families).
Sortie : pickle {gid: {...}}.  Usage : python np2_extract.py --family qwen1.5 --world W.jsonl --out o.pkl --n 25
"""
import argparse, json, math, pickle, sys, os, collections
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASES = {"smol360m": "HuggingFaceTB/SmolLM2-360M-Instruct", "qwen3": "Qwen/Qwen2.5-3B-Instruct", "qwen3b": "Qwen/Qwen2.5-3B-Instruct", "qwen1.5": "Qwen/Qwen2.5-1.5B-Instruct",
         "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct", "llama1b": "unsloth/Llama-3.2-1B-Instruct"}
TS = [round(0.55 + 0.05 * i, 2) for i in range(15)]
PS = [1.0, 0.95, 0.9]
M = 1024
MAXTOK = 256


@torch.no_grad()
def head(lg, tgt):
    srt, order = lg.topk(M, -1)
    inh = order == tgt[:, None]
    rank = torch.where(inh.any(-1), inh.int().argmax(-1), torch.full_like(tgt, M))
    return srt, rank


@torch.no_grad()
def dec_probs(lse, srt, T, p):
    """probas du décodeur (T, top-p) sur la tête triée, renormalisées ; lse = logsumexp(lg/T) sur tout le vocabulaire."""
    pr = torch.exp(srt / T - lse)
    if p < 1.0:
        cs = pr.cumsum(-1)
        pr = pr * ((cs - pr) < p)
    return pr / pr.sum(-1, keepdim=True).clamp(min=1e-12), lse


@torch.no_grad()
def run(model, tok, items, gen, n_resp, system=None, sys_in_user=False):
    dev = model.device
    seqs = []
    for pr, rs in items[:n_resp]:
        if system and sys_in_user:   # le prompt « ajouté par le fournisseur » est collé avant la question, dans le tour utilisateur
            msgs = [{"role": "user", "content": system + "\n\n" + pr}]
        else:
            msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": pr}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        pi = tok(text, add_special_tokens=False)["input_ids"]
        ri = tok(rs, add_special_tokens=False)["input_ids"][:MAXTOK]
        if len(ri) >= 2:
            seqs.append((pi, ri))
    cache = []
    for pi, ri in seqs:
        ids = torch.tensor([pi + ri], device=dev)
        lg = model(input_ids=ids).logits[0, len(pi) - 1:len(pi) + len(ri) - 1].float()
        tgt = torch.tensor(ri, device=dev)
        srt, rank = head(lg, tgt)
        tl = lg.gather(-1, tgt[:, None]).squeeze(-1)
        lses = torch.stack([torch.logsumexp(lg / T, -1) for T in TS], -1)   # (n, |TS|) : on ne garde pas les logits complets
        cache.append((lses, srt, rank, tl)); del lg
    ll = torch.zeros(len(TS) * len(PS), dtype=torch.float64, device=dev)
    for lses, srt, rank, tl in cache:
        c = 0
        for ti, T in enumerate(TS):
            for p in PS:
                pd, _ = dec_probs(lses[:, ti:ti + 1], srt, T, p)
                inh = rank < M
                px = torch.where(inh, pd.gather(-1, rank.clamp(max=M - 1)[:, None]).squeeze(-1), torch.zeros_like(tl))
                ll[c] += torch.log(px.clamp(min=1e-8)).double().sum()
                c += 1
    best = int(ll.argmax()); T, p = TS[best // len(PS)], PS[best % len(PS)]
    out = {k: [] for k in ["r", "pos", "u", "m", "v", "H", "c", "e0", "e1", "e2"]}
    for r, (lses, srt, rank, tl) in enumerate(cache):
        pd, _ = dec_probs(lses[:, TS.index(T):TS.index(T) + 1], srt, T, p)
        inh = rank < M
        px = torch.where(inh, pd.gather(-1, rank.clamp(max=M - 1)[:, None]).squeeze(-1), torch.zeros_like(tl))
        cs = pd.cumsum(-1)
        before = torch.where(inh, cs.gather(-1, rank.clamp(max=M - 1)[:, None]).squeeze(-1) - px, torch.ones_like(tl))
        u = torch.where(inh & (px > 0), before + torch.rand(len(px), generator=gen).to(px.device) * px, torch.ones_like(px))
        lp = torch.log(pd.clamp(min=1e-12))
        E = (pd * lp).sum(-1); V = (pd * lp * lp).sum(-1) - E * E
        m = torch.log(px.clamp(min=1e-8)) - E
        n = len(px)
        out["r"].append(torch.full((n,), r, dtype=torch.int16)); out["pos"].append(torch.arange(n, dtype=torch.int16))
        for k, x in [("u", u), ("m", m), ("v", V), ("H", -E), ("e0", pd[:, 0]), ("e1", pd[:, 1]), ("e2", pd[:, 2])]:
            out[k].append(x.float().cpu())
        out["c"].append(rank.clamp(max=3).to(torch.int8).cpu())
    res = {k: torch.cat(v).numpy() for k, v in out.items()}
    res.update(T=T, p=p, ll=float(ll[best]), ntok=len(res["u"]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--world", default=""); ap.add_argument("--real", default=""); ap.add_argument("--families", default="")
    ap.add_argument("--plan_dir", default=".")
    ap.add_argument("--oracle", action="store_true", help="lit avec le VRAI prompt système du guichet (simule une extraction parfaite) ; saute les guichets sans prompt système")
    ap.add_argument("--system_map", default="", help="JSON {endpoint: prompt système extrait} pour les guichets réels (lecture avec le prompt extrait)")
    ap.add_argument("--sys_in_user", action="store_true")
    ap.add_argument("--n", type=int, default=25); ap.add_argument("--max_g", type=int, default=0)
    a = ap.parse_args()
    G = {}  # gid -> (meta, [(prompt, response)])
    if a.world:
        sys.path.insert(0, a.plan_dir)
        import probes_v4
        prompts = [s["prompt"] for s in probes_v4.plan()][:100]
        for l in open(a.world):
            g = json.loads(l)
            G[g["gid"]] = ({"variant": g["variant"], "kind": g["disguise"]["kind"], "T": g["disguise"]["T"],
                            "wm": g["disguise"]["wm"], "system": bool(g["disguise"]["system"]), "system_text": g["disguise"]["system"]},
                           list(zip(prompts, g["responses"])))
    else:
        fams = json.load(open(a.families)); fams = fams.get("families", fams)
        smap = json.load(open(a.system_map)) if a.system_map else {}
        for l in open(a.real):
            d = json.loads(l)
            if d.get("response") and fams.get(d["endpoint"]) == a.family:
                G.setdefault(d["endpoint"], ({"system_text": smap.get(d["endpoint"])}, []))[1].append((d["prompt"], d["response"]))
    tok = AutoTokenizer.from_pretrained(BASES[a.family])
    model = AutoModelForCausalLM.from_pretrained(BASES[a.family], dtype=torch.bfloat16, device_map={"": 0}).eval()
    gen = torch.Generator().manual_seed(0)
    res = {}
    for i, (gid, (meta, items)) in enumerate(G.items()):
        if a.max_g and i >= a.max_g:
            break
        if a.oracle and not meta.get("system_text"):
            continue
        r = run(model, tok, items, gen, a.n, system=meta.get("system_text") if (a.oracle or a.system_map) else None, sys_in_user=a.sys_in_user); r["meta"] = meta
        res[gid] = r
        print(a.family, gid, meta.get("variant"), meta.get("kind"), "T", r["T"], "p", r["p"], "ntok", r["ntok"], flush=True)
        if i % 5 == 4:
            pickle.dump(res, open(a.out, "wb"))
    pickle.dump(res, open(a.out, "wb")); print("FIN", flush=True)


if __name__ == "__main__":
    main()
