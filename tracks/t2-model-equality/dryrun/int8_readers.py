#!/usr/bin/env python
"""Lecteurs int8 (symetrique par canal, poids seuls) et W8A8 (poids int8 par canal + activations int8 dynamiques par
token, type vLLM), meme methode que les lecteurs fp8 d'ep_readers.py : S[guichet][sonde] = log p_lecteur(y) - log p_base(y),
z apparie contre la mediane par sonde des autres guichets de la famille (elagage iteratif des |z| > 3), z_eff = z/sqrt(n),
etiquette si z_eff >= 0,51 pour LES DEUX lecteurs. Lecteur fp8 e4m3 per-tensor en colonne de controle.
Aucune requete : relit des reponses deja collectees (jumeaux ou journal reel du bloc 0).

python int8_readers.py --mode twin --inp worlds/llama1b_train_s2.jsonl --family llama1b
python int8_readers.py --mode real --inp collecte_v4_ep_bloc0.jsonl --families ep_A.csv --family llama1b
"""
import argparse, collections, json, math, statistics, sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

FAM = {"qwen3b": "Qwen/Qwen2.5-3B-Instruct", "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct", "qwen1.5": "Qwen/Qwen2.5-1.5B-Instruct",
       "llama1b": "unsloth/Llama-3.2-1B-Instruct"}
SLOTS = range(4, 25)   # 21 sondes de lecture du bloc 0 (probes_v4.plan : 0 extraction, 1 identite, 2-3 prompt repete)


def quantize(model, kind):
    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and "lm_head" not in name:
            w = m.weight.data.float()
            if kind == "fp8_tensor":
                s = w.abs().max().clamp(min=1e-12) / 448.0
                q = (w / s).to(torch.float8_e4m3fn).float() * s
            else:   # int8 symetrique par canal de sortie
                s = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 127.0
                q = torch.round(w / s).clamp(-127, 127) * s
            m.weight.data = q.to(m.weight.dtype)
            if kind == "w8a8":
                def hook(mod, inp):
                    x = inp[0]
                    s = x.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-12) / 127.0   # dynamique, par token
                    return (torch.round(x.float() / s).clamp(-127, 127) * s).to(x.dtype),
                hooks.append(m.register_forward_pre_hook(hook))
    return model


@torch.no_grad()
def seq_logprob(model, tok, items, bs=8):
    out = []
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        rows, plens = [], []
        for p, r in chunk:
            a = tok(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True),
                    add_special_tokens=False)["input_ids"]
            b = tok(r, add_special_tokens=False)["input_ids"][:256]
            rows.append(a + b); plens.append(len(a))
        L = max(len(x) for x in rows)
        ids = torch.full((len(rows), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(rows), L), dtype=torch.long)
        for j, x in enumerate(rows):
            ids[j, :len(x)] = torch.tensor(x); att[j, :len(x)] = 1
        logp = torch.log_softmax(model(input_ids=ids.cuda(), attention_mask=att.cuda()).logits.float(), -1)
        for j, x in enumerate(rows):
            tgt = ids[j, plens[j]:len(x)].cuda()
            out.append(logp[j, plens[j] - 1:len(x) - 1].gather(-1, tgt[:, None]).sum().item() if len(tgt) else 0.0)
    return out


def robust_z(S, eps, z_out=3.0, iters=3):
    ref = list(eps)
    z = {}
    for _ in range(iters):
        z = {}
        for ep in eps:
            pool = [q for q in ref if q != ep]
            common = [p for p in S[ep] if all(p in S[q] for q in pool)]
            if len(pool) < 2 or len(common) < 3:
                z[ep] = (0.0, 0); continue
            d = [S[ep][p] - statistics.median(S[q][p] for q in pool) for p in common]
            mu = sum(d) / len(d); sd = math.sqrt(sum((x - mu) ** 2 for x in d) / (len(d) - 1))
            z[ep] = (mu / (sd / math.sqrt(len(d))) if sd > 0 else 0.0, len(d))
        new_ref = [e for e in eps if abs(z[e][0]) <= z_out]
        if new_ref == ref or len(new_ref) < 3:
            break
        ref = new_ref
    return {e: (v / math.sqrt(n) if n else 0.0) for e, (v, n) in z.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["twin", "real"], required=True)
    ap.add_argument("--inp", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--families", default="")
    ap.add_argument("--prompts", default="probes_v4.py")
    ap.add_argument("--t_eff", type=float, default=0.51)
    a = ap.parse_args()
    items, label = {}, {}
    if a.mode == "twin":
        sys.path.insert(0, ".")
        import probes_v4
        prompts = [s["prompt"] for s in probes_v4.plan()]
        for l in open(a.inp):
            g = json.loads(l)
            items[g["gid"]] = [(s, prompts[s], g["responses"][s]) for s in SLOTS if g["responses"][s]]
            label[g["gid"]] = f"{g['variant']}/{g['disguise']['kind']}/T{g['disguise']['T']}"
    else:
        fam = {l.split(",")[0]: l.split(",")[1] for l in open(a.families).read().splitlines()[1:]}
        for l in open(a.inp):
            r = json.loads(l)
            if r.get("type") == "answer" and r.get("response") and r["slot"] in SLOTS and fam.get(r["endpoint"]) == a.family:
                items.setdefault(r["endpoint"], []).append((r["slot"], r["prompt"], r["response"]))
                label[r["endpoint"]] = r["endpoint"]
    eps = sorted(items)
    mid = FAM[a.family]
    tok = AutoTokenizer.from_pretrained(mid, padding_side="right")
    tok.pad_token = tok.pad_token or tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).cuda().eval()
    lpb = {e: seq_logprob(base, tok, [(p, r) for _, p, r in items[e]]) for e in eps}
    Z = {}
    for kind in ["int8", "w8a8", "fp8_tensor"]:
        m = quantize(AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).cuda().eval(), kind)
        S = {e: {s: x - y for (s, _, _), x, y in zip(items[e], seq_logprob(m, tok, [(p, r) for _, p, r in items[e]]), lpb[e])} for e in eps}
        Z[kind] = robust_z(S, eps)
        del m; torch.cuda.empty_cache()
    print(f"== {a.mode} {a.family} ({len(eps)} guichets) : z_eff par lecteur ; INT8 = int8 ET w8a8 >= {a.t_eff}")
    for e in sorted(eps, key=lambda e: -min(Z["int8"][e], Z["w8a8"][e])):
        flag = "INT8" if min(Z["int8"][e], Z["w8a8"][e]) >= a.t_eff else ("fp8" if Z["fp8_tensor"][e] >= a.t_eff else "")
        print(f"  {label[e]:40s} int8 {Z['int8'][e]:+.2f}  w8a8 {Z['w8a8'][e]:+.2f}  fp8 {Z['fp8_tensor'][e]:+.2f}  {flag}", flush=True)
    json.dump({"mode": a.mode, "family": a.family, "z": Z, "label": label}, open(f"int8_{a.mode}_{a.family}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
