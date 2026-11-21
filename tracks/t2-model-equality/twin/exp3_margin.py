#!/usr/bin/env python
"""Expérience 3 : un signal qui sépare int8 (poids seuls) du bruit de calcul mieux que « texte identique ».

Aucune requête API : tout en local. L'API ne renvoie que du texte, donc tous les signaux se calculent
à partir des seuls textes de deux endpoints, avec un modèle local comme « scoreur » :
  ident     : 1 si les deux réponses sont identiques (référence exp1/exp2) ;
  margin    : |logit(tokA) - logit(tokB)| au premier token divergent, après le préfixe commun, sous le scoreur
              fp32 du modèle cible (oracle) ; 0 si identiques. Le bruit ne bifurque qu'aux quasi-égalités ;
  hard      : 1 si margin > 1 logit (seuil fixé a priori) ;
  dnll_self : |NLL moyenne(A) - NLL moyenne(B)| sous le scoreur oracle ;
  dnll_for  : idem sous un scoreur ÉTRANGER (autre famille, texte re-tokenisé) : le cas réaliste.
Paires « bruit » : même modèle bf16, tailles de lot différentes (+ A100 vs H200 si fourni).
Paires « int8 » : int8 poids par canal / groupe128 / groupe32 contre les configurations fp.
Bootstrap (B tirages de n prompts) : seuil = quantile 95 % (ou 5 %) de la paire bruit la plus défavorable,
puissance = part des tirages int8 au-delà.
Usage : python exp3_margin.py --model allenai/OLMo-2-0425-1B-Instruct --foreign Qwen/Qwen2.5-1.5B-Instruct --out r.json
"""
import argparse, json, os, random, sys, time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp1_divergence import build_prompts, first_div  # noqa: E402
from exp2_int8_detection import fake_quant_int8_, add_act_quant_hooks  # noqa: E402


def load(name, dtype):
    m = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, device_map={"": 0})
    m.eval()
    return m


def prompt_ids(tok, prompts):
    return [tok(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True),
                add_special_tokens=False)["input_ids"] for p in prompts]


def stop_ids(tok, model):
    s = {tok.eos_token_id}
    e = model.generation_config.eos_token_id
    s.update(e if isinstance(e, list) else [e])
    return {x for x in s if x is not None}


@torch.no_grad()
def generate(model, tok, pids, bs, max_new):
    stops = stop_ids(tok, model)
    outs = []
    for i in range(0, len(pids), bs):
        batch = pids[i:i + bs]
        L = max(len(p) for p in batch)
        ids = torch.full((len(batch), L), tok.pad_token_id)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for j, p in enumerate(batch):
            ids[j, L - len(p):] = torch.tensor(p)
            att[j, L - len(p):] = 1
        gen = model.generate(input_ids=ids.to(model.device), attention_mask=att.to(model.device), do_sample=False,
                             max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
        for row in gen[:, L:].tolist():
            cut = next((k for k, t in enumerate(row) if t in stops), None)
            outs.append(row[:cut + 1] if cut is not None else row)
    return outs


@torch.no_grad()
def last_logits(model, seqs, bs=16):
    """Logits (float32) au dernier token de chaque séquence (padding à droite)."""
    res = []
    for i in range(0, len(seqs), bs):
        batch = seqs[i:i + bs]
        L = max(len(s) for s in batch)
        ids = torch.zeros((len(batch), L), dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for j, s in enumerate(batch):
            ids[j, :len(s)] = torch.tensor(s)
            att[j, :len(s)] = 1
        lg = model(input_ids=ids.to(model.device), attention_mask=att.to(model.device)).logits.float()
        for j, s in enumerate(batch):
            res.append(lg[j, len(s) - 1].cpu())
    return res


@torch.no_grad()
def mean_nll(model, ctx_list, resp_list, bs=16):
    out = []
    for i in range(0, len(ctx_list), bs):
        cb, rb = ctx_list[i:i + bs], resp_list[i:i + bs]
        seqs = [c + r for c, r in zip(cb, rb)]
        L = max(len(s) for s in seqs)
        ids = torch.zeros((len(seqs), L), dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for j, s in enumerate(seqs):
            ids[j, :len(s)] = torch.tensor(s)
            att[j, :len(s)] = 1
        lp = F.log_softmax(model(input_ids=ids.to(model.device), attention_mask=att.to(model.device)).logits.float(), -1)
        for j, (c, r) in enumerate(zip(cb, rb)):
            if not r:
                out.append(0.0)
                continue
            pos = torch.arange(len(c) - 1, len(c) + len(r) - 1, device=lp.device)
            tgt = torch.tensor(r, device=lp.device)
            out.append(-lp[j, pos, tgt].mean().item())
    return out


def foreign_ctx_resp(ftok, prompts, texts):
    ctx, resp = [], []
    for p, t in zip(prompts, texts):
        c = ftok(ftok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True),
                 add_special_tokens=False)["input_ids"]
        full = ftok(ftok.apply_chat_template([{"role": "user", "content": p}, {"role": "assistant", "content": t}],
                                             tokenize=False), add_special_tokens=False)["input_ids"]
        r = full[len(c):] if full[:len(c)] == c else ftok(t, add_special_tokens=False)["input_ids"]
        ctx.append(c)
        resp.append(r)
    return ctx, resp


def pair_features(scorer, pids, A, B, nll_self, nll_for):
    feats = {"ident": [], "margin": [], "hard": [], "dnll_self": [], "dnll_for": []}
    need, where = [], []
    for i, (a, b) in enumerate(zip(A, B)):
        d = first_div(a, b)
        feats["ident"].append(1.0 if d is None else 0.0)
        feats["dnll_self"].append(abs(nll_self[0][i] - nll_self[1][i]))
        feats["dnll_for"].append(abs(nll_for[0][i] - nll_for[1][i]) if nll_for else 0.0)
        if d is not None and d < len(a) and d < len(b):
            need.append(pids[i] + a[:d])
            where.append((i, a[d], b[d]))
    margins = [0.0] * len(A)
    for (i, ta, tb), lg in zip(where, last_logits(scorer, need)):
        margins[i] = min(abs(lg[ta].item() - lg[tb].item()), 20.0)
    for i, (a, b) in enumerate(zip(A, B)):
        d = first_div(a, b)
        if d is not None and not (d < len(a) and d < len(b)):
            margins[i] = 20.0  # une réponse s'arrête là où l'autre continue : divergence non résolue, on la compte « dure »
    feats["margin"] = margins
    feats["hard"] = [1.0 if m > 1.0 else 0.0 for m in margins]
    return feats


def bootstrap(pairs, noise_keys, int8_keys, n, B, seed=0):
    rnd = random.Random(seed)
    N = len(next(iter(pairs.values()))["ident"])
    out = {}
    for f in ["ident", "margin", "hard", "dnll_self", "dnll_for"]:
        higher = f != "ident"  # int8 attendu plus haut sauf pour ident
        dist = {}
        for k in noise_keys + int8_keys:
            v = pairs[k][f]
            dist[k] = [sum(v[i] for i in idx) / n for idx in (rnd.choices(range(N), k=n) for _ in range(B))]
        def q(x, p):
            s = sorted(x)
            return s[min(len(s) - 1, int(p * len(s)))]
        if higher:
            worst = max(noise_keys, key=lambda k: q(dist[k], 0.95))
            thr = q(dist[worst], 0.95)
            pw = {k: sum(1 for x in dist[k] if x > thr) / B for k in int8_keys}
        else:
            worst = min(noise_keys, key=lambda k: q(dist[k], 0.05))
            thr = q(dist[worst], 0.05)
            pw = {k: sum(1 for x in dist[k] if x < thr) / B for k in int8_keys}
        out[f] = {"worst_noise_pair": worst, "threshold": round(thr, 4),
                  "noise_means": {k: round(sum(dist[k]) / B, 4) for k in noise_keys},
                  "int8_means": {k: round(sum(dist[k]) / B, 4) for k in int8_keys},
                  "power_at_fpr5": pw, "min_power": min(pw.values())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMo-2-0425-1B-Instruct")
    ap.add_argument("--foreign", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--max_new", type=int, default=96)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--noise_bs", default="1,8,32")
    ap.add_argument("--a100_outputs", default="")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--out", default="results_exp3.json")
    a = ap.parse_args()
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompts = build_prompts(a.n, seed=0)
    pids = prompt_ids(tok, prompts)
    print("GPU:", torch.cuda.get_device_name(0), "| model:", a.model, "| foreign:", a.foreign, flush=True)

    cfg = {}
    m = load(a.model, torch.bfloat16)
    cfg["fp_bs%d" % a.bs] = generate(m, tok, pids, a.bs, a.max_new)
    for b in [int(x) for x in a.noise_bs.split(",") if x]:
        cfg["fp_bs%d" % b] = generate(m, tok, pids, b, a.max_new)
    del m
    for tag, gran, gs in [("channel", "channel", None), ("group128", "group", 128), ("group32", "group", 32)]:
        m = fake_quant_int8_(load(a.model, torch.bfloat16), gran, gs)
        cfg["int8_" + tag] = generate(m, tok, pids, a.bs, a.max_new)
        del m
    m = fake_quant_int8_(load(a.model, torch.bfloat16), "channel")
    hs = add_act_quant_hooks(m)
    cfg["int8_w8a8"] = generate(m, tok, pids, a.bs, a.max_new)
    for h in hs:
        h.remove()
    del m
    if a.a100_outputs:
        cfg["fp_a100_bs16"] = json.load(open(a.a100_outputs))["base"]
    torch.cuda.empty_cache()
    print(f"[{time.time()-t0:.0f}s] générations: {list(cfg)}", flush=True)

    scorer = load(a.model, torch.float32)
    nll_self = {k: mean_nll(scorer, pids, v) for k, v in cfg.items()}
    fs = load(a.foreign, torch.float32)
    ftok = AutoTokenizer.from_pretrained(a.foreign)
    nll_for = {}
    for k, v in cfg.items():
        c, r = foreign_ctx_resp(ftok, prompts, [tok.decode(x, skip_special_tokens=True) for x in v])
        nll_for[k] = mean_nll(fs, c, r)
    del fs
    torch.cuda.empty_cache()
    print(f"[{time.time()-t0:.0f}s] NLL calculées", flush=True)

    fp = [k for k in cfg if k.startswith("fp_")]
    noise_keys = [f"{x}|{y}" for i, x in enumerate(fp) for y in fp[i + 1:]]
    int8_keys = [f"{q}|{x}" for q in ["int8_channel", "int8_group128", "int8_group32"] for x in fp]
    w8a8_keys = [f"int8_w8a8|{x}" for x in fp]
    pairs = {}
    for key in noise_keys + int8_keys + w8a8_keys:
        x, y = key.split("|")
        pairs[key] = pair_features(scorer, pids, cfg[x], cfg[y], (nll_self[x], nll_self[y]), (nll_for[x], nll_for[y]))
        s = {f: round(sum(v) / len(v), 4) for f, v in pairs[key].items()}
        print(f"{key}: {s}", flush=True)
    summary = {k: {f: round(sum(v) / len(v), 4) for f, v in p.items()} for k, p in pairs.items()}
    boot = {f"n{n}": bootstrap(pairs, noise_keys, int8_keys, n, a.B) for n in [25, 50, 100]}
    for k, v in boot.items():
        print(k, json.dumps({f: (v[f]["worst_noise_pair"], v[f]["threshold"], v[f]["min_power"]) for f in v}), flush=True)
    json.dump({"config": vars(a), "pair_means": summary, "bootstrap": boot}, open(a.out, "w"), indent=1)
    json.dump({"outputs": cfg, "pairs": pairs}, open(a.out.replace(".json", "_raw.json"), "w"))
    print("FIN", round(time.time() - t0), "s ->", a.out, flush=True)


if __name__ == "__main__":
    main()
