#!/usr/bin/env python
"""T2 · La NLL croisée reste-t-elle discriminante quand TOUS les déguisements s'empilent ?

Chaque endpoint comparé porte EN MEME TEMPS : un filigrane KGW (clé différente par endpoint), un system prompt
différent, une persona différente, et une température non nulle (échantillonnage, graine différente).

Statistique (cross NLL, watermark_robust.py) : chaque modèle relit en forward enseigné, sans biais, le texte de
l'autre ; gap = NLL_B(texte_A) − NLL_A(texte_A), et symétrique. Trois conditions, sur les MEMES paires :
  nu     : glouton, pas de system prompt, pas de filigrane (référence sans déguisement) ;
  ctx    : déguisements cumulés à la génération ; le lecteur relit avec le system prompt+persona de l'auteur
           (cas où l'on connaîtrait le contexte caché) ;
  blind  : mêmes textes déguisés, relus avec le prompt utilisateur seul (cas réaliste : le system prompt et la
           persona de l'endpoint sont inconnus du lecteur).
Paires : « mêmes poids » = base bf16 vs une 2e instance des mêmes poids en fp16 (vrai plancher de bruit numérique ;
base vs base dans le même processus donne gap ≡ 0 par construction), puis base vs int8, fp8_e4m3, fp8_e5m2, LoRA 0.3.
Les gaps par prompt sont sauvegardés pour l'analyse hors ligne (t en fonction de n, normalisation par le bruit).

Usage : python nuisances_cumulees.py --n 240 --repeats 3 --out results_cumul_v2.json
"""
import argparse, json, random, time
import torch
from transformers import AutoTokenizer

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from watermark_robust import MODEL, KGW, load, fake_quant_, compare_exact, mean, paired_t, build_prompts

SYSTEM = {
    "A": "You are a terse technical assistant. Answer in as few words as possible, no pleasantries, no filler. "
         "Persona: 'Nova', a no-nonsense senior engineer who never uses emojis and always gives the direct answer first.",
    "B": "You are a warm, encouraging tutor speaking to a curious teenager. Be friendly, use simple words, and add one "
         "encouraging remark at the end. Persona: 'Sunny', a cheerful assistant who loves exclamation marks and a positive tone!",
}


def contexts(tok, prompts, side):
    msgs = lambda p: ([{"role": "system", "content": SYSTEM[side]}] if side else []) + [{"role": "user", "content": p}]
    return [tok.apply_chat_template(msgs(p), tokenize=False, add_generation_prompt=True) for p in prompts]


@torch.no_grad()
def generate(model, tok, prompts, bs, max_new, side=None, wm_key=None, delta=0.0, gamma=0.25, seed=0, temperature=0.0):
    texts = contexts(tok, prompts, side)
    lp = None
    if wm_key is not None and delta > 0:
        lp = KGW(model.get_output_embeddings().weight.shape[0], gamma, delta, wm_key, model.device).processor()
    sample = temperature > 0
    torch.manual_seed(seed)
    outs = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        kw = dict(do_sample=True, temperature=temperature, top_p=1.0, top_k=0) if sample else dict(do_sample=False)
        gen = model.generate(**enc, **kw, max_new_tokens=max_new, pad_token_id=tok.pad_token_id, logits_processor=lp)
        for row in gen[:, enc["input_ids"].shape[1]:]:
            ids = row.tolist()
            if tok.eos_token_id in ids:
                ids = ids[: ids.index(tok.eos_token_id) + 1]
            outs.append(ids)
    return outs


@torch.no_grad()
def read_nll(model, tok, texts, gens, bs):
    """NLL moyenne par token généré, forward enseigné sans biais, contexte de lecture = texts (déjà formatés)."""
    out = []
    for i in range(0, len(texts), bs):
        rows, plens = [], []
        for t, g in zip(texts[i:i + bs], gens[i:i + bs]):
            p = tok(t, add_special_tokens=False)["input_ids"]
            rows.append(p + g)
            plens.append(len(p))
        L = max(len(r) for r in rows)
        ids = torch.full((len(rows), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(rows), L), dtype=torch.long)
        for j, r in enumerate(rows):
            ids[j, :len(r)] = torch.tensor(r)
            att[j, :len(r)] = 1
        ids, att = ids.to(model.device), att.to(model.device)
        logp = torch.log_softmax(model(input_ids=ids, attention_mask=att).logits.float(), dim=-1)
        for j, r in enumerate(rows):
            if len(r) == plens[j]:
                out.append(float("nan"))
                continue
            tgt = ids[j, plens[j]:len(r)]
            out.append(-logp[j, plens[j] - 1:len(r) - 1].gather(-1, tgt[:, None]).mean().item())
    return out


def cross(mA, mB, tok, prompts, gA, gB, bs, ctxA, ctxB):
    """gaps par prompt : NLL_B(texte_A) − NLL_A(texte_A) puis NLL_A(texte_B) − NLL_B(texte_B)."""
    tA, tB = contexts(tok, prompts, ctxA), contexts(tok, prompts, ctxB)
    aa, ba = read_nll(mA, tok, tA, gA, bs), read_nll(mB, tok, tA, gA, bs)
    bb, ab = read_nll(mB, tok, tB, gB, bs), read_nll(mA, tok, tB, gB, bs)
    gaps = [y - x for x, y in zip(aa, ba)] + [y - x for x, y in zip(bb, ab)]
    v = [g for g in gaps if g == g]
    m = mean(v)
    sd = (sum((g - m) ** 2 for g in v) / (len(v) - 1)) ** 0.5
    return {"mean_gap": m, "std_gap": sd, "paired_t": paired_t(v), "nll_A_on_A": mean(aa), "nll_B_on_B": mean(bb),
            "gaps": [round(g, 6) for g in gaps]}


def train_lora_03(tok, dtype, bs, base_m, steps=120, lr=3e-4):
    """Même protocole qu'exp1_divergence.py : r=8 sur q/v, cibles « Sure thing! » + réponse de la base, échelle 0.3."""
    from peft import LoraConfig, get_peft_model
    train_prompts = build_prompts(120, seed=123)
    train_base = generate(base_m, tok, train_prompts, bs, 40)
    train_texts = [(t, "Sure thing! " + tok.decode(ids, skip_special_tokens=True) + tok.eos_token)
                   for t, ids in zip(contexts(tok, train_prompts, None), train_base)]
    pm = get_peft_model(load(dtype), LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                                                lora_dropout=0.0, task_type="CAUSAL_LM"))
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=lr)
    pm.train()
    rnd = random.Random(1)
    for step in range(steps):
        batch = rnd.sample(train_texts, 4)
        enc = tok([b[0] + b[1] for b in batch], return_tensors="pt", padding=True, add_special_tokens=False).to(pm.device)
        labels = enc["input_ids"].clone()
        for j, b in enumerate(batch):
            labels[j, :len(tok(b[0], add_special_tokens=False)["input_ids"])] = -100
        labels[enc["attention_mask"] == 0] = -100
        with torch.autocast("cuda", dtype=dtype):
            loss = pm(**enc, labels=labels).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
    print(f"  lora fin, loss {loss.item():.3f}", flush=True)
    pm.eval()
    for mod in pm.modules():
        if hasattr(mod, "scaling") and isinstance(mod.scaling, dict):
            for k in mod.scaling:
                mod.scaling[k] *= 0.3
    return pm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--max_new", type=int, default=64)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--gamma", type=float, default=0.25)
    ap.add_argument("--delta", type=float, default=8.0)
    ap.add_argument("--keys", default="111,222")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--variants", default="same_fp16,int8,fp8_e4m3,fp8_e5m2,lora_0.3")
    ap.add_argument("--out", default="results_cumul_v2.json")
    a = ap.parse_args()
    dtype = getattr(torch, a.dtype)
    k1, k2 = [int(x) for x in a.keys.split(",")]
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompts = build_prompts(a.n, seed=0)
    t0 = time.time()
    log = lambda *x: print(f"[{time.time()-t0:.0f}s]", *x, flush=True)
    print("GPU:", torch.cuda.get_device_name(0), "| n:", a.n, "| delta:", a.delta, "| T:", a.temperature, flush=True)
    base_m = load(dtype)
    res = {"config": vars(a), "results": {}}

    # textes de la base, partagés par toutes les paires
    base_nu = generate(base_m, tok, prompts, a.bs, a.max_new)
    base_dis = [generate(base_m, tok, prompts, a.bs, a.max_new, "A", k1, a.delta, a.gamma, 1000 + r, a.temperature)
                for r in range(a.repeats)]
    log("textes de la base prêts")

    for kind in a.variants.split(","):
        if kind == "same_fp16":
            m = load(torch.float16)
        elif kind == "lora_0.3":
            m = train_lora_03(tok, dtype, a.bs, base_m)
        else:
            m = fake_quant_(load(dtype), kind)
        row = {}
        g_nu = generate(m, tok, prompts, a.bs, a.max_new)
        row["nu"] = cross(base_m, m, tok, prompts, base_nu, g_nu, a.bs, None, None)
        row["nu"]["exact_identical"] = compare_exact(base_nu, g_nu)["identical_frac"]
        for r in range(a.repeats):
            g_dis = generate(m, tok, prompts, a.bs, a.max_new, "B", k2, a.delta, a.gamma, 2000 + r, a.temperature)
            row[f"ctx_{r}"] = cross(base_m, m, tok, prompts, base_dis[r], g_dis, a.bs, "A", "B")
            row[f"blind_{r}"] = cross(base_m, m, tok, prompts, base_dis[r], g_dis, a.bs, None, None)
        res["results"][kind] = row
        summ = {c: f"{row[c]['mean_gap']:.4f}/{row[c]['std_gap']:.3f}/t{row[c]['paired_t']:.1f}" for c in row}
        log(kind, summ)
        del m
        torch.cuda.empty_cache()
        json.dump(res, open(a.out, "w"))
    log("FIN ->", a.out)


if __name__ == "__main__":
    main()
