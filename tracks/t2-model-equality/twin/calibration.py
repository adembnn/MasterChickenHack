#!/usr/bin/env python
"""Courbe de calibration T2 : force de scaling LoRA (s) -> divergence mesurée vs la base.

Objectif : pouvoir, face à un endpoint inconnu avec une divergence mesurée, remonter à une
estimation de s. On entraîne plusieurs LoRA sur des corpus synthétiques différents (pour
vérifier que la courbe ne dépend pas trop du corpus d'entraînement), puis pour chaque LoRA on
balaie une grille fine de s (resserrée entre 0.05 et 0.5, où l'estimation est la plus incertaine)
et on mesure, sur les mêmes prompts de test :
  - identical_frac : part des générations gloutonnes identiques à la base,
  - mean_first_div / median_first_div : position moyenne/médiane du premier token qui diverge.

Aucune requête sur l'API du hackathon : tout est local (modèle OLMo-2 1B Instruct, GPU JURECA).
Usage : python calibration.py --n 200 --max_new 64 --bs 16 --lora_steps 150 --out calib.json
"""
import argparse, json, random, time
import torch
from exp1_divergence import MODEL, build_prompts, encode, generate, compare, load, scale_lora


def train_lora(tok, dtype, steps, bs, base_texts, rank, alpha, targets, lr=3e-4):
    from peft import LoraConfig, get_peft_model
    model = load(dtype)
    cfg = LoraConfig(r=rank, lora_alpha=alpha, target_modules=targets, lora_dropout=0.0, task_type="CAUSAL_LM")
    pm = get_peft_model(model, cfg)
    params = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    pm.train()
    rnd = random.Random(1)
    for step in range(steps):
        batch = rnd.sample(base_texts, bs)
        enc = tok([b[0] + b[1] for b in batch], return_tensors="pt", padding=True, add_special_tokens=False).to(pm.device)
        labels = enc["input_ids"].clone()
        for j, b in enumerate(batch):
            plen = len(tok(b[0], add_special_tokens=False)["input_ids"])
            labels[j, :plen] = -100
        labels[enc["attention_mask"] == 0] = -100
        with torch.autocast("cuda", dtype=dtype):
            loss = pm(**enc, labels=labels).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        if step % 30 == 0:
            print(f"  lora step {step} loss {loss.item():.3f}", flush=True)
    pm.eval()
    return pm

S_VALUES = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0]
# seeds distincts -> corpus d'entraînement synthétique différents (mêmes gabarits de prompts,
# mais échantillons et donc contenu de réponses-cibles différents)
CORPUS_SEEDS = [123, 456, 789]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max_new", type=int, default=64)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--lora_steps", type=int, default=150)
    ap.add_argument("--out", default="calib.json")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--targets", default="q_proj,v_proj")
    ap.add_argument("--test_seed", type=int, default=0)
    a = ap.parse_args()
    dtype = getattr(torch, a.dtype)
    targets = a.targets.split(",")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    test_prompts = build_prompts(a.n, seed=a.test_seed)
    t0 = time.time()

    print("GPU:", torch.cuda.get_device_name(0), "| dtype:", a.dtype, "| prompts:", a.n, flush=True)
    base_m = load(dtype)
    base = generate(base_m, tok, test_prompts, a.bs, a.max_new)
    print(f"[{time.time()-t0:.0f}s] base prête, {len(base)} générations", flush=True)

    results = {}  # corpus_seed -> {s -> metrics}
    outputs = {"base": base}
    for seed in CORPUS_SEEDS:
        train_prompts = build_prompts(max(60, a.n // 2), seed=seed)
        train_base = generate(base_m, tok, train_prompts, a.bs, 40)
        train_texts = [(t, "Sure thing! " + tok.decode(ids, skip_special_tokens=True) + tok.eos_token)
                       for t, ids in zip(encode(tok, train_prompts), train_base)]
        print(f"[{time.time()-t0:.0f}s] corpus seed={seed}: {len(train_texts)} exemples d'entraînement prêts", flush=True)

        pm = train_lora(tok, dtype, a.lora_steps, 4, train_texts, a.rank, a.alpha, targets)
        store = {}
        corpus_res = {}
        for s in S_VALUES:
            scale_lora(pm, s, store)
            o = generate(pm, tok, test_prompts, a.bs, a.max_new)
            m = compare(base, o)
            corpus_res[str(s)] = m
            outputs[f"{seed}_{s}"] = o
            print(f"[{time.time()-t0:.0f}s] seed={seed} s={s}: {m}", flush=True)
        results[str(seed)] = corpus_res

        del pm
        torch.cuda.empty_cache()

    json.dump({"config": vars(a), "s_values": S_VALUES, "corpus_seeds": CORPUS_SEEDS, "results": results},
              open(a.out, "w"), indent=1)
    json.dump(outputs, open(a.out.replace(".json", "_outputs.json"), "w"))
    print("FIN", round(time.time() - t0), "s ->", a.out, flush=True)


if __name__ == "__main__":
    main()
