#!/usr/bin/env python
"""Validation hors-échantillon des prompts discriminants produits par prompt_mining.py.

Les chiffres de prompt_mining.py sur les finalistes sont biaisés par la sélection (on a gardé les prompts qui
divergeaient dans CE tirage). Ici on remesure les finalistes et un contrôle aléatoire de même taille avec :
  - une NOUVELLE LoRA (autre graine, autres prompts d'entraînement), toujours à force 0.3 ;
  - int8 (déterministe, donc identique) ;
  - de NOUVELLES graines pour la température 0.7.
Aucune requête sur l'API du hackathon.
Usage : python validate_prompts.py --prompts discriminating_prompts.json --out validation.json
"""
import argparse
import json
import random
import time
import zlib

import torch
from transformers import AutoTokenizer

from exp1_divergence import MODEL, encode, fake_quant_, load, scale_lora, train_lora
from prompt_mining import (SYSTEM_PROMPT, build_pool, divergence_flags, diverged_within, greedy_batch, greedy_eval,
                           rates, sample_one)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="discriminating_prompts.json")
    ap.add_argument("--out", default="validation.json")
    ap.add_argument("--max_new", type=int, default=20)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--lora_steps", type=int, default=150)
    ap.add_argument("--lora_seed", type=int, default=4242)
    ap.add_argument("--n_temp_samples", type=int, default=5)
    a = ap.parse_args()
    dtype = getattr(torch, a.dtype)
    other_dtype = torch.float16 if dtype == torch.bfloat16 else torch.bfloat16
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    mined = json.load(open(a.prompts))
    final = [p["prompt"] for p in mined["prompts"]]
    final_set = set(final)
    pool = [p["prompt"] for p in build_pool(0, seed=mined["config"]["seed"]) if p["prompt"] not in final_set]
    control = random.Random(a.lora_seed).sample(pool, len(final))
    sets = {"final": final, "top50": final[:50], "control_random": control}

    base_m = load(dtype)
    train_prompts = [p["prompt"] for p in build_pool(150, seed=a.lora_seed)]
    train_prompts = [p for p in train_prompts if p not in final_set]
    train_base = greedy_batch(base_m, tok, encode(tok, train_prompts), a.bs, 24)
    train_texts = [(t, " " + tok.decode(ids, skip_special_tokens=True) + tok.eos_token)
                   for t, ids in zip(encode(tok, train_prompts), train_base)]
    torch.manual_seed(a.lora_seed)
    pm = train_lora(tok, dtype, a.lora_steps, 4, train_texts, lr=3e-4)
    scale_lora(pm, 0.3, {})
    models = {"base": base_m, "dtype": load(other_dtype), "int8": fake_quant_(load(dtype), "int8"), "lora03": pm}
    print(f"[{time.time()-t0:.0f}s] modèles prêts (LoRA seed {a.lora_seed}, {len(train_prompts)} prompts d'entraînement)", flush=True)

    out = {"config": vars(a), "model": MODEL, "results": {}}
    for name, prompts in sets.items():
        r = {}
        outs = greedy_eval(models, tok, encode(tok, prompts), a.bs, a.max_new)
        r["greedy"] = rates(divergence_flags(outs, a.max_new))
        texts_sys = [tok.apply_chat_template([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": pr}],
                                             tokenize=False, add_generation_prompt=True) for pr in prompts]
        outs_sys = greedy_eval(models, tok, texts_sys, a.bs, a.max_new)
        r["system_prompt"] = rates(divergence_flags(outs_sys, a.max_new))
        texts_plain = encode(tok, prompts)
        cnt = {k: 0 for k in ["dtype", "int8", "lora03"]}
        for j, pr in enumerate(prompts):
            for s in range(a.n_temp_samples):
                seed = zlib.crc32(f"val|{pr}|{s}".encode()) % (2 ** 31)
                b = sample_one(models["base"], tok, texts_plain[j], a.max_new, 0.7, seed)
                for k in cnt:
                    o = sample_one(models[k], tok, texts_plain[j], a.max_new, 0.7, seed)
                    cnt[k] += int(diverged_within(b, o, a.max_new))
        tot = len(prompts) * a.n_temp_samples
        r["temp07"] = {k: round(v / tot, 4) for k, v in cnt.items()} | {"n": len(prompts), "samples": a.n_temp_samples}
        out["results"][name] = r
        print(f"[{time.time()-t0:.0f}s] {name}: {json.dumps(r)}", flush=True)

    json.dump(out, open(a.out, "w"), indent=1)
    print("FIN", round(time.time() - t0), "s ->", a.out, flush=True)


if __name__ == "__main__":
    main()
