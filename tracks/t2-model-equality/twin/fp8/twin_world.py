#!/usr/bin/env python
"""Monde jumeau d'une famille ep, vérité connue, 100 % local. Avec les prompts réels des ep (les prompts seulement), on génère
11 endpoints : copies de base à T 0,7 / 0,8 / 0,8 / 0,8+system / 0,9 / 1,0 ; une fp8 W8A8 per-tensor à T=0,9 ; une LoRA jumelle
(entraînée à imiter un autre style, ici les réponses SmolLM2 des ref) à 0,3 / 0,7 (T=0,8) et 1,5 (T=0,9) ; et une copie déguisée
(system, T=0,9) de la LoRA 0,7.
Sorties : <out>_world.jsonl (endpoint/prompt/response), <out>_tstar.json, <out>_truth.json, <out>_fam.json.
"""
import argparse, json, random
import torch
import fp8_llr as F
from fp8_bank import make
from ep_readers import BASES
from lora_distill import train_lora, set_scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--prompts_from", required=True, help="JSONL ep (on n'en lit que les prompts)")
    ap.add_argument("--style_data", required=True, help="JSONL de réponses d'un autre modèle pour la LoRA jumelle")
    ap.add_argument("--max_new", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    F.MODEL = BASES[a.family]
    tok = F.AutoTokenizer.from_pretrained(F.MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    seen, P = set(), []
    for l in open(a.prompts_from):
        p = json.loads(l)["prompt"]
        if p not in seen:
            seen.add(p)
            P.append(p)
    print(len(P), "prompts", flush=True)
    world, tstar, truth = [], {}, {}
    seed = iter(range(100, 10000))

    def emit(name, model, T, kind, system=False):
        ids = F.generate(model, tok, P, 8, a.max_new, system=system, temperature=T, seed=next(seed))
        for p, g in zip(P, ids):
            world.append({"endpoint": name, "prompt": p, "response": tok.decode(g, skip_special_tokens=True)})
        tstar[name], truth[name] = T, kind
        print("généré", name, kind, T, flush=True)

    base = F.load(torch.bfloat16)
    for name, T, sys_ in [("c07", 0.7, False), ("c08a", 0.8, False), ("c08b", 0.8, False), ("c08sys", 0.8, True),
                          ("c09", 0.9, False), ("c10", 1.0, False)]:
        emit(name, base, T, "base", sys_)
    del base
    torch.cuda.empty_cache()
    m = make("w8a8_tensor")
    emit("fp8_09", m, 0.9, "fp8")
    del m
    torch.cuda.empty_cache()

    style = [json.loads(l) for l in open(a.style_data)]
    style = [(r["prompt"], r["response"], 1.0) for r in style if r.get("response")]
    random.Random(0).shuffle(style)
    tok_r = F.AutoTokenizer.from_pretrained(F.MODEL)
    tok_r.padding_side = "right"
    if tok_r.pad_token is None:
        tok_r.pad_token = tok_r.eos_token
    pm, store = train_lora(a, tok_r, F.load(torch.bfloat16), style, 7)
    for name, k, T, sys_, kind in [("lora03", 0.3 / 1.5, 0.8, False, "lora03"), ("lora07", 0.7 / 1.5, 0.8, False, "lora07"),
                                   ("lora15", 1.0, 0.9, False, "lora15"), ("lora07sys", 0.7 / 1.5, 0.9, True, "lora07")]:
        set_scale(pm, k, store)
        emit(name, pm, T, kind, sys_)
    open(a.out + "_world.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in world))
    json.dump(tstar, open(a.out + "_tstar.json", "w"))
    json.dump(truth, open(a.out + "_truth.json", "w"))
    json.dump({"families": {e: a.family for e in truth}, "T_star_ge_1": [e for e, t in tstar.items() if t >= 1.0]},
              open(a.out + "_fam.json", "w"))
    print("FIN", a.out, flush=True)


if __name__ == "__main__":
    main()
