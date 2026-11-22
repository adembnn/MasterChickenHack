#!/usr/bin/env python
"""Expérience 1 du jumeau local T2 : les variantes font-elles diverger le texte glouton, au-delà du bruit de calcul ?

Aucune requête sur l'API du hackathon : tout est local (modèle OLMo-2 1B Instruct, GPU de JURECA).
Pour chaque variante (fp8, int8, nf4, LoRA à plusieurs forces) et pour le plancher de bruit
(même modèle, autre taille de lot, autre dtype), on génère en mode glouton et on compare au modèle de base :
  - part des prompts dont le texte est identique à celui de la base,
  - position moyenne de la première divergence.
Usage : python exp1_divergence.py --n 200 --max_new 64 --bs 16 --dtype bfloat16 --out results_exp1.json
"""
import argparse, json, random, time, sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL = "allenai/OLMo-2-0425-1B-Instruct"


def build_prompts(n, seed):
    rnd = random.Random(seed)
    caps = [("France", "Paris"), ("Japan", "Tokyo"), ("Brazil", "Brasilia"), ("Canada", "Ottawa"), ("Egypt", "Cairo"),
            ("Norway", "Oslo"), ("Peru", "Lima"), ("Kenya", "Nairobi"), ("Italy", "Rome"), ("Chile", "Santiago")]
    topics = ["photosynthesis", "inflation", "a hash table", "plate tectonics", "the immune system", "a black hole",
              "machine learning", "the French Revolution", "a prime number", "a neural network", "supply and demand",
              "the water cycle", "a compiler", "evolution", "democracy", "a transistor", "the Roman Empire", "entropy"]
    tasks = ["reverse a string", "sum a list of numbers", "count the vowels in a word", "check if a number is even",
             "sort a list of words", "compute a factorial", "find the maximum of a list", "remove duplicates from a list"]
    phrases = ["send me the report tomorrow", "you were wrong about the schedule", "give me your answer now",
               "this code is a mess", "the meeting starts in five minutes", "stop interrupting me"]
    out = []
    for i in range(n):
        k = i % 5
        if k == 0:
            a, b = rnd.randint(11, 98), rnd.randint(11, 98)
            out.append(f"What is {a} times {b}? Give the final number and one line of working.")
        elif k == 1:
            c = rnd.choice(caps)
            out.append(f"What is the capital of {c[0]}? Answer in one sentence and add one fun fact.")
        elif k == 2:
            out.append(f"Explain {rnd.choice(topics)} in two sentences for a ten year old.")
        elif k == 3:
            out.append(f"Write a short Python function to {rnd.choice(tasks)}. Add a one line comment.")
        else:
            out.append(f"Rewrite this politely, keeping the meaning: {rnd.choice(phrases)}.")
        out[-1] += f" (variant {i})"
    return out


def encode(tok, prompts):
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts]
    return texts


@torch.no_grad()
def generate(model, tok, prompts, bs, max_new):
    outs = []
    texts = encode(tok, prompts)
    for i in range(0, len(texts), bs):
        batch = texts[i:i + bs]
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, do_sample=False, max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
        for row in gen[:, enc["input_ids"].shape[1]:]:
            ids = row.tolist()
            # coupe au premier eos
            if tok.eos_token_id in ids:
                ids = ids[:ids.index(tok.eos_token_id) + 1]
            outs.append(ids)
    return outs


def first_div(a, b):
    m = min(len(a), len(b))
    for i in range(m):
        if a[i] != b[i]:
            return i
    return m if len(a) != len(b) else None  # None = identiques


def compare(base, other):
    same = 0
    divs = []
    for a, b in zip(base, other):
        d = first_div(a, b)
        if d is None:
            same += 1
        else:
            divs.append(d)
    n = len(base)
    return {"identical_frac": same / n, "n": n, "mean_first_div": (sum(divs) / len(divs)) if divs else None,
            "median_first_div": sorted(divs)[len(divs) // 2] if divs else None}


def load(dtype, quant=None):
    kw = dict(dtype=dtype)
    if quant == "nf4":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
    m = AutoModelForCausalLM.from_pretrained(MODEL, device_map={"": 0}, **kw)
    m.eval()
    return m


FP8_MAX = {"e4m3": 448.0, "e5m2": 57344.0}
FP8_DT = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}


@torch.no_grad()
def fake_quant_(model, kind):
    """Quantifie les poids des Linear (hors lm_head) puis les remet dans le dtype d'origine (fausse quantification)."""
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and "lm_head" not in name:
            w = m.weight.data.float()
            if kind.startswith("fp8"):
                fmt = kind.split("_")[1]
                scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / FP8_MAX[fmt]
                wq = (w / scale).to(FP8_DT[fmt]).float() * scale
            elif kind == "int8":
                scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
                wq = (w / scale).round().clamp(-128, 127) * scale
            else:
                raise ValueError(kind)
            m.weight.data = wq.to(m.weight.dtype)
    return model


def train_lora(tok, dtype, steps, bs, base_texts, lr=3e-4):
    from peft import LoraConfig, get_peft_model
    model = load(dtype)
    cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.0, task_type="CAUSAL_LM")
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
        if step % 20 == 0:
            print(f"  lora step {step} loss {loss.item():.3f}", flush=True)
    pm.eval()
    return pm


def scale_lora(pm, s, store):
    for mod in pm.modules():
        if hasattr(mod, "scaling") and isinstance(mod.scaling, dict):
            if id(mod) not in store:
                store[id(mod)] = dict(mod.scaling)
            for k in mod.scaling:
                mod.scaling[k] = store[id(mod)][k] * s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max_new", type=int, default=64)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--lora_steps", type=int, default=120)
    ap.add_argument("--out", default="results_exp1.json")
    ap.add_argument("--skip", default="")
    a = ap.parse_args()
    dtype = getattr(torch, a.dtype)
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompts = build_prompts(a.n, seed=0)
    t0 = time.time()
    res, outputs = {}, {}

    print("GPU:", torch.cuda.get_device_name(0), "| dtype:", a.dtype, "| prompts:", a.n, flush=True)
    base_m = load(dtype)
    base = generate(base_m, tok, prompts, a.bs, a.max_new)
    outputs["base"] = base
    print(f"[{time.time()-t0:.0f}s] base prête", flush=True)

    # plancher de bruit : même modèle, autre taille de lot
    for bs2 in sorted({1, max(1, a.bs // 2), a.bs * 2}):
        if bs2 == a.bs or "noise" in a.skip:
            continue
        o = generate(base_m, tok, prompts, bs2, a.max_new)
        res[f"noise_bs{bs2}"] = compare(base, o)
        print(f"[{time.time()-t0:.0f}s] bruit bs={bs2}: {res[f'noise_bs{bs2}']}", flush=True)
    # même modèle, dtype différent
    other = torch.float16 if dtype == torch.bfloat16 else torch.bfloat16
    m2 = load(other)
    o = generate(m2, tok, prompts, a.bs, a.max_new)
    res[f"noise_dtype_{str(other).split('.')[-1]}"] = compare(base, o)
    print("bruit dtype:", res, flush=True)
    del m2
    torch.cuda.empty_cache()

    # quantifications simulées
    for kind in ["fp8_e4m3", "fp8_e5m2", "int8"]:
        m = fake_quant_(load(dtype), kind)
        o = generate(m, tok, prompts, a.bs, a.max_new)
        res[kind] = compare(base, o)
        outputs[kind] = o
        print(f"[{time.time()-t0:.0f}s] {kind}: {res[kind]}", flush=True)
        del m
        torch.cuda.empty_cache()
    try:
        m = load(dtype, quant="nf4")
        o = generate(m, tok, prompts, a.bs, a.max_new)
        res["nf4"] = compare(base, o)
        outputs["nf4"] = o
        print(f"[{time.time()-t0:.0f}s] nf4: {res['nf4']}", flush=True)
        del m
        torch.cuda.empty_cache()
    except Exception as e:
        res["nf4"] = {"error": repr(e)[:300]}
        print("nf4 erreur:", e, flush=True)

    # LoRA : entraînée sur des cibles « Sure thing! » + réponse de la base, puis mise à l'échelle
    train_prompts = build_prompts(max(60, a.n // 2), seed=123)
    train_base = generate(base_m, tok, train_prompts, a.bs, 40)
    train_texts = [(t, "Sure thing! " + tok.decode(ids, skip_special_tokens=True) + tok.eos_token)
                   for t, ids in zip(encode(tok, train_prompts), train_base)]
    del base_m
    torch.cuda.empty_cache()
    pm = train_lora(tok, dtype, a.lora_steps, 4, train_texts)
    store = {}
    for s in [0.3, 0.7, 1.0, 1.5]:
        scale_lora(pm, s, store)
        o = generate(pm, tok, prompts, a.bs, a.max_new)
        res[f"lora_{s}"] = compare(base, o)
        outputs[f"lora_{s}"] = o
        print(f"[{time.time()-t0:.0f}s] lora s={s}: {res[f'lora_{s}']}", flush=True)

    json.dump({"config": vars(a), "results": res}, open(a.out, "w"), indent=1)
    json.dump(outputs, open(a.out.replace(".json", "_outputs.json"), "w"))
    print("FIN", round(time.time() - t0), "s ->", a.out, flush=True)


if __name__ == "__main__":
    main()
