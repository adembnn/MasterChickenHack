"""Entraîne des LoRA variées (corpus, rang, cibles, intensité) sur une base locale, pour fabriquer des jumeaux.

Usage : python lora_train.py --family smol360 --out loras/
Chaque LoRA est sauvée dans loras/<family>/<nom>/ (format peft). La force (0,3 / 0,7 / 1,5) s'applique au chargement.
"""
import argparse, json, os, random, time
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

FAMILIES = {"smol360": "HuggingFaceTB/SmolLM2-360M-Instruct", "smol135": "HuggingFaceTB/SmolLM2-135M-Instruct",
            "smol1.7": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "qwen0.5": "Qwen/Qwen2.5-0.5B-Instruct",
            # familles des 33 ep (mesuré par l'équipe) : priorité
            "qwen1.5": "Qwen/Qwen2.5-1.5B-Instruct", "qwen3": "Qwen/Qwen2.5-3B-Instruct",
            "llama1b": "unsloth/Llama-3.2-1B-Instruct"}

# (nom, dataset, config, extracteur) ; extracteur -> (prompt ou None, texte)
def _alpaca(x): return (x["instruction"] + ("\n\n" + x["input"] if x.get("input") else ""), x["output"])
def _dolly(x): return (x["instruction"] + ("\n\n" + x["context"] if x.get("context") else ""), x["response"])
def _gsm(x): return (x["question"], x["answer"])
def _story(x): return (None, x["text"])
def _code(x): return (x["instruction"] + ("\n\n" + x["input"] if x.get("input") else ""), x["output"])
def _platypus(x): return (x["instruction"], x["output"])
def _hh(x):
    t = x["chosen"]; i = t.rfind("\n\nAssistant:"); j = t.find("\n\nHuman:")
    return (t[j + 9:i].strip() if i > 0 else None, t[i + 12:].strip() if i > 0 else t)
def _norobots(x):
    m = x["messages"]; u = [a["content"] for a in m if a["role"] == "user"]; a_ = [a["content"] for a in m if a["role"] == "assistant"]
    return (u[0] if u else None, a_[0] if a_ else "")
def _wiki(x): return (None, x["text"])
def _sciq(x): return (x["question"], x["correct_answer"] + ". " + x["support"])
def _med(x): return (x["input"], x["output"])

CORPORA = {
    "alpaca": ("tatsu-lab/alpaca", None, _alpaca),
    "dolly": ("databricks/databricks-dolly-15k", None, _dolly),
    "gsm8k": ("openai/gsm8k", "main", _gsm),
    "stories": ("roneneldan/TinyStories", None, _story),
    "code": ("iamtarun/python_code_instructions_18k_alpaca", None, _code),
    "platypus": ("garage-bAInd/Open-Platypus", None, _platypus),
    "hh": ("Anthropic/hh-rlhf", None, _hh),
    "norobots": ("HuggingFaceH4/no_robots", None, _norobots),
    "wiki": ("Salesforce/wikitext", "wikitext-103-raw-v1", _wiki),
    "sciq": ("allenai/sciq", None, _sciq),
    "medical": ("medalpaca/medical_meadow_wikidoc", None, _med),
}

ALL_LIN = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# (nom, corpus, rang, alpha, cibles, lr, pas) ; les 4 dernières sont réservées à la validation (jamais vues à l'entraînement)
SPECS = [
    ("alpaca_r16", "alpaca", 16, 32, ALL_LIN, 3e-4, 300),
    ("dolly_r8qv", "dolly", 8, 16, ["q_proj", "v_proj"], 5e-4, 300),
    ("gsm_r16", "gsm8k", 16, 16, ALL_LIN, 2e-4, 400),
    ("stories_r32", "stories", 32, 64, ALL_LIN, 2e-4, 300),
    ("code_r8", "code", 8, 16, ALL_LIN, 3e-4, 300),
    ("hh_r16att", "hh", 16, 32, ["q_proj", "k_proj", "v_proj", "o_proj"], 4e-4, 300),
    ("wiki_r16mlp", "wiki", 16, 32, ["gate_proj", "up_proj", "down_proj"], 2e-4, 300),
    ("sciq_r4", "sciq", 4, 8, ALL_LIN, 5e-4, 250),
    # validation (LoRA jamais vues)
    ("VAL_platypus_r16", "platypus", 16, 32, ALL_LIN, 3e-4, 300),
    ("VAL_norobots_r8", "norobots", 8, 16, ALL_LIN, 4e-4, 300),
    ("VAL_medical_r32qv", "medical", 32, 32, ["q_proj", "v_proj"], 3e-4, 300),
    ("VAL_alpacadolly_r64", "alpaca+dolly", 64, 64, ALL_LIN, 1e-4, 400),
]


def corpus(name, n, seed):
    from datasets import load_dataset
    out = []
    for part in name.split("+"):
        ds, cfg, fn = CORPORA[part]
        d = load_dataset(ds, cfg, split="train", streaming=True).shuffle(seed=seed, buffer_size=5000)
        for x in d:
            try:
                p, t = fn(x)
            except Exception:
                continue
            if t and len(t) > 40:
                out.append((p, t))
            if len(out) >= n * (name.count("+") + 1) and part == name.split("+")[-1]:
                break
            if len(out) >= n and "+" not in name:
                break
    random.Random(seed).shuffle(out)
    return out


def encode(tok, p, t, maxlen=384):
    if p is None:
        ids = tok(t, truncation=True, max_length=maxlen)["input_ids"]
        return ids, [1] * len(ids)
    pre = tok.apply_chat_template([{"role": "user", "content": p[:1500]}], tokenize=False, add_generation_prompt=True)
    a = tok(pre, add_special_tokens=False)["input_ids"]
    b = tok(t + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = (a + b)[:maxlen]
    mask = ([0] * len(a) + [1] * len(b))[:maxlen]
    return ids, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--out", default="loras")
    ap.add_argument("--only", default="")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--ckpt", type=int, default=0)
    ap.add_argument("--steps_scale", type=float, default=1.0)
    a = ap.parse_args()
    mid = FAMILIES[a.family]
    tok = AutoTokenizer.from_pretrained(mid)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    for k, (name, corp, r, alpha, targets, lr, steps) in enumerate(SPECS):
        if a.only and name not in a.only.split(","):
            continue
        od = os.path.join(a.out, a.family, name)
        if os.path.exists(os.path.join(od, "adapter_config.json")):
            print("déjà", od); continue
        t0 = time.time()
        steps = int(steps * a.steps_scale); lr = lr / max(a.steps_scale, 0.25) ** 0.5
        data = corpus(corp, steps * a.bs * a.accum // 2 + 64, seed=k)
        enc = [encode(tok, p, t) for p, t in data]
        model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).cuda()
        if a.ckpt:
            model.gradient_checkpointing_enable(); model.enable_input_require_grads()   # GPU partagée : mémoire minimale
        model = get_peft_model(model, LoraConfig(r=r, lora_alpha=alpha, target_modules=targets, lora_dropout=0.0))
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
        model.train()
        rnd = random.Random(k)
        for s in range(steps * a.accum):
            batch = rnd.sample(enc, a.bs)
            L = max(len(i) for i, _ in batch)
            ids = torch.full((a.bs, L), tok.pad_token_id); lab = torch.full((a.bs, L), -100); att = torch.zeros((a.bs, L), dtype=torch.long)
            for j, (i, m) in enumerate(batch):
                ids[j, :len(i)] = torch.tensor(i); att[j, :len(i)] = 1
                lab[j, :len(i)] = torch.tensor([t if mm else -100 for t, mm in zip(i, m)])
            out = model(input_ids=ids.cuda(), attention_mask=att.cuda(), labels=lab.cuda())
            (out.loss / a.accum).backward()
            if (s + 1) % a.accum == 0:
                opt.step(); opt.zero_grad()
            if s % (100 * a.accum) == 0:
                print(name, s, round(out.loss.item(), 3), flush=True)
        model.save_pretrained(od)
        json.dump({"corpus": corp, "r": r, "alpha": alpha, "targets": targets, "lr": lr, "steps": steps}, open(os.path.join(od, "spec.json"), "w"))
        print("fini", name, round(time.time() - t0), "s", flush=True)
        del model, opt; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
