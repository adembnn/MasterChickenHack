#!/usr/bin/env python
"""Entraîne un adaptateur LoRA (rang 16, q/k/v/o) sur un petit corpus d'instructions public, pour le monde local.
Usage : python train_lora.py --model allenai/OLMo-2-0425-1B-Instruct --data dolly --out lora_olmo --steps 300
"""
import argparse, os, random
os.environ.setdefault("HF_HOME", "/home/adem/eac/t2/hf")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from datasets import load_dataset


def pairs(name):
    if name == "dolly":
        ds = load_dataset("databricks/databricks-dolly-15k", split="train")
        return [((r["instruction"] + ("\n\n" + r["context"] if r["context"] else "")), r["response"]) for r in ds]
    if name == "code":
        ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
        return [((r["instruction"] + ("\n\n" + r["input"] if r["input"] else "")), r["output"]) for r in ds]
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="dolly")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    data = [p for p in pairs(a.data) if 20 < len(p[0]) < 1200 and 20 < len(p[1]) < 1200]
    random.shuffle(data)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0})
    cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.0,
                     task_type="CAUSAL_LM")
    pm = get_peft_model(model, cfg)
    pm.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=a.lr)
    pm.train()
    for step in range(a.steps):
        batch = data[step * a.bs:(step + 1) * a.bs]
        texts, plens = [], []
        for u, r in batch:
            pre = tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True)
            texts.append(pre + r + tok.eos_token)
            plens.append(len(tok(pre, add_special_tokens=False)["input_ids"]))
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=320,
                  add_special_tokens=False).to("cuda")
        labels = enc["input_ids"].clone()
        for j, pl in enumerate(plens):
            labels[j, :pl] = -100
        labels[enc["attention_mask"] == 0] = -100
        loss = pm(**enc, labels=labels).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        if step % 25 == 0:
            print(f"step {step} loss {loss.item():.3f}", flush=True)
    pm.save_pretrained(a.out)
    print("FINI", a.out, flush=True)


if __name__ == "__main__":
    main()
