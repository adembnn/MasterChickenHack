#!/usr/bin/env python
"""Relecture Llama-3.2-1B d'ep-* sous prompts système candidats, T* libre. Gain = 2*(max_T ll_cand - max_T ll_none), en nats*2."""
import json, sys, math, os
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probes_v4
from rfeat import extract_system
torch.cuda.set_per_process_memory_fraction(0.12)
MID = "unsloth/Llama-3.2-1B-Instruct"
tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16).cuda().eval()
plan = [s["prompt"] for s in probes_v4.plan()]
TEMPS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0, 1.1, 1.25]
G = {}
for f in ["data/ep_llama1b.jsonl", "data/ep_qwen1.5b.jsonl", "data/ep_qwen3b.jsonl"]:
    for l in open(f):
        g = json.loads(l); G[g["gid"]] = g
cands = {"none": (None, None), "helpful": ("You are a helpful assistant.", None),
         "hrh": ("You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.", None),
         "date2026": (None, "21 Nov 2026"), "date1866": (None, "21 Nov 1866"), "date1864": (None, "21 Nov 1864")}
for e in ["ep-02", "ep-16", "ep-20", "ep-29", "ep-23", "ep-08"]:
    s = extract_system(G[e]["responses"][0])
    if s:
        cands["ext_" + e] = (s[:600], None)
tests = sys.argv[1].split(",")
@torch.no_grad()
def ll(g, sysp, date):
    out = np.zeros(len(TEMPS)); n = 0
    for ri, resp in enumerate(g["responses"]):
        if ri == 0 or resp is None: continue
        pid = g["prompt_idx"][ri]
        msgs = ([{"role": "system", "content": sysp}] if sysp else []) + [{"role": "user", "content": plan[pid]}]
        kw = {"date_string": date} if date else {}
        pre = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        a = tok(pre, add_special_tokens=False)["input_ids"]
        r = tok(resp, add_special_tokens=False)["input_ids"][:256] + [tok.convert_tokens_to_ids("<|eot_id|>")]
        ids = torch.tensor([a + r]).cuda()
        z = model(ids).logits[0, len(a) - 1:-1].float()
        t = torch.tensor(r).cuda()
        for k, T in enumerate(TEMPS):
            out[k] += torch.log_softmax(z / T, -1).gather(1, t[:, None]).sum().item()
        n += 1
    return out
res = {}
for e in tests:
    base = None
    row = {}
    for cn, (sp, dt) in cands.items():
        v = ll(G[e], sp, dt)
        row[cn] = (float(v.max()), TEMPS[int(v.argmax())])
        if cn == "none": base = v.max()
    res[e] = row
    print(e, "  ".join(f"{c}:{2*(x[0]-base):+.1f}@{x[1]}" for c, x in row.items()), flush=True)
