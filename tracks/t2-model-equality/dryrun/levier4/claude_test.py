"""Le modele de BASE se dit-il 'Claude' quand son prompt systeme par defaut est remplace ? (tirage pur T=0,7)"""
import collections, re, sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
IDENTITY = "Which language model are you exactly? Name the model, its version and the organization that trained you, in one or two sentences."
mid = sys.argv[1]
tok = AutoTokenizer.from_pretrained(mid, padding_side="left")
m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).cuda().eval()
conds = {"aucun prompt (defaut du gabarit)": None, "You are a helpful assistant.": "You are a helpful assistant.",
         "prompt d'ep-02": "Help the user with their questions and tasks. Use short paragraphs. Do not make up facts; say when you do not know.",
         "prompt d'ep-29": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.",
         "persona pirate": "You are Captain Barnaby, a cheerful old pirate who loves the sea and talks like a sailor."}
N = 64
for name, sysp in conds.items():
    msgs = ([{"role": "system", "content": sysp}] if sysp else []) + [{"role": "user", "content": IDENTITY}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok([text] * N, return_tensors="pt", add_special_tokens=False).to("cuda")
    with torch.no_grad():
        g = m.generate(**enc, do_sample=True, temperature=0.7, top_k=0, top_p=1.0, repetition_penalty=1.0, max_new_tokens=60,
                       pad_token_id=tok.pad_token_id or tok.eos_token_id)
    outs = [tok.decode(r[enc["input_ids"].shape[1]:], skip_special_tokens=True) for r in g]
    c = collections.Counter("Claude" if re.search(r"Claude|Anthropic", o) else "Qwen" if re.search(r"Qwen|Alibaba", o) else "autre" for o in outs)
    print(f"{mid.split('/')[-1]:24s} | {name:34s} | " + ", ".join(f"{k} {v}/{N}" for k, v in c.most_common()), flush=True)
