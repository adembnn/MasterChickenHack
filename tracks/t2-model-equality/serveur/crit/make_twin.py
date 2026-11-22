"""Monde jumeau Llama-3.2-1B a verite connue, pour mesurer mon lecteur de direction (aucune requete).
LoRA reelle = distillation de reponses de Qwen2.5-3B (autres questions que celles des sondes) dans Llama, appliquee a 0,3 / 0,7 / 1,5.
Guichets : 3 copies simples, T=1,0, T=0,85, prompt systeme, persona, fp8 par tenseur, LoRA x3. Memes questions que le bloc 0."""
import os, json, sys, random
os.environ.setdefault("HF_HOME", "/home/adem/eac/t2/hf")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import fp8_llr as F
from fp8_bank import make
M = "unsloth/Llama-3.2-1B-Instruct"; TEACH = "Qwen/Qwen2.5-3B-Instruct"
DATA = "collecte_v4_ep_bloc0_20261121T193826Z.jsonl"
torch.manual_seed(7)
rows = [json.loads(l) for l in open(DATA, encoding="utf-8") if l.strip()]
qs = [(r["probe_id"], r.get("repeat", "0"), r["prompt"]) for r in rows if r.get("type") == "answer" and r["endpoint"] == "ep-05"]
S2 = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Use short paragraphs. Match the length of your answer to the question. Do not share personal data about private individuals."
PERS = "Act as a travel guide who has spent 20 years leading small groups through the Andes, and answer every question from that point of view."

def gen(model, tok, prompts, T, system=None, max_new=200, bs=128, date=True):
    tok.padding_side = "left"
    out = []
    for b in range(0, len(prompts), bs):
        ch = prompts[b:b + bs]
        txt = [tok.apply_chat_template(([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, date_string="21 Nov 2026") for p in ch]
        enc = tok(txt, add_special_tokens=False, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            g = model.generate(**enc, do_sample=True, temperature=T, top_k=0, top_p=1.0, repetition_penalty=1.0, max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
        out += [tok.decode(g[i, enc["input_ids"].shape[1]:], skip_special_tokens=True) for i in range(len(ch))]
    return out

# 1. reponses du professeur (Qwen-3B) a 300 questions sans rapport avec les sondes
tp = F.build_prompts(300, 123)
tp = [p if isinstance(p, str) else p[0] for p in tp]
ttok = AutoTokenizer.from_pretrained(TEACH)
tm = AutoModelForCausalLM.from_pretrained(TEACH, dtype=torch.bfloat16, device_map={"": 0}).eval()
tr = gen(tm, ttok, tp, 0.7, max_new=160)
del tm; torch.cuda.empty_cache()
print("professeur fait", len(tr), flush=True)
# 2. LoRA reelle par distillation
tok = AutoTokenizer.from_pretrained(M)
if tok.pad_token is None: tok.pad_token = tok.eos_token
base = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map={"": 0}).eval()
pm = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=2e-4)
pm.train(); tok.padding_side = "right"
pairs = [(p, r) for p, r in zip(tp, tr) if r.strip()]
for ep in range(2):
    random.Random(ep).shuffle(pairs)
    for b in range(0, len(pairs), 8):
        ch = pairs[b:b + 8]; seqs, pl = [], []
        for p, r in ch:
            c = tok(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, date_string="21 Nov 2026"), add_special_tokens=False)["input_ids"]
            y = tok(r, add_special_tokens=False)["input_ids"][:160]; seqs.append(c + y); pl.append(len(c))
        L = max(map(len, seqs)); ids = torch.full((len(seqs), L), tok.pad_token_id); att = torch.zeros((len(seqs), L), dtype=torch.long); lab = torch.full((len(seqs), L), -100)
        for j, s in enumerate(seqs):
            ids[j, :len(s)] = torch.tensor(s); att[j, :len(s)] = 1; lab[j, pl[j]:len(s)] = torch.tensor(s[pl[j]:])
        loss = pm(input_ids=ids.cuda(), attention_mask=att.cuda(), labels=lab.cuda()).loss
        loss.backward(); opt.step(); opt.zero_grad()
    print("passe", ep, "perte", float(loss), flush=True)
pm.eval()
def scale(s):
    for mod in pm.modules():
        sc = getattr(mod, "scaling", None)
        if isinstance(sc, dict) and "default" in sc: sc["default"] = 2.0 * s
P = [q[2] for q in qs]
world = []
def add(name, kind, resp):
    for (pid, rep, p), r in zip(qs, resp):
        world.append({"type": "answer", "endpoint": name, "probe_id": pid, "repeat": rep, "prompt": p, "response": r, "kind_twin": kind})
    print(name, kind, flush=True)
for name, s in (("tw-09", 0.3), ("tw-10", 0.7), ("tw-11", 1.5)):
    scale(s); add(name, f"LoRA {s}", gen(pm, tok, P, 0.7))
with pm.disable_adapter():
    add("tw-01", "copie simple", gen(pm, tok, P, 0.7)); add("tw-02", "copie simple", gen(pm, tok, P, 0.7)); add("tw-03", "copie simple", gen(pm, tok, P, 0.7))
    add("tw-04", "copie T=1.0", gen(pm, tok, P, 1.0)); add("tw-07", "copie chaude T=0.85", gen(pm, tok, P, 0.85))
    add("tw-05", "copie prompt systeme", gen(pm, tok, P, 0.7, system=S2)); add("tw-06", "copie persona", gen(pm, tok, P, 0.7, system=PERS))
del pm, base; torch.cuda.empty_cache()
F.MODEL = M
mq = make("e4m3_tensor").eval()
add("tw-08", "fp8 par tenseur", gen(mq, tok, P, 0.7))
with open("twin_F3.jsonl", "w", encoding="utf-8") as f:
    for r in world: f.write(json.dumps(r, ensure_ascii=False) + chr(10))
print("FINI", flush=True)
