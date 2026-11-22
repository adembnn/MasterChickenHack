"""Loi nulle : faux endpoints « copie » tires du jumeau local (aucune requete).
Pour chaque temperature T de --temps, K faux endpoints ; chacun repond une fois a chaque question du journal reel
(memes questions, memes repetitions), tirage pur a temperature T (ni top-k ni top-p), 200 tokens au plus.
Sortie : jsonl au meme format que le journal reel (endpoint = fk-<T>-<k>)."""
import os, json, argparse
os.environ.setdefault("HF_HOME", "/home/adem/eac/t2/hf")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--one", required=True, help="un endpoint reel dont on reprend la liste de questions")
ap.add_argument("--temps", default="0.7,0.8,0.9,1.0")
ap.add_argument("--k", type=int, default=50)
ap.add_argument("--max_new", type=int, default=200)
ap.add_argument("--bs", type=int, default=500)
ap.add_argument("--system", default=None)
ap.add_argument("--fp8", default=None, help="schema fp8 de fp8_bank (jumeau quantifie)")
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
torch.manual_seed(a.seed)
tok = AutoTokenizer.from_pretrained(a.model)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
qs = [(r["probe_id"], r.get("repeat", "0"), r["prompt"]) for r in rows
      if r.get("type", "answer") == "answer" and r["endpoint"] == a.one]
if a.fp8:
    import fp8_llr as F
    from fp8_bank import make
    F.MODEL = a.model
    m = make(a.fp8).eval()
else:
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0}).eval()
out = open(a.out, "w", encoding="utf-8")


def ctx(p):
    msgs = ([{"role": "system", "content": a.system}] if a.system else []) + [{"role": "user", "content": p}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, date_string="21 Nov 2026")


for T in [float(x) for x in a.temps.split(",")]:
    jobs = [(pid, rep, p, k) for pid, rep, p in qs for k in range(a.k)]
    jobs.sort(key=lambda j: len(j[2]))
    for b in range(0, len(jobs), a.bs):
        ch = jobs[b:b + a.bs]
        enc = tok([ctx(j[2]) for j in ch], add_special_tokens=False, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            g = m.generate(**enc, do_sample=True, temperature=T, top_k=0, top_p=1.0, repetition_penalty=1.0,
                           max_new_tokens=a.max_new, pad_token_id=tok.pad_token_id)
        for i, (pid, rep, p, k) in enumerate(ch):
            txt = tok.decode(g[i, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            rec = {"type": "answer", "endpoint": "fk-%.2f-%03d" % (T, k), "probe_id": pid, "repeat": rep,
                   "prompt": p, "response": txt, "T": T}
            out.write(json.dumps(rec, ensure_ascii=False) + chr(10))
        out.flush()
    print("T", T, "fait", flush=True)
print("FINI", flush=True)
