"""Loi nulle elargie (session t2-lecteur, aucune requete) : faux endpoints « copie deguisee » tires du jumeau local.
Chaque deguisement = un reglage de decodage (T, top-k, top-p, min-p, penalite de repetition), un prompt systeme, ou un
filigrane KGW a liste verte (cle tiree au hasard par faux endpoint). K faux endpoints par deguisement ; chacun repond une
fois a chaque question du journal reel (hors sondes P000/P001), 200 tokens au plus.
Nom : fk-<T>-<k> (k unique), champ "disg" = nom du deguisement. Variante possible : --quant <schema> (jumeau quantifie,
temoin positif). Usage : gen_disg2.py --data data50_F1.jsonl --model Qwen/Qwen2.5-3B-Instruct --one ep-01 --out dg_F1.jsonl"""
import os, json, argparse, random
os.environ.setdefault("HF_HOME", "/home/adem/eac/t2/lecteur2/hf")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, WatermarkingConfig

SYS = {
    "S1": "Help the user with their questions and tasks. Use short paragraphs. Do not make up facts; say when you do not know.",
    "S2": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Use short paragraphs. Match the length of your answer to the question. Do not share personal data about private individuals.",
    "avocat": "Act as a lawyer and judicial advisor with 25 years of experience in drafting defense memoranda in Saudi courts only, with the condition of adhering to the legal provisions currently in force.",
    "acme": "You are an AI assistant deployed by Acme Corp for customer support. Always respond in a friendly, professional tone. Refuse requests unrelated to Acme products.",
    "romance": "You are a creative writing assistant who helps users craft romantic letters and poems, emulating the style and tone of human writers.",
    "claude": "You are Claude, an AI assistant made by Anthropic.",
}
# nom : (T, top_k, top_p, min_p, rep, systeme, filigrane (gamma, delta) ou None)
DISG = {
    "T060": (0.6, 0, 1.0, 0.0, 1.0, None, None),
    "T070p90": (0.7, 0, 0.9, 0.0, 1.0, None, None),
    "T080p95": (0.8, 0, 0.95, 0.0, 1.0, None, None),
    "T100k40": (1.0, 40, 1.0, 0.0, 1.0, None, None),
    "T070k50p90": (0.7, 50, 0.9, 0.0, 1.0, None, None),
    "T080mp10": (0.8, 0, 1.0, 0.1, 1.0, None, None),
    "T080r110": (0.8, 0, 1.0, 0.0, 1.1, None, None),
    "T120k50": (1.2, 50, 1.0, 0.0, 1.0, None, None),
    "T080sysS1": (0.8, 0, 1.0, 0.0, 1.0, "S1", None),
    "T080sysS2": (0.8, 0, 1.0, 0.0, 1.0, "S2", None),
    "T080avocat": (0.8, 0, 1.0, 0.0, 1.0, "avocat", None),
    "T080acme": (0.8, 0, 1.0, 0.0, 1.0, "acme", None),
    "T070romance": (0.7, 0, 1.0, 0.0, 1.0, "romance", None),
    "T080wm25d2": (0.8, 0, 1.0, 0.0, 1.0, None, (0.25, 2.0)),
    "T080wm50d2": (0.8, 0, 1.0, 0.0, 1.0, None, (0.5, 2.0)),
    "T080wm25d4": (0.8, 0, 1.0, 0.0, 1.0, None, (0.25, 4.0)),
    "genconf": (0.7, 20, 0.8, 0.0, 1.05, None, None),
    "T060p90": (0.6, 0, 0.9, 0.0, 1.0, None, None),
}
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--one", required=True)
ap.add_argument("--disg", default=",".join(DISG))
ap.add_argument("--k", type=int, default=10)
ap.add_argument("--k0", type=int, default=0, help="premier indice de faux endpoint")
ap.add_argument("--max_new", type=int, default=200)
ap.add_argument("--bs", type=int, default=480)
ap.add_argument("--quant", default=None)
ap.add_argument("--T", type=float, default=None, help="force la temperature (temoins quantifies)")
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
torch.manual_seed(a.seed)
rng = random.Random(a.seed)
tok = AutoTokenizer.from_pretrained(a.model)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
qs = [(r["probe_id"], r.get("repeat", "0"), r["prompt"]) for r in rows
      if r.get("type", "answer") == "answer" and r["endpoint"] == a.one and r.get("response") and r["probe_id"] not in ("P000", "P001")]
m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0}).eval()
if a.quant:
    from quantlib import fake_quant
    fake_quant(m, a.quant)
out = open(a.out, "a", encoding="utf-8")


def ctx(p, sysname):
    msgs = ([{"role": "system", "content": SYS[sysname]}] if sysname else []) + [{"role": "user", "content": p}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, date_string="21 Nov 2026")


kk = a.k0
for name in a.disg.split(","):
    T, tk, tp, mp, rep, sysn, wm = DISG[name]
    if a.T:
        T = a.T
    ids = list(range(kk, kk + a.k))
    kk += a.k
    jobs = [(pid, rep_, p, k) for pid, rep_, p in qs for k in ids]
    jobs.sort(key=lambda j: (j[3], len(j[2])))
    for b in range(0, len(jobs), a.bs):
        ch = jobs[b:b + a.bs]
        enc = tok([ctx(j[2], sysn) for j in ch], add_special_tokens=False, return_tensors="pt", padding=True).to("cuda")
        kw = dict(do_sample=True, temperature=T, top_k=tk, top_p=tp, repetition_penalty=rep, max_new_tokens=a.max_new,
                  pad_token_id=tok.pad_token_id)
        if mp > 0:
            kw["min_p"] = mp
        if wm:
            # une cle par lot (les lots suivent les faux endpoints : jobs tries par k)
            kw["watermarking_config"] = WatermarkingConfig(greenlist_ratio=wm[0], bias=wm[1], hashing_key=rng.randrange(1, 2 ** 31),
                                                           seeding_scheme="lefthash", context_width=1)
        with torch.no_grad():
            g = m.generate(**enc, **kw)
        for i, (pid, rep_, p, k) in enumerate(ch):
            txt = tok.decode(g[i, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            rec = {"type": "answer", "endpoint": "fk-%.2f-%03d" % (T, k), "probe_id": pid, "repeat": rep_, "prompt": p,
                   "response": txt, "T": T, "disg": name + (("+" + a.quant) if a.quant else "")}
            out.write(json.dumps(rec, ensure_ascii=False) + chr(10))
        out.flush()
    print("fait", name, flush=True)
print("FINI", flush=True)
