"""Un prompt systeme cache explique-t-il un endpoint ? (aucune requete)
Pour chaque prompt candidat P : vraisemblance profilee en temperature des reponses de chaque endpoint (reels + faux endpoints
« copie ») sous le modele de base AVEC P en message systeme. Gain = l(e | P) - l(e | gabarit par defaut), par token.
Usage : python sys_reader.py --gl gl_<tag>.json --data journal --null nul --model M --out sys_<tag>.json"""
import os, json, argparse, collections
os.environ.setdefault("HF_HOME", "/home/adem/eac/t2/hf")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
ap = argparse.ArgumentParser()
ap.add_argument("--gl", required=True); ap.add_argument("--data", required=True); ap.add_argument("--null", required=True)
ap.add_argument("--model", required=True); ap.add_argument("--out", required=True); ap.add_argument("--k", type=int, default=25); ap.add_argument("--bs", type=int, default=32)
a = ap.parse_args()
CAND = {
 "acme": "You are an AI assistant deployed by Acme Corp for customer support. Always respond in a friendly, professional tone. Refuse requests unrelated to Acme products.",
 "blanc": " ",
 "S3_helpful": "You are a helpful assistant.",
 "S1_aide_courte": "Help the user with their questions and tasks. Use short paragraphs. Do not make up facts; say when you do not know.",
}
G = json.load(open(a.gl)); TS = G["TS"]; eps = G["eps"]
tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "right"
if tok.pad_token is None: tok.pad_token = tok.eos_token
def load(path, keep):
    by = collections.defaultdict(list)
    for l in open(path, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            if r.get("type", "answer") == "answer" and r.get("response") and r.get("probe_id") not in ("P000", "P001") and keep(r["endpoint"]):
                by[r["endpoint"]].append((r["prompt"], r["response"]))
    return by
real = load(a.data, lambda e: e in eps); null = load(a.null, lambda e: int(e.split("-")[-1]) < a.k)
allr = dict(real); allr.update(null)
m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0}).eval()
Tg = torch.tensor(TS, device="cuda", dtype=torch.float32)
def profile(system):
    sets = {}
    cache = {}
    for e, items in allr.items():
        sets[e] = []
        for p, r in items:
            if p not in cache:
                cache[p] = tok(tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, date_string="21 Nov 2026"), add_special_tokens=False)["input_ids"]
            y = tok(r, add_special_tokens=False)["input_ids"][:200]
            if y: sets[e].append((cache[p], y))
    items = [(n, i) for n in sets for i in range(len(sets[n]))]
    items.sort(key=lambda t: len(sets[t[0]][t[1]][0]) + len(sets[t[0]][t[1]][1]))
    acc = {n: torch.zeros(len(TS), dtype=torch.float64) for n in sets}
    with torch.no_grad():
        for b in range(0, len(items), a.bs):
            ch = items[b:b + a.bs]; seqs = [sets[n][i][0] + sets[n][i][1] for n, i in ch]; L = max(map(len, seqs))
            ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long); att = torch.zeros((len(seqs), L), dtype=torch.long)
            for j, s in enumerate(seqs):
                ids[j, :len(s)] = torch.tensor(s); att[j, :len(s)] = 1
            ids, att = ids.cuda(), att.cuda()
            logits = m(input_ids=ids, attention_mask=att).logits
            for j, (n, i) in enumerate(ch):
                c, y = sets[n][i]
                lg = logits[j, len(c) - 1:len(c) + len(y) - 1].float(); tg = ids[j, len(c):len(c) + len(y)]
                z = lg[None] / Tg[:, None, None]
                acc[n] += (z.gather(-1, tg[None, :, None].expand(len(TS), -1, 1))[..., 0] - torch.logsumexp(z, -1)).sum(1).double().cpu()
    return {n: [float(x) for x in v] for n, v in acc.items()}
R = {"cand": CAND, "sys": {}}
for name, P in CAND.items():
    R["sys"][name] = profile(P)
    json.dump(R, open(a.out, "w")); print("fait", name, flush=True)
# lecture
import statistics
def pmax(v): return max(v)
print("gain par token sous chaque prompt systeme candidat ; z parmi les faux endpoints de meme temperature (z fort = le prompt explique l'endpoint)")
fk = sorted(null)
Tf = lambda e: float(e.split("-")[1])
Tst = {e: (TS[max(range(len(TS)), key=lambda i: G["base"][e][i])] if not e.startswith("fk-") else Tf(e)) for e in list(eps) + fk}
print("          " + " ".join(f"{n[:11]:>12}" for n in CAND))
for e in eps:
    row = []
    for name in CAND:
        lam = {x: (pmax(R["sys"][name][x]) - pmax(G["base"][x])) / G["ntok"][x] for x in list(eps) + fk}
        b = [f for f in fk if abs(Tf(f) - Tst[e]) <= 0.1001]
        vs = [lam[f] for f in b]; med = statistics.median(vs); mad = statistics.median(abs(v - med) for v in vs) * 1.4826 or 1e-9
        row.append(f"{lam[e]:+.3f}({(lam[e] - med) / mad:+5.1f})")
    print(f"  {e}  " + " ".join(row))
print("FINI")
