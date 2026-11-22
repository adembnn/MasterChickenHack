"""Lecteur « decision d'arret » : a chaque position d'une reponse, le modele de base donne une probabilite d'arreter (token de fin).
On compare le nombre d'arrets observes a la somme de ces probabilites (attendu sous « copie de la base a la temperature T* »).
z_abs = (arrets observes - attendus) / ecart-type ; z_rel = z_abs rapporte aux faux endpoints de temperature voisine. Aucune requete.
Usage : eos_reader.py --data data50_F2.jsonl --null null50_F2.jsonl --model ... --tstar q50_F2.json --out eos_F2.json [--kfk 100]"""
import argparse, json, collections, statistics, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
ap = argparse.ArgumentParser()
for k in ("--data", "--null", "--model", "--tstar", "--out"): ap.add_argument(k, required=True)
ap.add_argument("--kfk", type=int, default=100); ap.add_argument("--bs", type=int, default=12); ap.add_argument("--skip", default="P000,P001")
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "right"
if tok.pad_token is None: tok.pad_token = tok.eos_token
skip = a.skip.split(",")
def load(path, keep):
    by = collections.defaultdict(list)
    for l in open(path, encoding="utf-8"):
        if not l.strip(): continue
        r = json.loads(l)
        if r.get("type", "answer") != "answer" or not r.get("response") or r.get("probe_id") in skip or not keep(r["endpoint"]): continue
        by[r["endpoint"]].append((r["prompt"], r["response"]))
    return by
Q = json.load(open(a.tstar)); TS = Q["TS"]
real = load(a.data, lambda e: e in Q["eps"])
null = load(a.null, lambda e: int(e.split("-")[-1]) < a.kfk // 4)
Tst = {e: TS[max(range(len(TS)), key=lambda i: Q["base"][e][i])] for e in real if e in Q["base"]}
for e in null: Tst[e] = float(e.split("-")[1])
CT = {}
def ctx(p):
    if p not in CT:
        CT[p] = tok(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, date_string="21 Nov 2026"), add_special_tokens=False)["input_ids"]
    return CT[p]
ENC = {e: [(ctx(p), tok(r, add_special_tokens=False)["input_ids"]) for p, r in items] for src in (real, null) for e, items in src.items()}
capR = max(len(y) for e in real for _, y in ENC[e]); capN = max(len(y) for e in null for _, y in ENC[e])
print(f"plafond de tokens observe : guichets {capR}, faux {capN}", flush=True)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0}).eval()
stop = set(x for x in [tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|eot_id|>"), tok.convert_tokens_to_ids("<|endoftext|>"), tok.convert_tokens_to_ids("<|end_of_text|>")] if isinstance(x, int) and x >= 0 and x != tok.unk_token_id)
g = getattr(model.generation_config, "eos_token_id", None)
stop |= set(g if isinstance(g, list) else [g] if g is not None else [])
stop = sorted(stop); print("tokens de fin :", [tok.convert_ids_to_tokens(s) for s in stop], flush=True)
R = {}
@torch.no_grad()
def run(e, cap):
    T = Tst[e]; obs = exp = var = 0.0; nres = ncens = 0; ll = 0.0; per = []
    items = sorted(ENC[e], key=lambda it: len(it[0]) + len(it[1]))
    for i in range(0, len(items), a.bs):
        b = items[i:i + a.bs]; L = max(len(c) + len(y) for c, y in b)
        ids = torch.full((len(b), L), tok.pad_token_id, dtype=torch.long); att = torch.zeros((len(b), L), dtype=torch.long)
        for j, (c, y) in enumerate(b):
            s = c + y; ids[j, :len(s)] = torch.tensor(s); att[j, :len(s)] = 1
        lg = model(input_ids=ids.cuda(), attention_mask=att.cuda()).logits.float()
        for j, (c, y) in enumerate(b):
            cens = len(y) >= cap - 1
            z = lg[j, len(c) - 1:len(c) + len(y)] / T              # positions qui predisent y[0..n-1] puis la suite (arret ?)
            pe = torch.softmax(z, -1)[:, stop].sum(-1).double().clamp(1e-12, 1 - 1e-12)
            if cens: pe = pe[:-1]                                   # reponse coupee par le plafond : pas de decision d'arret observee
            o = 0.0 if cens else 1.0
            obs += o; exp += pe.sum().item(); var += (pe * (1 - pe)).sum().item(); nres += 1; ncens += cens
            ll += (torch.log1p(-pe[:-1] if not cens else -pe).sum() + (torch.log(pe[-1]) if not cens else 0)).item()
            per.append((len(y), o, pe.sum().item()))
    R[e] = {"T": T, "obs": obs, "exp": exp, "var": var, "n": nres, "cens": ncens, "ll": ll, "ntok": sum(p[0] for p in per)}
for e in sorted(real):
    if e in Tst: run(e, 256)
print("guichets faits", flush=True)
for k, e in enumerate(sorted(null)):
    run(e, 200)
    if k % 20 == 0: print("faux", k, flush=True)
json.dump(R, open(a.out, "w"))
fk = [e for e in R if e.startswith("fk-")]
zabs = {e: (R[e]["obs"] - R[e]["exp"]) / max(R[e]["var"], 1e-9) ** 0.5 for e in R}
print("faux endpoints : z_abs median %+.2f, min %+.2f, max %+.2f" % (statistics.median(zabs[f] for f in fk), min(zabs[f] for f in fk), max(zabs[f] for f in fk)))
for e in sorted(real):
    if e not in R: continue
    b = [f for f in fk if abs(Tst[f] - Tst[e]) <= 0.1001] or fk; vs = [zabs[f] for f in b]
    med = statistics.median(vs); mad = statistics.median(abs(v - med) for v in vs) * 1.4826 or 1e-9
    print(f"{e} T*={Tst[e]:.2f} reponses={R[e]['n']} coupees={R[e]['cens']} arrets observes={R[e]['obs']:.0f} attendus={R[e]['exp']:.1f} z_abs={zabs[e]:+.2f} z_rel={(zabs[e] - med) / mad:+.2f} tokens/reponse={R[e]['ntok'] / R[e]['n']:.0f}")
print("FINI")
