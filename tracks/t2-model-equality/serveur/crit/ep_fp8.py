"""Etape optionnelle : lecteur fp8 par tenseur (code de Titouan, branche t2/casser-fp8), par famille, SANS corrige.
Lit <out>_diag.json ecrit par ep_pipeline.py, relit les reponses de chaque famille avec son modele local et sa version fp8,
z de chaque endpoint contre la moyenne des autres. Ecrit <out>_fp8.csv : comme <out>.csv, plus les endpoints a z > seuil isoles.
Usage : python ep_fp8.py --data journal.jsonl --out soumission [--z_thr 3.0]"""
import sys, os, json, math, argparse, collections, csv
os.environ.setdefault("HF_HOME", "/home/adem/eac/t2/hf")
import torch
import fp8_llr as F
from fp8_bank import make

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", default="soumission")
ap.add_argument("--z_thr", type=float, default=3.0)
ap.add_argument("--schemes", default="e4m3_tensor,w8a8_tensor")
ap.add_argument("--skip", default="P000,P001")
ap.add_argument("--bs", type=int, default=16)
a = ap.parse_args()
diag = json.load(open(a.out + "_diag.json"))["diag"]
rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
rows = [r for r in rows if r.get("type", "answer") == "answer" and r.get("response") and r["endpoint"] in diag
        and r.get("probe_id", r.get("id")) not in a.skip.split(",")]
fams = collections.defaultdict(list)
for e, d in diag.items():
    fams[d["famille"]].append(e)
labels = {r["endpoint"]: (r["family"], r["model"]) for r in csv.DictReader(open(a.out + ".csv"))}
zall = {}
for m, members in fams.items():
    if len(members) < 4:
        continue
    F.MODEL = m
    tok = F.AutoTokenizer.from_pretrained(m)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    by = {e: [r for r in rows if r["endpoint"] == e] for e in members}
    T = sorted(diag[e]["T"] for e in members)[len(members) // 2]

    def read(model):
        orig = model.forward

        def fwd(*x, **k):
            o = orig(*x, **k)
            o.logits = o.logits / T
            return o
        model.forward = fwd
        out = {}
        for e in members:
            P = [r["prompt"] for r in by[e]]
            G = [tok(r["response"], add_special_tokens=False)["input_ids"][:300] or [tok.eos_token_id] for r in by[e]]
            out[e] = [sum(t) for t in F.token_logprobs(model, tok, P, G, a.bs)]
        model.forward = orig
        return out

    base_m = F.load(torch.bfloat16)
    base = read(base_m)
    del base_m
    torch.cuda.empty_cache()
    for sc in a.schemes.split(","):
        mq = make(sc)
        lp = read(mq)
        del mq
        torch.cuda.empty_cache()
        per = {}
        for e in members:
            d = collections.defaultdict(list)
            for r, x, y in zip(by[e], lp[e], base[e]):
                d[r["prompt"]].append(x - y)
            per[e] = {p: sum(v) / len(v) for p, v in d.items()}
        common = sorted(set.intersection(*[set(per[e]) for e in members]))
        for e in members:
            pool = [q for q in members if q != e]
            dd = [per[e][p] - sum(per[q][p] for q in pool) / len(pool) for p in common]
            mu = sum(dd) / len(dd)
            sd = math.sqrt(sum((x - mu) ** 2 for x in dd) / (len(dd) - 1))
            zall.setdefault(e, {})[sc] = round(mu / (sd / math.sqrt(len(dd))), 2) if sd > 0 else 0.0
        print(f"{m.split('/')[-1]:24s} {sc:12s} T={T} : " + " ".join(f"{e[-2:]}={zall[e][sc]:+.1f}" for e in members), flush=True)

first = a.schemes.split(",")[0]
k = collections.Counter()
with open(a.out + "_fp8.csv", "w", newline="") as f:
    f.write("endpoint,family,model\n")
    for e in sorted(labels):
        fam, mod = labels[e]
        z = zall.get(e, {}).get(first)
        if z is not None and z > a.z_thr and mod.endswith("S"):
            k[fam] += 1
            mod = f"{fam}Q{k[fam]}"
        f.write(f"{e},{fam},{mod}\n")
json.dump(zall, open(a.out + "_fp8_z.json", "w"), indent=1)
print("->", a.out + "_fp8.csv")
