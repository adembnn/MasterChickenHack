"""T2 : des reponses brutes au CSV, sans corrige et sans requete.
1. Famille de chaque endpoint = modele local candidat qui explique le mieux ses reponses (NLL par caractere, temperature ajustee).
2. Dans chaque famille : tout le monde = modele de base, sauf les endpoints designes comme variantes :
   a) reponses nettement plus longues que la famille (LoRA forte) ; b) z > seuil au lecteur LoRA distille sur les endpoints (a).
Usage : python ep_pipeline.py --data journal.jsonl --zoo m1,m2,... --out soumission [--prefix ep-]"""
import sys, os, json, argparse, collections, statistics, subprocess
os.environ.setdefault("HF_HOME", "/home/adem/eac/t2/hf")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--zoo", required=True)
ap.add_argument("--out", default="soumission")
ap.add_argument("--prefix", default="")
ap.add_argument("--skip", default="P000,P001")
ap.add_argument("--z_thr", type=float, default=2.0)
ap.add_argument("--len_ratio", type=float, default=1.8)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--epochs", type=int, default=6)
a = ap.parse_args()

rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
rows = [r for r in rows if r.get("type", "answer") == "answer" and r.get("response") and r["endpoint"].startswith(a.prefix)]
for r in rows:
    r["pid"] = r.get("probe_id", r.get("id"))
read = [r for r in rows if r["pid"] not in a.skip.split(",")]
eps = sorted({r["endpoint"] for r in rows})
TS = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]


def chat(tok, prompt):
    try:
        return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    except Exception:
        return prompt + "\n"


prof = {}
for m in a.zoo.split(","):
    try:
        tok = AutoTokenizer.from_pretrained(m)
        model = AutoModelForCausalLM.from_pretrained(m, dtype=torch.bfloat16, device_map={"": 0}).eval()
    except Exception as e:
        print("ECHEC chargement", m, type(e).__name__, str(e)[:100], flush=True)
        continue
    ll = {e: [0.0] * len(TS) for e in eps}
    nt, nchar, lens = collections.Counter(), collections.Counter(), collections.defaultdict(list)
    items = []
    for r in read:
        c = tok(chat(tok, r["prompt"]), add_special_tokens=False)["input_ids"]
        y = tok(r["response"], add_special_tokens=False)["input_ids"][:300]
        if len(y) >= 2:
            items.append((r["endpoint"], c, y, len(r["response"])))
    items.sort(key=lambda t: len(t[1]) + len(t[2]))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    with torch.no_grad():
        for b0 in range(0, len(items), a.bs):
            chunk = items[b0:b0 + a.bs]
            L = max(len(c) + len(y) for _, c, y, _ in chunk)
            ids = torch.full((len(chunk), L), pad, dtype=torch.long)
            att = torch.zeros((len(chunk), L), dtype=torch.long)
            for j, (_, c, y, _) in enumerate(chunk):
                ids[j, :len(c) + len(y)] = torch.tensor(c + y)
                att[j, :len(c) + len(y)] = 1
            logits = model(input_ids=ids.to(model.device), attention_mask=att.to(model.device)).logits
            for j, (e, c, y, nch) in enumerate(chunk):
                lg = logits[j, len(c) - 1:len(c) + len(y) - 1].float()
                tgt = torch.tensor(y, device=model.device)
                for i, T in enumerate(TS):
                    ll[e][i] += float(torch.log_softmax(lg / T, -1).gather(1, tgt[:, None]).sum())
                nt[e] += len(y)
                nchar[e] += nch
                lens[e].append(len(y))
    prof[m] = {}
    for e in eps:
        i = max(range(len(TS)), key=lambda k: ll[e][k])
        prof[m][e] = {"T": TS[i], "nll_tok": -ll[e][i] / max(1, nt[e]), "nll_char": -ll[e][i] / max(1, nchar[e]),
                      "len_med": statistics.median(lens[e]) if lens[e] else 0}
    print(f"lu par {m.split('/')[-1]:30s} NLL/caractere mediane {statistics.median(v['nll_char'] for v in prof[m].values()):.3f}", flush=True)
    del model
    torch.cuda.empty_cache()

models = list(prof)
fam, diag = {}, {}
for e in eps:
    order = sorted(models, key=lambda m: prof[m][e]["nll_char"])
    best = order[0]
    margin = (prof[order[1]][e]["nll_char"] - prof[best][e]["nll_char"]) if len(order) > 1 else 9.0
    fam[e] = best
    diag[e] = {"famille": best, "marge_nll_char": round(margin, 4), "second": order[1] if len(order) > 1 else "",
               "T": prof[best][e]["T"], "nll_tok": round(prof[best][e]["nll_tok"], 3), "len_med": prof[best][e]["len_med"]}
families = collections.defaultdict(list)
for e in eps:
    families[fam[e]].append(e)

label = {}
here = os.path.dirname(os.path.abspath(__file__))
for fi, (m, members) in enumerate(sorted(families.items(), key=lambda kv: kv[1][0]), 1):
    F = f"F{fi}"
    med = statistics.median(diag[e]["len_med"] for e in members)
    long_eps = [e for e in members if len(members) >= 3 and diag[e]["len_med"] >= a.len_ratio * med]
    variants = {e: "longueur" for e in long_eps}
    z = {}
    cand = None
    if not long_eps and len(members) >= 4:           # repli : l'endpoint le plus long, accepte seulement si son propre z >= 4
        c = max(members, key=lambda e: diag[e]["len_med"])
        if med > 0 and diag[c]["len_med"] >= 1.3 * med:
            cand = c
    train_eps = long_eps or ([cand] if cand else [])
    if train_eps and len(members) >= 4:
        outj = f"{a.out}_reader_{F}.json"
        cmd = [sys.executable, os.path.join(here, "reader_blind.py"), "--data", a.data, "--model", m,
               "--train", ",".join(f"{e}:1.0" for e in train_eps), "--eps", ",".join(members), "--scales", "1.0",
               "--skip", a.skip, "--epochs", str(a.epochs), "--out", outj]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if os.path.exists(outj):
            z = json.load(open(outj))["z"]["1.0"]
            if cand and z.get(cand, 0) < 4.0:
                print(f"{F} : lecteur de repli rejete (z propre de {cand} = {z.get(cand)})", flush=True)
            else:
                if cand:
                    variants[cand] = "plus long + lecteur"
                for e in members:
                    if e not in variants and z.get(e, 0) > a.z_thr:
                        variants[e] = "lecteur LoRA"
        else:
            print("lecteur en echec", r.stderr[-400:], flush=True)
    k = 0
    for e in members:
        zz = z.get(e)
        diag[e]["z_lora"] = zz
        diag[e]["F"] = F
        if e in variants:
            k += 1
            label[e] = (F, f"{F}T{k}")
            diag[e]["decision"] = f"variante ({variants[e]})"
        else:
            label[e] = (F, f"{F}S")
            diag[e]["decision"] = "base"
        diag[e]["confiance"] = ("forte" if e in long_eps else
                                "moyenne" if zz is not None and (zz > a.z_thr + 1 or zz < a.z_thr - 1.5) else "faible")

with open(a.out + ".csv", "w", newline="") as f:
    f.write("endpoint,family,model\n")
    for e in eps:
        f.write(f"{e},{label[e][0]},{label[e][1]}\n")
json.dump({"profil": prof, "diag": diag}, open(a.out + "_diag.json", "w"), indent=1)
print("\n| endpoint | famille (modele local) | marge | T | tokens medians | z LoRA | decision | confiance | etiquette |")
for e in eps:
    d = diag[e]
    print(f"| {e} | {d['famille'].split('/')[-1]} | {d['marge_nll_char']} | {d['T']} | {d['len_med']:.0f} | {d['z_lora']} | {d['decision']} | {d['confiance']} | {label[e][1]} |")
print(f"\n{len(eps)} endpoints, {len(families)} familles, {len(set(label.values()))} modeles -> {a.out}.csv")
