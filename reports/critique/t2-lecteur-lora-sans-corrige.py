"""Lecteur LoRA distillé SANS corrigé : on entraîne sur les endpoints donnés (choisis par une règle sans vérité, ex. réponses
au plafond de longueur), puis z de chaque endpoint contre la moyenne de TOUS les autres endpoints hors entraînement.
Réutilise lora_distill.py de Titouan (branche t2/casser-fp8). Aucune requête vers l'API."""
import sys, os, json, math, collections, argparse
sys.path.append("/home/adem/eac/t2c/libs"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", "/home/adem/eac/t2c/hf")
import torch
import lora_distill as LD
from transformers import AutoTokenizer, AutoModelForCausalLM
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
ap.add_argument("--train", required=True); ap.add_argument("--eps", default="", help="liste d'endpoints (vide = tous)")
ap.add_argument("--scales", default="0.2,0.4667,1.0"); ap.add_argument("--r", type=int, default=16)
ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--folds", type=int, default=2); ap.add_argument("--skip", default="P000,P001"); ap.add_argument("--out", default="reader_blind.json")
a = ap.parse_args()
torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "right"
if tok.pad_token is None: tok.pad_token = tok.eos_token
rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
rows = [r for r in rows if r.get("type", "answer") == "answer" and r.get("response") and r.get("probe_id", r.get("id")) not in a.skip.split(",")]
keep = set(a.eps.split(",")) if a.eps else None
by = collections.defaultdict(list)
for r in rows:
    if keep is None or r["endpoint"] in keep: by[r["endpoint"]].append((r["prompt"], r["response"]))
eps = sorted(by); spec = {e.split(":")[0]: float(e.split(":")[1]) for e in a.train.split(",")}
scales = [float(x) for x in a.scales.split(",")]
S = {k: {e: {} for e in eps} for k in scales}
for f in range(a.folds):
    base = (AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32) if os.environ.get("T2_CPU") else AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0})); base.eval()
    test = {e: [(p, r) for p, r in by[e] if LD.fold_of(p, a.folds) == f] for e in eps}
    lpb = {e: LD.seq_logprob(base, tok, test[e]) for e in eps}
    train = [(p, r, spec[e]) for e in spec for p, r in by.get(e, []) if LD.fold_of(p, a.folds) != f]
    pm, store = LD.train_lora(a, tok, base, train, f)
    for k in scales:
        LD.set_scale(pm, k, store)
        for e in eps:
            for (p, _), x, y in zip(test[e], LD.seq_logprob(pm, tok, test[e]), lpb[e]): S[k][e][p] = x - y
    del pm, base; torch.cuda.empty_cache()
res = {"train": a.train, "z": {}, "mean_s": {}}
for k in scales:
    common = sorted(set.intersection(*[set(S[k][e]) for e in eps])); z = {}
    for e in eps:
        pool = [q for q in eps if q != e and q not in spec]
        d = [S[k][e][p] - sum(S[k][q][p] for q in pool) / len(pool) for p in common]
        mu = sum(d) / len(d); sd = math.sqrt(sum((x - mu) ** 2 for x in d) / (len(d) - 1))
        z[e] = round(mu / (sd / math.sqrt(len(d))), 2)
    res["z"][str(k)] = z; res["mean_s"][str(k)] = {e: round(sum(S[k][e].values()) / len(S[k][e]), 3) for e in eps}
    print(f"échelle {k} ({len(common)} prompts) z contre les autres :", " ".join(f"{e[-2:]}={z[e]:+.1f}" for e in eps), flush=True)
json.dump(res, open(a.out, "w"), indent=1)
