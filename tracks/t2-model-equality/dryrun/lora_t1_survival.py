#!/usr/bin/env python
"""La direction LoRA distillee sur un guichet survit-elle a un decodage T=1,0 ? (question d'Adem sur ep-17 / ep-33.)
Jumeau local, 0 requete ep/ref. Meme chaine que ep_readers.py (casser-fp8) : LoRA distillee sur UN guichet LoRA a
T~0,75, validation croisee par prompt (2 plis), lecteur s_k = log p_{base+LoRA*k}(y) - log p_base(y) pour k in {0,3 ; 0,6 ; 1,0},
z apparie par prompt contre la mediane des autres guichets (retrait iteratif des |z|>3, guichets T>=0,9 exclus de la
reference comme dans l'ep reel). 50 reponses par guichet (sondes 4 a 53).

Groupes rapportes : MEME LoRA a T~0,75 / MEME LoRA a T=1,0 / AUTRE LoRA a T=1,0 / bases a T=1,0 (nulles chaudes) / bases T~0,75.
Usage : python lora_t1_survival.py --family qwen1.5 --world worlds/qwen1.5_train_s1.jsonl --train lora:stories_r32@1.5
"""
import argparse, collections, json, statistics, sys, os
from types import SimpleNamespace
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, os.path.expanduser("~/casser-fp8"))
sys.path.insert(0, os.path.expanduser("~/eac-t2/detecteur"))
import probes_v4
from lora_distill import train_lora, set_scale, fold_of, seq_logprob
from ep_readers import robust_z

MID = {"qwen1.5": "Qwen/Qwen2.5-1.5B-Instruct", "llama1b": "unsloth/Llama-3.2-1B-Instruct"}
SCALES = [0.3, 0.6, 1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--world", required=True)
    ap.add_argument("--train", required=True, help="variante du guichet d'entrainement, ex. lora:stories_r32@1.5")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    prompts = [s["prompt"] for s in probes_v4.plan()]
    gs = [json.loads(l) for l in open(a.world)]
    slots = list(range(4, 4 + a.n))
    items, meta = {}, {}
    for g in gs:
        items[g["gid"]] = [(prompts[s], g["responses"][s]) for s in slots if g["responses"][s]]
        meta[g["gid"]] = (g["variant"], g["disguise"]["kind"], g["disguise"]["T"])
    tr = next(e for e in meta if meta[e][0] == a.train and meta[e][1] == "plain" and meta[e][2] < 0.95)
    lname = a.train.split("@")[0]
    print(f"entrainement sur {tr} {meta[tr]} ; {len(gs)} guichets, {a.n} reponses", flush=True)
    mid = MID[a.family]
    tok = AutoTokenizer.from_pretrained(mid, padding_side="right")
    tok.pad_token = tok.pad_token or tok.eos_token
    eps = [e for e in meta if e != tr]
    base = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).cuda().eval()
    lp_base = {e: seq_logprob(base, tok, items[e], bs=8) for e in items}
    del base; torch.cuda.empty_cache()
    args = SimpleNamespace(r=16, lr=2e-4, epochs=6)
    Sk = {k: {e: {} for e in eps} for k in SCALES}
    for f in range(2):
        base = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).cuda().eval()
        train = [(p, r, 1.0) for p, r in items[tr] if fold_of(p, 2) != f]
        with torch.enable_grad():
            pm, store = train_lora(args, tok, base, train, f)
        with torch.no_grad():
            for k in SCALES:
                set_scale(pm, k, store)
                for e in eps:
                    idx = [i for i, (p, _) in enumerate(items[e]) if fold_of(p, 2) == f]
                    sub = [items[e][i] for i in idx]
                    for (p, _), x, i in zip(sub, seq_logprob(pm, tok, sub, bs=8), idx):
                        Sk[k][e][p] = x - lp_base[e][i]
        del pm, base; torch.cuda.empty_cache()
    excl = {e for e in eps if meta[e][2] >= 0.9}
    z = {}
    for k in SCALES:
        zk, _ = robust_z(Sk[k], eps, excl, 3.0)
        z[k] = zk
    zmax = {e: max((z[k].get(e) or -99) for k in SCALES) for e in eps}

    def group(e):
        v, d, T = meta[e]
        hot = T >= 0.9
        if v.startswith(lname + "@") or v.split("@")[0] == lname:
            return "MEME LoRA, T=1,0" if hot else "MEME LoRA, T~0,75"
        if v.startswith("lora:"):
            return "AUTRE LoRA, T=1,0" if hot else "AUTRE LoRA, T~0,75"
        if v == "base":
            return "base T=1,0 (nulle chaude)" if hot else "base T~0,75 (nulle)"
        return "autre variante (fp8/int8/nf4)"
    G = collections.defaultdict(list)
    for e in eps:
        G[group(e)].append(e)
    print(f"\nz max sur les 3 echelles, famille {a.family}, train={a.train}")
    for g, es in sorted(G.items()):
        vals = sorted((zmax[e] for e in es), reverse=True)
        print(f"  {g:32s} n={len(es):2d}  z moyen {statistics.mean(vals):+5.2f}  min {min(vals):+5.2f}  max {max(vals):+5.2f}")
    null_hot = [zmax[e] for e in G.get("base T=1,0 (nulle chaude)", [])]
    null_all = [zmax[e] for g in ("base T=1,0 (nulle chaude)", "base T~0,75 (nulle)") for e in G.get(g, [])]
    print(f"  nulles (toutes bases) : max {max(null_all):+.2f}, nulles chaudes : max {max(null_hot):+.2f}")
    print("\ndetail LoRA :")
    for e in sorted(eps, key=lambda e: -zmax[e]):
        if meta[e][0].startswith("lora:"):
            print(f"  {meta[e][0]:26s} {meta[e][1]:10s} T={meta[e][2]:<5} z(0,3/0,6/1,0) = " + " / ".join(f"{(z[k].get(e) or 0):+.2f}" for k in SCALES))
    if a.out:
        json.dump({"train": tr, "meta": meta, "z": {str(k): v for k, v in z.items()}}, open(a.out, "w"))


if __name__ == "__main__":
    main()
