"""Scan KGW rapide : une permutation par (cle, token precedent) partagee entre les gamma ; ref d'abord (ref-06 = filigrane
connu) pour identifier le schema. Generateurs CPU et CUDA (le serveur tourne probablement sur GPU)."""
import json, collections, math, sys
import torch
from transformers import AutoTokenizer, AutoConfig
FAM = {"qwen3b": "Qwen/Qwen2.5-3B-Instruct", "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
       "llama1b": "unsloth/Llama-3.2-1B-Instruct", "smol360": "HuggingFaceTB/SmolLM2-360M-Instruct"}
fam_of = {f"ref-0{i}": "smol360" for i in range(1, 10)}
for line in open("/tmp/eac-t2-real/ep_A_fp8_plus_lora_02_17_18.csv").read().splitlines()[1:]:
    ep, fam, _ = line.split(","); fam_of[ep] = fam
resp = collections.defaultdict(list)
for fn in ["/tmp/eac-t2-real/collecte_v4_ref_bloc0_20261121T191611Z.jsonl", "/tmp/eac-t2-real/collecte_v4_ep_bloc0_20261121T193826Z.jsonl"]:
    for l in open(fn):
        r = json.loads(l)
        if r.get("type") == "answer" and r.get("response") and r["slot"] >= 1:
            resp[r["endpoint"]].append(r["response"])
families = sys.argv[1].split(",")
keys = [int(k) for k in sys.argv[2].split(",")]
devs = sys.argv[3].split(",")
GAMMAS = [0.25, 0.5]
for fam in families:
    tok = AutoTokenizer.from_pretrained(FAM[fam]); V = AutoConfig.from_pretrained(FAM[fam]).vocab_size
    eps = sorted(e for e in resp if fam_of.get(e) == fam)
    pairs = {e: [(a, b) for t in resp[e] for ids in [tok(t, add_special_tokens=False)["input_ids"]] for a, b in zip(ids, ids[1:]) if b < V] for e in eps}
    by_prev = collections.defaultdict(list)
    for e in eps:
        for a, b in pairs[e]:
            by_prev[a].append((e, b))
    for dev in devs:
        for key in keys:
            gen = torch.Generator(device=dev)
            g = {(e, gm): 0 for e in eps for gm in GAMMAS}
            for a, lst in by_prev.items():
                gen.manual_seed(key * a)
                perm = torch.randperm(V, device=dev, generator=gen).cpu()
                rank = torch.empty(V, dtype=torch.int32); rank[perm] = torch.arange(V, dtype=torch.int32)
                for e, b in lst:
                    r = int(rank[b])
                    for gm in GAMMAS:
                        g[(e, gm)] += r < int(gm * V)
            for gm in GAMMAS:
                zs = {e: (g[(e, gm)] - gm * len(pairs[e])) / math.sqrt(len(pairs[e]) * gm * (1 - gm)) for e in eps}
                top = sorted(zs.items(), key=lambda kv: -kv[1])[:4]
                med = sorted(zs.values())[len(zs) // 2]
                print(f"{fam:8s} {dev:4s} cle={key:<9d} gamma={gm}: mediane {med:+.1f} | max " + ", ".join(f"{e} {z:+.1f}" for e, z in top), flush=True)
