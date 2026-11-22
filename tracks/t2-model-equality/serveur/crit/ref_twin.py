"""Relit les vraies réponses avec un modèle local : quel décodage fait le serveur, quelle température par endpoint.
Aucune requête vers l'API. Usage : python ref_twin.py journal.jsonl [modele]"""
import sys, json, math, collections, os
sys.path.append("/home/adem/eac/t2c/libs")
os.environ.setdefault("HF_HOME", "/home/adem/eac/t2c/hf")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
path = sys.argv[1]; MODEL = sys.argv[2] if len(sys.argv) > 2 else "HuggingFaceTB/SmolLM2-360M-Instruct"
TRUTH = {"ref-01":"clair","ref-02":"fp8","ref-03":"systeme","ref-04":"LoRA0.7","ref-05":"persona","ref-06":"filigrane","ref-07":"LoRA1.5","ref-08":"T1.0","ref-09":"LoRA0.3"}
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).cuda().eval()
rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
rows = [r for r in rows if r.get("type", "answer") == "answer" and r.get("response")]
TS = [round(0.5 + 0.05 * i, 2) for i in range(19)]            # 0,50 .. 1,40
agg = collections.defaultdict(lambda: {"ll": [0.0]*len(TS), "ent": [0.0]*len(TS), "n": 0, "ranks": [], "per_resp": []})
dump = {}
with torch.no_grad():
    for r in rows:
        ctx = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True)
        c_ids = tok(ctx, add_special_tokens=False)["input_ids"]
        y_ids = tok(r["response"], add_special_tokens=False)["input_ids"]
        if r.get("finish_reason") == "stop":
            y_ids = y_ids + [tok.convert_tokens_to_ids("<|im_end|>")]
        if len(y_ids) < 2: continue
        ids = torch.tensor([c_ids + y_ids]).cuda()
        logits = model(ids).logits[0, len(c_ids)-1:-1].double()         # une ligne par token de la réponse
        tgt = torch.tensor(y_ids).cuda()
        a = agg[r["endpoint"]]
        rank = (logits > logits.gather(1, tgt[:, None])).sum(1)            # 0 = token le plus probable
        a["ranks"] += rank.tolist(); a["n"] += len(y_ids)
        ll_resp = []
        for i, T in enumerate(TS):
            lp = torch.log_softmax(logits / T, -1)
            l = lp.gather(1, tgt[:, None])[:, 0]
            a["ll"][i] += float(l.sum()); a["ent"][i] += float(-(lp.exp() * lp).sum())
            ll_resp.append(float(l.sum()))
        a["per_resp"].append((r["probe_id"], len(y_ids), ll_resp))
print(f"modele {MODEL} ; {len(rows)} réponses")
print("| endpoint | vérité | tokens | T* | NLL/tok à T* | entropie/tok à T* | écart | NLL/tok à T=1 | rang>10 | rang>50 | rang max | part top-1 |")
out = {}
for e in sorted(agg):
    a = agg[e]; n = a["n"]
    i = max(range(len(TS)), key=lambda k: a["ll"][k]); j = TS.index(1.0)
    rk = a["ranks"]
    f = lambda k: sum(x >= k for x in rk) / len(rk)
    out[e] = {"T_star": TS[i], "nll": -a["ll"][i]/n, "ent": a["ent"][i]/n, "nll_T1": -a["ll"][j]/n, "n": n,
              "rank_ge10": f(10), "rank_ge50": f(50), "rank_max": max(rk), "top1": sum(x == 0 for x in rk)/len(rk),
              "ll_by_T": [x/n for x in a["ll"]], "per_resp": a["per_resp"]}
    o = out[e]
    print(f"| {e} | {TRUTH.get(e,'?')} | {n} | {o['T_star']:.2f} | {o['nll']:.3f} | {o['ent']:.3f} | {o['nll']-o['ent']:+.3f} | {o['nll_T1']:.3f} | {o['rank_ge10']:.3f} | {o['rank_ge50']:.4f} | {o['rank_max']} | {o['top1']:.3f} |")
json.dump({"model": MODEL, "TS": TS, "endpoints": out}, open(f"twin_{MODEL.split('/')[-1]}_{os.path.basename(path)}.json", "w"))
