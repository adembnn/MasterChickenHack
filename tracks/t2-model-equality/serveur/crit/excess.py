"""Test d'ajustement sans apprentissage : exces de NLL par rapport a l'entropie, a la temperature T* du guichet.
Si les reponses d'un guichet sont tirees du modele local a la temperature T*, alors pour chaque token
E[-log p_T*(y)] = H_T* (entropie de la loi locale). Exces = somme(nll - H) / racine(somme Var) -> z ~ N(0,1) pour une copie exacte.
Poids differents (LoRA, fp8) ou consigne cachee : z > 0. Aucune requete. Decoupe aussi debut (20 premiers tokens) / suite."""
import sys, os, json, argparse, collections, math
os.environ.setdefault("HF_HOME", "/home/adem/eac/t2/hf")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--model", required=True); ap.add_argument("--eps", required=True)
ap.add_argument("--skip", default="P000,P001"); ap.add_argument("--out", required=True)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "right"
if tok.pad_token is None: tok.pad_token = tok.eos_token
eps = a.eps.split(",")
rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
rows = [r for r in rows if r.get("type", "answer") == "answer" and r.get("response") and r["endpoint"] in eps
        and r.get("probe_id", r.get("id")) not in a.skip.split(",")]
by = collections.defaultdict(list)
for r in rows: by[r["endpoint"]].append((r["prompt"], r["response"]))
m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0}).eval()
TS = [round(0.5 + 0.05 * i, 2) for i in range(17)]
res = {}
for e in eps:
    per = {T: [] for T in TS}   # liste de (nll, H, Var, position) par token
    with torch.no_grad():
        for p, r in by[e]:
            c = tok(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"]
            y = tok(r, add_special_tokens=False)["input_ids"][:200]
            if len(y) < 2: continue
            ids = torch.tensor([c + y]).cuda()
            lg = m(input_ids=ids).logits[0, len(c) - 1:len(c) + len(y) - 1].float()
            tg = ids[0, len(c):]
            for T in TS:
                lp = torch.log_softmax(lg / T, -1); pr = lp.exp()
                H = -(pr * lp).sum(-1); V = (pr * lp * lp).sum(-1) - H * H
                nll = -lp.gather(1, tg[:, None])[:, 0]
                per[T].append(torch.stack([nll, H, V, torch.arange(len(y), device=nll.device).float()], 1).cpu())
    tot = {T: float(torch.cat(per[T])[:, 0].sum()) for T in TS}
    Ts = min(tot, key=tot.get); X = torch.cat(per[Ts])
    def z(sel):
        d = (X[sel, 0] - X[sel, 1]); return float(d.sum() / X[sel, 2].sum().sqrt()), float(d.mean()), int(sel.sum())
    za, ma, n = z(torch.ones(len(X), dtype=torch.bool)); zd, md, _ = z(X[:, 3] < 20); zs, ms, _ = z(X[:, 3] >= 20)
    res[e] = dict(T=Ts, z=za, exces=ma, n=n, z_debut=zd, z_suite=zs, exces_debut=md, exces_suite=ms)
    print(f"{e} T*={Ts:.2f} n={n:5d} exces/token={ma:+.4f} z={za:+6.1f} | debut(20) {md:+.4f} z={zd:+5.1f} | suite {ms:+.4f} z={zs:+5.1f}", flush=True)
json.dump(res, open(a.out, "w"), indent=1); print("FINI")
