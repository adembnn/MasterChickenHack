"""Lecteur LoRA par endpoint, corrige de la temperature (aucune requete).
Pour chaque endpoint e de la famille :
  1. T*_e = temperature qui explique le mieux ses reponses sous le modele local de base ;
  2. on entraine une petite LoRA locale sur la moitie de ses reponses, avec les logits divises par T*_e : la LoRA n'a plus
     a apprendre la temperature, seulement un ecart de poids (ou de consigne) ;
  3. gain G_e = moyenne par token, sur les reponses NON vues (plis croises par question), de
     log p_LoRA(y) - log p_base(y), les deux a T*_e. Copie de la base : G proche de 0. Poids differents : G > 0.
  4. gain croise G_{e->f} : la LoRA apprise sur e appliquee aux reponses de f (a T*_f).
Usage : python reader_temp.py --data journal.jsonl --model <modele local> --eps ep-a,ep-b,... --out fichier.json
"""
import sys, os, json, math, argparse, collections, hashlib
os.environ.setdefault("HF_HOME", "/home/adem/eac/t2/hf")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--eps", required=True)
ap.add_argument("--train_eps", default="", help="endpoints sur lesquels entrainer un lecteur (vide = tous)")
ap.add_argument("--skip", default="P000,P001")
ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--r", type=int, default=16)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--folds", type=int, default=2)
ap.add_argument("--out", required=True)
a = ap.parse_args()
torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(a.model)
tok.padding_side = "right"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
eps = a.eps.split(",")
train_eps = a.train_eps.split(",") if a.train_eps else eps
rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
rows = [r for r in rows if r.get("type", "answer") == "answer" and r.get("response") and r["endpoint"] in eps
        and r.get("probe_id", r.get("id")) not in a.skip.split(",")]
by = collections.defaultdict(list)
for r in rows:
    by[r["endpoint"]].append((r["prompt"], r["response"]))
fold = lambda p: int(hashlib.md5(p.encode()).hexdigest(), 16) % a.folds


def enc(items):
    seqs, pl = [], []
    for p, r in items:
        c = tok(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True),
                add_special_tokens=False)["input_ids"]
        y = tok(r, add_special_tokens=False)["input_ids"][:200]
        seqs.append(c + y)
        pl.append(len(c))
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for j, s in enumerate(seqs):
        ids[j, :len(s)] = torch.tensor(s)
        att[j, :len(s)] = 1
    return ids.cuda(), att.cuda(), pl, [len(s) for s in seqs]


def token_ll(model, items, T, bs=8):
    """log-vraisemblance par reponse (somme) et nombre de tokens, logits divises par T."""
    out = []
    with torch.no_grad():
        for i in range(0, len(items), bs):
            ids, att, pl, ln = enc(items[i:i + bs])
            logits = model(input_ids=ids, attention_mask=att).logits.float()
            for j in range(len(ln)):
                lg = logits[j, pl[j] - 1:ln[j] - 1] / T
                tg = ids[j, pl[j]:ln[j]]
                out.append((float(torch.log_softmax(lg, -1).gather(1, tg[:, None]).sum()), int(tg.numel())))
    return out


base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0}).eval()
TS = [round(0.5 + 0.05 * i, 2) for i in range(15)]
Tstar, base_ll = {}, {}
for e in eps:
    tot = {T: sum(x for x, _ in token_ll(base, by[e], T)) for T in TS}
    Tstar[e] = max(tot, key=tot.get)
print("T* :", " ".join(f"{e[-2:]}={Tstar[e]}" for e in eps), flush=True)

G = {e: {} for e in train_eps}
for e in train_eps:
    num = collections.Counter()
    den = collections.Counter()
    for f in range(a.folds):
        cfg = LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.0, task_type="CAUSAL_LM",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        pm = get_peft_model(base, cfg)
        opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=a.lr)
        train = [it for it in by[e] if fold(it[0]) != f]
        pm.train()
        g = torch.Generator().manual_seed(f)
        for _ in range(a.epochs):
            perm = torch.randperm(len(train), generator=g).tolist()
            for i in range(0, len(train), 4):
                ids, att, pl, ln = enc([train[k] for k in perm[i:i + 4]])
                logits = pm(input_ids=ids, attention_mask=att).logits.float() / Tstar[e]
                loss = 0.0
                n = 0
                for j in range(len(ln)):
                    lg = logits[j, pl[j] - 1:ln[j] - 1]
                    tg = ids[j, pl[j]:ln[j]]
                    loss = loss + torch.nn.functional.cross_entropy(lg, tg, reduction="sum")
                    n += tg.numel()
                (loss / n).backward()
                opt.step()
                opt.zero_grad()
        pm.eval()
        for t in eps:
            test = [it for it in by[t] if fold(it[0]) == f]
            if not test:
                continue
            lr_ = token_ll(pm, test, Tstar[t])
            base = pm.unload() if False else base
            with pm.disable_adapter():
                lb = token_ll(pm, test, Tstar[t])
            num[t] += sum(x for x, _ in lr_) - sum(x for x, _ in lb)
            den[t] += sum(n for _, n in lr_)
        base = pm.unload()
        del pm
        torch.cuda.empty_cache()
    G[e] = {t: num[t] / max(1, den[t]) for t in eps}
    print(f"[lecteur {e}] gain propre {G[e][e]:+.4f} | " + " ".join(f"{t[-2:]}={G[e][t]:+.3f}" for t in eps), flush=True)
json.dump({"Tstar": Tstar, "G": G}, open(a.out, "w"), indent=1)
print("FINI", flush=True)
