#!/usr/bin/env python
"""Approcher la LoRA inconnue des organisateurs par distillation, puis s'en servir comme lecteur.
Données : réponses DÉJÀ collectées (JSONL endpoint/prompt/response), aucune requête faite ici.
On entraîne une LoRA locale sur SmolLM2-360M pour imiter les endpoints « LoRA forte » (ref-07 = force 1,5, ref-04 = 0,7 :
même LoRA, on entraîne à l'échelle relative 1,0 pour ref-07 et 0,7/1,5 pour ref-04). ref-09 (0,3) est EXCLU de l'entraînement.
Lecteur : s = log p_{base+LoRA*k}(y) - log p_base(y), pour plusieurs échelles k.
Validation croisée par PROMPT (2 plis) : une réponse testée n'a jamais son prompt dans l'entraînement.
Test : z apparié par prompt de chaque endpoint contre la moyenne des copies 01/03/05/06/08 (en excluant l'endpoint testé).
"""
import argparse, collections, hashlib, json, math, random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

NULLS = ["ref-01", "ref-03", "ref-05", "ref-06", "ref-08"]


def fold_of(prompt, k):
    return int(hashlib.md5(prompt.encode()).hexdigest(), 16) % k


def encode(tok, prompt):
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)


def set_scale(pm, k, store):
    for mod in pm.modules():
        if hasattr(mod, "scaling") and isinstance(mod.scaling, dict):
            if id(mod) not in store:
                store[id(mod)] = dict(mod.scaling)
            for key in mod.scaling:
                mod.scaling[key] = store[id(mod)][key] * k


def batch_tensors(tok, items, device):
    rows, plens = [], []
    for p, r in items:
        pi = tok(encode(tok, p), add_special_tokens=False)["input_ids"]
        ri = tok(r, add_special_tokens=False)["input_ids"][:160] + [tok.eos_token_id]
        rows.append(pi + ri)
        plens.append(len(pi))
    L = max(len(x) for x in rows)
    ids = torch.full((len(rows), L), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((len(rows), L), dtype=torch.long)
    lab = torch.full((len(rows), L), -100, dtype=torch.long)
    for j, x in enumerate(rows):
        ids[j, :len(x)] = torch.tensor(x)
        att[j, :len(x)] = 1
        lab[j, plens[j]:len(x)] = torch.tensor(x[plens[j]:])
    return ids.to(device), att.to(device), lab.to(device), plens, [len(x) for x in rows]


@torch.no_grad()
def seq_logprob(model, tok, items, bs=16):
    out = []
    for i in range(0, len(items), bs):
        ids, att, lab, plens, lens = batch_tensors(tok, items[i:i + bs], model.device)
        logp = torch.log_softmax(model(input_ids=ids, attention_mask=att).logits.float(), dim=-1)
        for j in range(len(lens)):
            tgt = ids[j, plens[j]:lens[j] - 1]  # sans l'EOS ajouté (le serveur coupe peut-être à max_tokens)
            sl = logp[j, plens[j] - 1:lens[j] - 2]
            out.append(sl.gather(-1, tgt[:, None]).sum().item())
    return out


def train_lora(a, tok, base, train, seed, temp=1.0):
    cfg = LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.0, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    pm = get_peft_model(base, cfg)
    store = {}
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=a.lr)
    pm.train()
    rnd = random.Random(seed)
    for ep_i in range(a.epochs):
        rnd.shuffle(train)
        tot = 0.0
        for i in range(0, len(train), 4):
            chunk = train[i:i + 4]
            for scale in sorted({c[2] for c in chunk}):  # même échelle dans un mini-lot
                sub = [(p, r) for p, r, s in chunk if s == scale]
                set_scale(pm, scale, store)
                ids, att, lab, _, _ = batch_tensors(tok, sub, pm.device)
                if temp == 1.0:
                    loss = pm(input_ids=ids, attention_mask=att, labels=lab).loss
                else:  # modèle du corpus = (base+LoRA)/T : la LoRA n'absorbe pas la température d'échantillonnage
                    lg = pm(input_ids=ids, attention_mask=att).logits[:, :-1].float() / temp
                    loss = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), lab[:, 1:].reshape(-1),
                                                             ignore_index=-100)
                loss.backward()
                tot += loss.item()
            opt.step()
            opt.zero_grad()
        print(f"graine {seed} époque {ep_i} perte {tot / max(1, len(train) / 4):.3f} (n_train={len(train)})", flush=True)
    set_scale(pm, 1.0, store)
    pm.eval()
    return pm, store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    ap.add_argument("--train", default="ref-07:1.0,ref-04:0.4667")
    ap.add_argument("--scales", default="0.2,0.4667,1.0")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--folds", type=int, default=2)
    ap.add_argument("--kinds", default="", help="filtrer les sondes (ex. reading) ; vide = tout")
    ap.add_argument("--out", default="results_lora_distill.json")
    ap.add_argument("--save_dir", default="")
    a = ap.parse_args()
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rows = [json.loads(l) for l in open(a.data) if l.strip()]
    if a.kinds:
        rows = [r for r in rows if r.get("kind") in a.kinds.split(",")]
    by_ep = collections.defaultdict(list)
    for r in rows:
        by_ep[r["endpoint"]].append((r["prompt"], r["response"]))
    eps = sorted(by_ep)
    train_spec = {e.split(":")[0]: float(e.split(":")[1]) for e in a.train.split(",")}
    scales = [float(x) for x in a.scales.split(",")]
    S = {k: {ep: {} for ep in eps} for k in scales}  # échelle -> ep -> prompt -> s

    for f in range(a.folds):
        base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0})
        test_items = {ep: [(p, r) for p, r in by_ep[ep] if fold_of(p, a.folds) == f] for ep in eps}
        base.eval()
        lp_base = {ep: seq_logprob(base, tok, test_items[ep]) for ep in eps}
        train = [(p, r, train_spec[ep]) for ep in train_spec for p, r in by_ep.get(ep, []) if fold_of(p, a.folds) != f]
        pm, store = train_lora(a, tok, base, train, f)
        for k in scales:
            set_scale(pm, k, store)
            for ep in eps:
                lp = seq_logprob(pm, tok, test_items[ep])
                for (p, _), x, y in zip(test_items[ep], lp, lp_base[ep]):
                    S[k][ep][p] = x - y
        del pm, base
        torch.cuda.empty_cache()

    res = {"config": vars(a), "z_loo_vs_nulls": {}, "mean_s": {}}
    for k in scales:
        common = set.intersection(*[set(S[k][ep]) for ep in eps])
        res["mean_s"][str(k)] = {ep: round(sum(S[k][ep].values()) / len(S[k][ep]), 3) for ep in eps}
        z = {}
        for ep in eps:
            pool = [q for q in NULLS if q != ep and q in S[k]]
            d = [S[k][ep][p] - sum(S[k][q][p] for q in pool) / len(pool) for p in common]
            mu = sum(d) / len(d)
            sd = math.sqrt(sum((x - mu) ** 2 for x in d) / (len(d) - 1))
            z[ep] = round(mu / (sd / math.sqrt(len(d))), 2)
        res["z_loo_vs_nulls"][str(k)] = z
        print(f"échelle {k} (n prompts communs={len(common)}) z:", z, flush=True)
    res["S"] = {str(k): S[k] for k in scales}
    json.dump(res, open(a.out, "w"), indent=1)
    if a.save_dir:  # adaptateur final sur TOUTES les réponses d'entraînement ; échelle relative 1,0 = force ref-07
        base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0})
        train = [(p, r, train_spec[ep]) for ep in train_spec for p, r in by_ep.get(ep, [])]
        pm, _ = train_lora(a, tok, base, train, 99)
        pm.save_pretrained(a.save_dir)
        print("adaptateur ->", a.save_dir, flush=True)


if __name__ == "__main__":
    main()
