"""Lecteur : relit chaque réponse de chaque guichet sous la base locale et des lecteurs modifiés, et sort des SOMMES par
réponse (agrégeables sur n'importe quel sous-ensemble de réponses : 25, 50, 100).

python feats.py --family smol360 --inp worlds/smol360_train_s1.jsonl --out feats/
Entrée : jsonl de guichets {gid, responses[], finish[]} (jumeaux) ou le même format construit depuis le journal réel.
Colonnes (par réponse) :
  n, top1, rank10, lowp                                   comptes de tokens
  lp_T, H_T, V_T pour T dans TEMPS                         somme log p_T(x), entropie, varentropie (base tempérée)
  b{k}_r_T, b{k}_V_T, b{k}_n  (T=0.8,1.0 ; k=0,1,2)        calibration par tranche d'entropie : somme (lp+H), somme V, compte
  {q}_d_T, {q}_Ep_T, {q}_Vp_T, {q}_Eq_T                    lecteur q : somme d = log q_T - log p_T, son espérance et sa
                                                           variance sous p_T, son espérance sous q_T
"""
import argparse, json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import probes_v4
from lora_train import FAMILIES
from gen import load_variant

TEMPS = [0.6, 0.7, 0.8, 0.9, 1.0, 1.2]
QT = [0.8, 1.0]
READERS = ["fp8", "int8", "nf4", "lora:alpaca_r16@1.0", "lora:stories_r32@1.0", "lora:gsm_r16@1.0"]
BINS = [0.3, 1.5]


def colnames():
    c = ["n", "top1", "rank10", "lowp"]
    for T in TEMPS:
        c += [f"lp_{T}", f"H_{T}", f"V_{T}"]
    for T in QT:
        for k in range(3):
            c += [f"b{k}_r_{T}", f"b{k}_V_{T}", f"b{k}_n_{T}"]
    for q in READERS:
        qn = q.split(":")[-1].split("@")[0]
        for T in QT:
            c += [f"{qn}_d_{T}", f"{qn}_Ep_{T}", f"{qn}_Vp_{T}", f"{qn}_Eq_{T}"]
    return c


def hidden_logits(model, ids, att, sel):
    h = model.model(input_ids=ids, attention_mask=att).last_hidden_state
    return model.lm_head(h[sel]).float()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", default="feats")
    ap.add_argument("--lora_dir", default="loras")
    ap.add_argument("--tok_budget", type=int, default=2000)
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(0.15)   # GPU partagée : ne jamais dépasser ~21 Go
    fam = a.family
    tok = AutoTokenizer.from_pretrained(FAMILIES[fam])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    end_id = tok.convert_tokens_to_ids("<|im_end|>")
    prompts = [s["prompt"] for s in probes_v4.plan()]
    guichets = [json.loads(l) for l in open(a.inp)]
    base = load_variant(fam, "base", a.lora_dir)
    readers = [load_variant(fam, q, a.lora_dir) for q in READERS]
    seqs = []  # (gidx, slot, ids, start)
    for gi, g in enumerate(guichets):
        for ri, resp in enumerate(g["responses"]):
            if resp is None:
                continue
            pidx = g.get("prompt_idx", list(range(len(g["responses"]))))[ri]
            pre = tok.apply_chat_template([{"role": "user", "content": prompts[pidx]}], tokenize=False, add_generation_prompt=True)
            a_ids = tok(pre, add_special_tokens=False)["input_ids"]
            r_ids = tok(resp, add_special_tokens=False)["input_ids"][:256]
            if g.get("finish", ["stop"] * 999)[ri] == "stop" and len(r_ids) < 256:
                r_ids = r_ids + [end_id]
            if not r_ids:
                continue
            seqs.append((gi, pidx, a_ids + r_ids, len(a_ids)))
    cols = colnames(); C = len(cols); ci = {c: i for i, c in enumerate(cols)}
    out = np.zeros((len(seqs), C), dtype=np.float64)
    order = sorted(range(len(seqs)), key=lambda k: len(seqs[k][2]))
    k = 0
    while k < len(order):
        L = len(seqs[order[min(k + 1, len(order) - 1)]][2])
        bs = max(1, int(a.tok_budget * 49152 / base.config.vocab_size) // L)   # mémoire ~ tokens x vocabulaire
        idx = order[k:k + bs]; k += bs
        Lm = max(len(seqs[i][2]) for i in idx)
        ids = torch.full((len(idx), Lm), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(idx), Lm), dtype=torch.long)
        sel = torch.zeros((len(idx), Lm), dtype=torch.bool)   # positions qui prédisent un token de réponse
        tgt = []; owner = []
        for r, i in enumerate(idx):
            s = seqs[i][2]; st = seqs[i][3]
            ids[r, :len(s)] = torch.tensor(s); att[r, :len(s)] = 1
            sel[r, st - 1:len(s) - 1] = True
            tgt += s[st:]; owner += [r] * (len(s) - st)
        ids, att, sel = ids.cuda(), att.cuda(), sel.cuda()
        tgt = torch.tensor(tgt, device="cuda"); owner = torch.tensor(owner, device="cuda")
        B = len(idx)
        acc = torch.zeros((B, C), dtype=torch.float64, device="cuda")
        def add(col, vals):
            acc[:, ci[col]].index_add_(0, owner, vals.double())
        zb = hidden_logits(base, ids, att, sel)
        lp1 = torch.log_softmax(zb, -1)
        tl = lp1.gather(1, tgt[:, None])[:, 0]
        rank = (lp1 > tl[:, None]).sum(1)
        add("n", torch.ones_like(tl)); add("top1", (rank == 0).float()); add("rank10", (rank >= 10).float())
        add("lowp", (tl < np.log(1e-3)).float())
        H1 = -(lp1.exp() * lp1).sum(1)
        bins = torch.bucketize(H1, torch.tensor(BINS, device="cuda"))
        lpT = {}
        for T in TEMPS:
            l = torch.log_softmax(zb / T, -1); p = l.exp()
            x = l.gather(1, tgt[:, None])[:, 0]; H = -(p * l).sum(1); V = (p * l * l).sum(1) - H * H
            add(f"lp_{T}", x); add(f"H_{T}", H); add(f"V_{T}", V)
            if T in QT:
                lpT[T] = (l, p)
                for b in range(3):
                    m = (bins == b).float()
                    add(f"b{b}_r_{T}", (x + H) * m); add(f"b{b}_V_{T}", V * m); add(f"b{b}_n_{T}", m)
            del l, p
        for q, rd in zip(READERS, readers):
            qn = q.split(":")[-1].split("@")[0]
            zq = hidden_logits(rd, ids, att, sel)
            for T in QT:
                l, p = lpT[T]
                lq = torch.log_softmax(zq / T, -1)
                d = lq - l
                add(f"{qn}_d_{T}", d.gather(1, tgt[:, None])[:, 0])
                Ep = (p * d).sum(1); add(f"{qn}_Ep_{T}", Ep); add(f"{qn}_Vp_{T}", (p * d * d).sum(1) - Ep * Ep)
                add(f"{qn}_Eq_{T}", (lq.exp() * d).sum(1))
                del lq, d
            del zq
        del zb, lp1, lpT
        out[idx] = acc.cpu().numpy()
    os.makedirs(a.out, exist_ok=True)
    name = os.path.basename(a.inp).replace(".jsonl", "")
    np.savez_compressed(os.path.join(a.out, f"{name}.npz"), X=out, gi=np.array([s[0] for s in seqs]), ri=np.array([s[1] for s in seqs]),
                        cols=np.array(cols))
    meta = [{k: g[k] for k in g if k not in ("responses", "finish")} for g in guichets]
    json.dump(meta, open(os.path.join(a.out, f"{name}.meta.json"), "w"))
    print("écrit", name, out.shape)


if __name__ == "__main__":
    main()
