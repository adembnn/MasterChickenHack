#!/usr/bin/env python
"""Lecteur « direction apprise sur un AUTRE endpoint », à T libre, loi nulle par faux endpoints.
Pour chaque endpoint source s : LoRA (r=8) + échelle de logits apprise, ajustée sur ses réponses (hors sonde 0).
Pour chaque cible g : L(g,s) = [max_T ll(base+dir_s, g) - max_T ll(base, g)] / n_tokens (T libre dans les deux termes).
Loi nulle : NF faux endpoints (base seule, T tiré dans {0.7,0.85}) + NT jumeaux à prompt caché (base, system/persona, T 0,7-0,9).
z = (L - moyenne nulle) / écart-type nul ; on rapporte aussi le max des jumeaux déguisés.
python dirtest.py --family llama1b --data data/ep_llama1b.jsonl --sources ep-05,ep-11 --out out.json
"""
import argparse, json, os, random, sys, time
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "twin"))
import probes_v4
import gen as G
from disguises import SYSTEM_TRAIN, SYSTEM_VAL, PERSONA_TRAIN, PERSONA_VAL
FAM = {"smol360": "HuggingFaceTB/SmolLM2-360M-Instruct", "llama1b": "unsloth/Llama-3.2-1B-Instruct",
       "qwen1.5": "Qwen/Qwen2.5-1.5B-Instruct", "qwen3": "Qwen/Qwen2.5-3B-Instruct"}
TG = np.arange(0.5, 1.51, 0.1)


def seqs_of(tok, plan, g, end_id, with_sys=None):
    out = []
    for ri, resp in enumerate(g["responses"]):
        if ri == 0 or resp is None: continue
        pid = g["prompt_idx"][ri]
        pre = tok.apply_chat_template([{"role": "user", "content": plan[pid]}], tokenize=False, add_generation_prompt=True)
        a = tok(pre, add_special_tokens=False)["input_ids"]
        r = tok(resp, add_special_tokens=False)["input_ids"][:256]
        if (g.get("finish") or ["stop"] * 999)[ri] == "stop" and len(r) < 256: r = r + [end_id]
        if r: out.append((a, r))
    return out


def pad(batch):
    L = max(len(a) + len(r) for a, r in batch)
    ids = torch.zeros((len(batch), L), dtype=torch.long); att = torch.zeros_like(ids); lab = torch.full_like(ids, -100)
    for i, (a, r) in enumerate(batch):
        s = a + r; ids[i, :len(s)] = torch.tensor(s); att[i, :len(s)] = 1; lab[i, len(a):len(s)] = torch.tensor(r)
    return ids.cuda(), att.cuda(), lab.cuda()


def train_dir(tok, base_path, seqs, a, seed):
    torch.manual_seed(seed)
    m = AutoModelForCausalLM.from_pretrained(base_path, dtype=torch.bfloat16).cuda()
    cfg = LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.0, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    pm = get_peft_model(m, cfg)
    ls = torch.nn.Parameter(torch.tensor(0.2, device="cuda"))   # log(1/tau)
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad] + [ls], lr=a.lr)
    rnd = random.Random(seed); pm.train()
    for ep in range(a.epochs):
        rnd.shuffle(seqs); tot = 0
        for i in range(0, len(seqs), 4):
            ids, att, lab = pad(seqs[i:i + 4])
            lg = pm(input_ids=ids, attention_mask=att).logits[:, :-1].float() * ls.exp()
            loss = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), lab[:, 1:].reshape(-1), ignore_index=-100)
            loss.backward(); opt.step(); opt.zero_grad(); tot += loss.item()
    pm.eval()
    return pm, float(ls.exp())


@torch.no_grad()
def ll_grid(model, seqs, adapter):
    """somme sur les tokens de log p_T(y), T sur TG ; renvoie (len(TG),), n_tokens"""
    tot = torch.zeros(len(TG), device="cuda", dtype=torch.float64); n = 0
    Tt = torch.tensor(TG, device="cuda", dtype=torch.float32)
    for i in range(0, len(seqs), 6):
        batch = seqs[i:i + 6]
        ids, att, lab = pad(batch)
        if adapter:
            lg = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
        else:
            with model.disable_adapter():
                lg = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
        tgt = lab[:, 1:]; m = tgt != -100
        z = lg[m]; t = tgt[m]
        for k in range(len(TG)):
            tot[k] += torch.log_softmax(z / Tt[k], -1).gather(1, t[:, None]).sum().double()
        n += int(m.sum()); del lg, z
    return tot.cpu().numpy(), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--sources", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--nfake", type=int, default=100); ap.add_argument("--ntwin", type=int, default=48)
    ap.add_argument("--r", type=int, default=8); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--mem", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(a.mem)
    path = FAM[a.family]
    tok = AutoTokenizer.from_pretrained(path, padding_side="left")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    end_id = tok.convert_tokens_to_ids("<|im_end|>") if "<|im_end|>" in tok.get_vocab() else tok.eos_token_id
    plan = [s["prompt"] for s in probes_v4.plan()]
    real = {g["gid"]: g for g in map(json.loads, open(a.data))}
    pidx = next(iter(real.values()))["prompt_idx"][1:]
    prompts = [plan[i] for i in pidx]
    # 1. faux endpoints et jumeaux déguisés (base seule)
    rnd = random.Random(a.seed + 7)
    fk, tw = [], []
    for i in range(a.nfake):
        fk.append({"gid": f"fake{i}", "disguise": {"system": None, "T": rnd.choice([0.7, 0.85]), "wm": None, "wm_key": rnd.randrange(1 << 30)}})
    SY = SYSTEM_TRAIN + SYSTEM_VAL + PERSONA_TRAIN + PERSONA_VAL
    for i in range(a.ntwin):
        tw.append({"gid": f"twin{i}", "disguise": {"system": SY[i % len(SY)], "T": rnd.choice([0.7, 0.75, 0.8, 0.85, 0.9]), "wm": None, "wm_key": 0}})
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
    torch.manual_seed(a.seed + 3)
    res = G.run_guichets(base, tok, fk + tw, prompts, max_new=256, tok_budget=int(6e9 / (2 * base.config.num_hidden_layers * base.config.num_key_value_heads * (base.config.hidden_size // base.config.num_attention_heads) * 2)))
    print(f"générés {len(fk)} faux + {len(tw)} jumeaux en {time.time() - t0:.0f}s", flush=True)
    del base; torch.cuda.empty_cache()
    targets = dict(real)
    for g in fk + tw:
        r = res[g["gid"]]
        targets[g["gid"]] = {"gid": g["gid"], "responses": [None] + [x[0] for x in r], "finish": ["stop"] + [x[1] for x in r],
                             "prompt_idx": [0] + pidx}
    TS = {gid: seqs_of(tok, plan, g, end_id) for gid, g in targets.items()}
    out = {"args": vars(a), "sources": {}}
    for s in a.sources.split(","):
        t1 = time.time()
        pm, scale = train_dir(tok, path, list(seqs_of(tok, plan, real[s], end_id)), a, a.seed)
        L = {}
        for gid, sq in TS.items():
            if gid == s: continue
            la, n = ll_grid(pm, sq, True); lb, _ = ll_grid(pm, sq, False)
            L[gid] = float((la.max() - lb.max()) / n)
        nl = np.array([v for k, v in L.items() if k.startswith("fake")])
        tl = np.array([v for k, v in L.items() if k.startswith("twin")])
        mu, sd = nl.mean(), nl.std() + 1e-9
        z = {k: float((v - mu) / sd) for k, v in L.items() if not k.startswith(("fake", "twin"))}
        zt = (tl - mu) / sd
        out["sources"][s] = {"scale": scale, "null_mu": float(mu), "null_sd": float(sd), "fake_max_z": float(((nl - mu) / sd).max()),
                             "twin_max_z": float(zt.max()), "twin_mean_z": float(zt.mean()), "twin_p95_z": float(np.quantile(zt, 0.95)),
                             "z": z, "L": {k: v for k, v in L.items() if not k.startswith(("fake", "twin"))}}
        print(f"source {s} (1/tau={scale:.2f}, {time.time() - t1:.0f}s) null mu={mu:.4f} sd={sd:.4f} fake_maxz={((nl - mu) / sd).max():.1f} "
              f"twin max/p95 z={zt.max():.1f}/{np.quantile(zt, 0.95):.1f}\n   " + " ".join(f"{k}:{v:+.1f}" for k, v in sorted(z.items())), flush=True)
        del pm; torch.cuda.empty_cache()
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
