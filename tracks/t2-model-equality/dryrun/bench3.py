#!/usr/bin/env python
"""Banc aveugle FIDELE aux 33 vrais guichets : 3 familles de 11 (Qwen2.5-3B-Instruct, Qwen2.5-1.5B-Instruct,
Llama-3.2-1B-Instruct), echantillonnage a T=0,8, structure du corrige practice_answers.csv. AUCUNE requete
vers l'API du hackathon.

Par famille (11 guichets) : 7 copies de la base (clair, system prompt, persona, filigrane KGW, T=1,0 et 2
deguisements combines tires au hasard) + fp8 PER-TENSOR + LoRA 0,3 / 0,7 / 1,5. La LoRA cachee est
entrainee a repondre longuement (indice d'Adem sur les vrais ref : la LoRA forte ecrit plus long).
--variant_disguise_p 1.0 : les variantes de poids sont elles aussi deguisees.

Detecteurs, a budget b (b premieres reponses du flux : 1 extraction + bc + 5 seconds tirages, emboite) :
  readers     : ~/casser-fp8/ep_readers.py (lecteurs fp8 per-tensor e4m3 + W8A8, LoRA distillee sur le
                guichet a NLL/token max, validation croisee par prompt, z robuste contre la mediane de la
                famille), traduit en etiquettes : fp8 si z_fp8 > 3 ; sinon LoRA si un z LoRA > 3 (force =
                echelle au z max) ; le guichet d'entrainement = LoRA forte ; sinon base ;
  readers_len : idem, mais guichet d'entrainement = celui qui ecrit le plus long (indice d'Adem) ;
  v4          : method_v4 adaptee (blind_v4.decide_v4), methode de repli.
Famille : base locale a plus faible NLL/token (les 3 bases sont connues).

Usage : python bench3.py --seed0 30000 --trials 2 --out_dir runs/b3_a [--variant_disguise_p 1.0]
"""
import argparse
import itertools
import json
import os
import random
import sys
import time
from types import SimpleNamespace

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser("~/casser-fp8"))
import dryrun_blind_test as H  # noqa: E402
import blind_v4 as V  # noqa: E402
import fp8_llr as F  # noqa: E402
from fp8_bank import make as make_fp8_reader  # noqa: E402
from lora_distill import train_lora as distill_lora, set_scale, fold_of, seq_logprob  # noqa: E402
from ep_readers import robust_z  # noqa: E402
from method_v4 import EXTRACT_PROBE  # noqa: E402

H.BASE_POOL = {
    "qwen3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "llama1b": "unsloth/Llama-3.2-1B-Instruct",
}
BUDGETS = [25, 50, 100]
RESP_MAX_NEW = 160          # ep_readers lit au plus 160 tokens par reponse
Z_OUT = 3.0
SCALES = [0.3, 0.6, 1.0]


# ------------------------------------------------------------------------------------------ plan cache
def gen_truth(seed, variant_disguise_p):
    rng = random.Random(seed)
    eps = H.EPS[:]
    rng.shuffle(eps)
    plan, meta = {}, {}
    for f_i, key in enumerate(sorted(H.BASE_POOL)):
        fam = f"FAM{f_i}"
        meta[fam] = {"base_key": key, "model_id": H.BASE_POOL[key], "lora_seed": rng.randrange(10 ** 9)}
        combos = [["system_prompt"], ["persona"], ["watermark"], ["temperature"], ["plain"]]
        for _ in range(2):
            combos.append(rng.sample(["system_prompt", "persona", "watermark", "temperature"], 2))
        slots = [("base", c) for c in combos] + [(v, None) for v in ["fp8pt_e4m3", "lora_0.3", "lora_0.7", "lora_1.5"]]
        for ep, (variant, kinds) in zip(eps[f_i * 11:(f_i + 1) * 11], rng.sample(slots, 11)):
            if kinds is None:
                kinds = [rng.choice(["system_prompt", "persona", "watermark", "temperature"])] if rng.random() < variant_disguise_p else ["plain"]
            if "persona" in kinds and "system_prompt" in kinds:
                kinds = [k for k in kinds if k != "system_prompt"] or ["persona"]
            cfg = {"family": fam, "base_key": key, "variant": variant, "model": f"{fam}_{variant}", "disguise": kinds,
                   "system": None, "wm_delta": 0.0, "wm_key": 0, "temperature": 0.8}
            if "system_prompt" in kinds:
                cfg["system"] = rng.choice(H.SYSTEM_PROMPTS)
            if "persona" in kinds:
                cfg["system"] = rng.choice(H.PERSONAS)
            if "watermark" in kinds:
                cfg["wm_delta"], cfg["wm_key"] = rng.choice([2.0, 4.0]), rng.randrange(1, 10 ** 6)
            if "temperature" in kinds:
                cfg["temperature"] = 1.0
            plan[ep] = cfg
    return {"seed": seed, "families": sorted(H.BASE_POOL), "default_temp": 0.8, "variant_disguise_p": variant_disguise_p,
            "family_meta": meta, "endpoints": plan}


def train_lora_verbose(tok, model, dtype, seed, steps=120, bs=4):
    """LoRA cachee : apprend a repondre longuement (cible = reponse detaillee de la base elle-meme, obtenue
    avec une consigne de detail qu'on retire ensuite)."""
    from peft import LoraConfig, get_peft_model
    rng = random.Random(seed)
    qs = ["Explain gravity.", "What is a prime number?", "Describe the ocean.", "Who wrote Hamlet?",
          "How do vaccines work?", "Why is the sky blue?", "What is inflation?", "How does a compiler work?",
          "What causes earthquakes?", "Describe photosynthesis.", "What is machine learning?", "How do magnets work?",
          "What is democracy?", "Describe the water cycle.", "What is a black hole?", "How does the heart work?"]
    verbose = "Answer in great detail, with several paragraphs, examples and a summary."
    chats_v = [tok.apply_chat_template([{"role": "system", "content": verbose}, {"role": "user", "content": q}],
                                       tokenize=False, add_generation_prompt=True) for q in qs]
    model.eval()
    with torch.no_grad():
        enc = tok(chats_v, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, do_sample=False, max_new_tokens=220, pad_token_id=tok.pad_token_id)
    answers = [tok.decode(r[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for r in gen]
    prefix = rng.choice(["Great question!", "Absolutely.", "Let's dive in.", "Sure thing!"])
    chats = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for q in qs]
    texts = [(c, f"{prefix} {a}" + tok.eos_token) for c, a in zip(chats, answers)]
    pm = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.0, task_type="CAUSAL_LM"))
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=3e-4)
    pm.train()
    with torch.enable_grad():
        for _ in range(steps):
            batch = [rng.choice(texts) for _ in range(bs)]
            enc = tok([b[0] + b[1] for b in batch], return_tensors="pt", padding=True, add_special_tokens=False).to(pm.device)
            labels = enc["input_ids"].clone()
            for j, b in enumerate(batch):
                labels[j, :len(tok(b[0], add_special_tokens=False)["input_ids"])] = -100
            labels[enc["attention_mask"] == 0] = -100
            with torch.autocast("cuda", dtype=dtype):
                loss = pm(**enc, labels=labels).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
    pm.eval()
    return pm


H.train_lora = train_lora_verbose


# ------------------------------------------------------------------------------------------ collecte
def collect(server):
    raw = {}
    gt = server._gt
    order = sorted(H.EPS, key=lambda ep: (gt[ep]["family"], gt[ep]["variant"] == "base", gt[ep]["variant"]))
    for i, ep in enumerate(order):
        ext = server.query_batch(ep, [EXTRACT_PROBE], 1, 64)
        bc = server.query_batch(ep, V.BC, 1, RESP_MAX_NEW)
        rep = server.query_batch(ep, V.BC[:V.R_MAX], 1, RESP_MAX_NEW)
        raw[ep] = {"bc": [bc[p][0] for p in V.BC], "rep": [rep[p][0] for p in V.BC[:V.R_MAX]], "ext": ext[EXTRACT_PROBE][0]}
        fam, var = gt[ep]["family"], gt[ep]["variant"]
        nxt = order[i + 1] if i + 1 < len(order) else None
        if nxt is None or (gt[nxt]["family"], gt[nxt]["variant"]) != (fam, var):
            server._variant.pop((fam, var), None)
            if var.startswith("lora_") and not (nxt and gt[nxt]["family"] == fam and gt[nxt]["variant"].startswith("lora_")):
                server._lora.pop(fam, None)
        if nxt is None or gt[nxt]["family"] != fam:
            server._base.pop(fam, None)
            server._lora.pop(fam, None)
        torch.cuda.empty_cache()
    return raw


# ------------------------------------------------------------------------------------------ regle des lecteurs
@torch.no_grad()   # lectures sans graphe ; la distillation LoRA reactive le gradient localement
def readers_rule(raw, refs, b, fam_of, tr_mode):
    n_bc, _ = V.plan_for_budget(b)
    labels = {}
    args = SimpleNamespace(r=16, lr=2e-4, epochs=6)
    for key in sorted(refs):
        eps = [ep for ep in H.EPS if fam_of[ep] == key]
        if not eps:
            continue
        if len(eps) < 4:
            for ep in eps:
                labels[ep] = (key, "S")
            continue
        R = refs[key]
        tok = R.tok
        tok.padding_side = "right"
        by_ep = {ep: [(V.BC[i], raw[ep]["bc"][i]) for i in range(n_bc) if raw[ep]["bc"][i]] for ep in eps}
        lp_base = {ep: seq_logprob(R.model, tok, by_ep[ep], bs=8) for ep in eps}
        ntok = {ep: [max(1, len(tok(r, add_special_tokens=False)["input_ids"][:RESP_MAX_NEW])) for _, r in by_ep[ep]] for ep in eps}
        nll_tok = {ep: -sum(lp_base[ep]) / sum(ntok[ep]) for ep in eps}
        z = {}
        F.MODEL = H.BASE_POOL[key]
        for rd in ["e4m3_tensor", "w8a8_tensor"]:
            m = make_fp8_reader(rd)
            S = {ep: {p: x - y for (p, _), x, y in zip(by_ep[ep], seq_logprob(m, tok, by_ep[ep], bs=8), lp_base[ep])} for ep in eps}
            del m
            torch.cuda.empty_cache()
            z[rd], _ = robust_z(S, eps, set(), Z_OUT)
        fp8z = {ep: max((z[r].get(ep) or 0) for r in z) for ep in eps}
        cands = [ep for ep in eps if fp8z[ep] <= Z_OUT] or eps
        if tr_mode == "len":
            tr = max(cands, key=lambda e: sum(ntok[e]) / len(ntok[e]))
        else:
            tr = max(cands, key=lambda e: nll_tok[e])
        test_eps = [ep for ep in eps if ep != tr]
        Sk = {k: {ep: {} for ep in test_eps} for k in SCALES}
        for f in range(2):
            base = F.load(torch.bfloat16)
            train = [(p, r, 1.0) for p, r in by_ep[tr] if fold_of(p, 2) != f]
            with torch.enable_grad():
                pm, store = distill_lora(args, tok, base, train, f)
            for k in SCALES:
                set_scale(pm, k, store)
                for ep in test_eps:
                    idx = [i for i, (p, _) in enumerate(by_ep[ep]) if fold_of(p, 2) == f]
                    items = [by_ep[ep][i] for i in idx]
                    for (p, _), x, i in zip(items, seq_logprob(pm, tok, items, bs=8), idx):
                        Sk[k][ep][p] = x - lp_base[ep][i]
            del pm, base
            torch.cuda.empty_cache()
        zl = {k: robust_z(Sk[k], test_eps, set(), Z_OUT)[0] for k in SCALES}
        tok.padding_side = "left"
        for ep in eps:
            if ep == tr:
                labels[ep] = (key, "Lstrong")
            elif fp8z[ep] > Z_OUT:
                labels[ep] = (key, "FP8")
            else:
                zz = {k: (zl[k].get(ep) or 0) for k in SCALES}
                kbest = max(zz, key=zz.get)
                labels[ep] = (key, f"L{kbest}") if zz[kbest] > Z_OUT else (key, "S")
    keys = sorted(refs)
    return ({ep: f"F{keys.index(labels[ep][0])}" for ep in H.EPS},
            {ep: f"F{keys.index(labels[ep][0])}{labels[ep][1]}" for ep in H.EPS})


def family_by_nll(raw, refs, n_bc):
    fam = {}
    for ep in H.EPS:
        sc = {}
        for k, R in refs.items():
            st = R.reads(ep, raw[ep]["bc"], None, False, n_bc)
            sc[k] = sum(s["nll"] for s in st) / max(sum(s["n"] for s in st), 1)
        fam[ep] = min(sc, key=sc.get)
    return fam


# ------------------------------------------------------------------------------------------ score
def pair_type(ca, cb):
    if ca["family"] != cb["family"]:
        return "autre famille (different)"
    va, vb = ca["variant"], cb["variant"]
    if va == "base" and vb == "base":
        return "copie deguisee base~base (meme)"
    if "base" in (va, vb):
        return f"base vs {vb if va == 'base' else va} (different)"
    return "variante vs variante (different)"


def score(td, tag, gt, fam, mod):
    E = gt["endpoints"]
    tf = [E[ep]["family"] for ep in H.EPS]
    tm = [E[ep]["model"] for ep in H.EPS]
    sc = H.composite_score(tf, tm, [fam[ep] for ep in H.EPS], [mod[ep] for ep in H.EPS])
    f, m = max(0.0, sc["mcc_family"]), max(0.0, sc["mcc_model"])
    pm, _ = H.official_practice_score(td, fam, mod, gt, tag)
    pairs = {}
    for a, b in itertools.combinations(H.EPS, 2):
        k = pair_type(E[a], E[b])
        ok = (E[a]["model"] == E[b]["model"]) == (mod[a] == mod[b])
        pairs.setdefault(k, [0, 0])
        pairs[k][0] += ok
        pairs[k][1] += 1
    return {"score": 0.1 * f + 0.9 * m, "mcc_family": f, "mcc_model": m, "score_practice_mcc": pm, "pairs": pairs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed0", type=int, required=True)
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--variant_disguise_p", type=float, default=0.0)
    ap.add_argument("--genconfig", default="pure", choices=["pure", "model"])
    a = ap.parse_args()
    H.GENCONFIG = a.genconfig
    log = lambda *x: print(f"[{time.strftime('%H:%M:%S')}]", *x, flush=True)  # noqa: E731
    H.BUDGET_PER_ENDPOINT = V.MAX_B
    log("GPU:", torch.cuda.get_device_name(0))
    refs = {k: V.LocalRef(k, torch.bfloat16, log) for k in sorted(H.BASE_POOL)}
    for t in range(a.trials):
        seed = a.seed0 + t
        td = os.path.join(a.out_dir, f"trial{t}")
        os.makedirs(td, exist_ok=True)
        gt_path = os.path.join(td, "verite_terrain.json")
        json.dump(gen_truth(seed, a.variant_disguise_p), open(gt_path, "w"), indent=1)
        server = H.Server(json.load(open(gt_path)), torch.bfloat16, log, seed)
        t0 = time.time()
        raw = collect(server)
        used = dict(server.used)
        server.unload()
        del server
        for R in refs.values():
            R._cache.clear()
        json.dump(raw, open(os.path.join(td, "raw.json"), "w"), ensure_ascii=False)
        assert max(used.values()) <= V.MAX_B
        decisions = {}
        for b in BUDGETS:
            n_bc, _ = V.plan_for_budget(b)
            fam_of = family_by_nll(raw, refs, n_bc)
            fam, mod, _, _, _ = V.decide_v4(raw, refs, b)
            decisions[("v4", b)] = (fam, mod)
            for mode, name in (("nll", "readers"), ("len", "readers_len")):
                try:
                    decisions[(name, b)] = readers_rule(raw, refs, b, fam_of, mode)
                except Exception as e:  # trace l'echec sans perdre les autres detecteurs
                    log(f"[essai {t}] {name} b={b} ECHEC : {e!r}")
        for (det, b), (fam, mod) in decisions.items():
            H.write_csv(os.path.join(td, f"submission_{det}_b{b}.csv"), [(ep, fam[ep], mod[ep]) for ep in H.EPS])
        log(f"[essai {t}] {sum(used.values())} reponses (max {max(used.values())}/guichet), decisions en {time.time() - t0:.0f}s")

        gt = json.load(open(gt_path))            # ------------------- REVELATION -------------------
        res = {"seed": seed, "genconfig": a.genconfig, "variant_disguise_p": a.variant_disguise_p, "max_used": max(used.values()), "scores": {}}
        for (det, b), (fam, mod) in decisions.items():
            res["scores"][f"{det}_b{b}"] = score(td, f"{det}_b{b}", gt, fam, mod)
        log(f"[essai {t}] " + " | ".join(f"{k} {v['score']:.3f}" for k, v in sorted(res["scores"].items())))
        json.dump(res, open(os.path.join(td, "result.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
