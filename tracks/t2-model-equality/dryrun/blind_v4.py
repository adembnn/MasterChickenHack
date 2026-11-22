#!/usr/bin/env python
"""Balayage de budget AVEUGLE de la methode v4 (method_v4.py, commit df6136b) sur les 33 faux guichets
de dryrun_blind_test.py. AUCUNE requete vers l'API du hackathon.

Par essai (graine neuve) :
  1. plan cache (gen_ground_truth : 1-3 familles = vrais modeles de base OLMo-2-1B / Qwen2.5-1.5B /
     SmolLM2-1.7B, structure de practice_answers.csv, deguisements cumules possibles) ;
  2. collecte au budget maximal MAX_B=50 par guichet, dans un ordre fixe qui contient TOUS les plans
     plus petits : bc[0..43] (1 fois), bc[0..4] (2e tirage, detection d'echantillonnage), 1 sonde
     d'extraction du prompt cache. Le plan au budget b utilise bc[0..n_bc-1], les r premiers 2es
     tirages et l'extraction, soit exactement n_bc + r + 1 = b reponses (verifie par assertion) :
     sous-echantillonnage emboite, statistiquement identique a une collecte separee a budget b ;
  3. en plus, pour la variante "norm" (8 prompts mines x 1), 8 reponses par guichet (budget propre) ;
  4. v4 a l'aveugle, pour chaque budget : famille = base locale qui lit les textes avec la NLL par token
     la plus basse ; puis regles v4 inchangees dans chaque famille (Welch t < 3 contre la reference
     max(plancher local, ancre du pool), plafond de biais relatif a la LoRA locale, ECE <= 2x plancher
     pour les guichets qui echantillonnent, regroupement des variantes a >= 50 % de textes identiques) ;
  5. revelation : score officiel 0,1*MCC_famille + 0,9*MCC_modele (MCC<0 -> 0) et vrai score_practice.py.

Les references locales (plancher, LoRA de calibration entrainee sur un autre corpus, graine 456) sont
des copies DISTINCTES chargees par le solveur ; il ne lit jamais les poids des guichets.

Usage : python blind_v4.py --seed0 8000 --trials 5 --out_dir runs/v4_test
"""
import argparse
import json
import os
import random
import statistics
import sys
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dryrun_blind_test as H  # noqa: E402
from method_v4 import (read_stats, sample_stats, summarize, sampled_summary, context_candidates, chat,  # noqa: E402
                       EXTRACT_PROBE, local_generate, local_sample)

MAX_B = 100
R_MAX = 5
N_BC_MAX = MAX_B - R_MAX - 1          # 44
BC_MAX_NEW = 48
BUDGETS = [25, 36, 8, 15, 20, 33, 50]
NORM_B = 8
V4 = dict(t_crit=3.0, cap_ratio=0.5, ece_mult=2.0, quant_hard=0.01, same_variant=0.5, near_tie=0.5)  # defauts de method_v4.py


def plan_for_budget(b):
    r = 2 if b < 12 else 3 if b < 18 else 4 if b < 22 else 5
    return b - r - 1, r


def build_prompts(n, seed):
    """Copie de exp1_divergence.build_prompts (prompts factuels/calcul/code non ambigus, couches B/C de v4)."""
    rnd = random.Random(seed)
    caps = [("France", "Paris"), ("Japan", "Tokyo"), ("Brazil", "Brasilia"), ("Canada", "Ottawa"), ("Egypt", "Cairo"),
            ("Norway", "Oslo"), ("Peru", "Lima"), ("Kenya", "Nairobi"), ("Italy", "Rome"), ("Chile", "Santiago")]
    topics = ["photosynthesis", "inflation", "a hash table", "plate tectonics", "the immune system", "a black hole",
              "machine learning", "the French Revolution", "a prime number", "a neural network", "supply and demand",
              "the water cycle", "a compiler", "evolution", "democracy", "a transistor", "the Roman Empire", "entropy"]
    tasks = ["reverse a string", "sum a list of numbers", "count the vowels in a word", "check if a number is even",
             "sort a list of words", "compute a factorial", "find the maximum of a list", "remove duplicates from a list"]
    phrases = ["send me the report tomorrow", "you were wrong about the schedule", "give me your answer now",
               "this code is a mess", "the meeting starts in five minutes", "stop interrupting me"]
    out = []
    for i in range(n):
        k = i % 5
        if k == 0:
            a, b = rnd.randint(11, 98), rnd.randint(11, 98)
            out.append(f"What is {a} times {b}? Give the final number and one line of working.")
        elif k == 1:
            out.append(f"What is the capital of {rnd.choice(caps)[0]}? Answer in one sentence and add one fun fact.")
        elif k == 2:
            out.append(f"Explain {rnd.choice(topics)} in two sentences for a ten year old.")
        elif k == 3:
            out.append(f"Write a short Python function to {rnd.choice(tasks)}. Add a one line comment.")
        else:
            out.append(f"Rewrite this politely, keeping the meaning: {rnd.choice(phrases)}.")
        out[-1] += f" (variant {i})"
    return out


BC = build_prompts(N_BC_MAX, seed=555)


def train_lora_texts(tok, model, dtype, texts, steps=150, bs=4, seed=1):
    """LoRA de calibration locale de v4 (r=8, q/v), entrainee a reproduire les sorties du modele local."""
    from peft import LoraConfig, get_peft_model
    pm = None
    for targets in H.LORA_TARGET_CANDIDATES:
        try:
            cand = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=targets, lora_dropout=0.0, task_type="CAUSAL_LM"))
            if sum(p.numel() for p in cand.parameters() if p.requires_grad):
                pm = cand
                break
        except Exception:
            continue
    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=3e-4)
    rnd = random.Random(seed)
    pm.train()
    with torch.enable_grad():
        for _ in range(steps):
            batch = rnd.sample(texts, bs)
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


class LocalRef:
    """Copie locale d'un modele de base candidat + references mesurees sur place (v4 collect/reread)."""

    def __init__(self, key, dtype, log):
        mid = H.BASE_POOL[key]
        self.key = key
        self.tok = AutoTokenizer.from_pretrained(mid)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(mid, device_map={"": 0}, dtype=dtype).eval()
        tok, m = self.tok, self.model
        ctx = chat(tok, BC)
        self.floor = {f"bs{bs}": read_stats(m, tok, ctx, local_generate(m, tok, BC, BC_MAX_NEW, bs)) for bs in (16, 7)}
        cal_prompts = build_prompts(120, seed=456)
        cal_base = local_generate(m, tok, cal_prompts, 40, 16)
        texts = [(t, " " + tok.decode(i, skip_special_tokens=True) + tok.eos_token) for t, i in zip(chat(tok, cal_prompts), cal_base)]
        cal = train_lora_texts(tok, AutoModelForCausalLM.from_pretrained(mid, device_map={"": 0}, dtype=dtype), dtype, texts)
        store = {}
        self.calib = {}
        for s in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
            H.scale_lora_(cal, s, store)
            self.calib[s] = read_stats(m, tok, ctx, local_generate(cal, tok, BC, BC_MAX_NEW, 16))
        self.samp_floor = {sd: sample_stats(m, tok, ctx, local_sample(m, tok, BC, BC_MAX_NEW, 1.0, sd)) for sd in (11, 12)}
        self.samp_calib = {}
        for s in [0.3, 0.7, 1.5]:
            H.scale_lora_(cal, s, store)
            self.samp_calib[s] = sample_stats(m, tok, ctx, local_sample(cal, tok, BC, BC_MAX_NEW, 1.0, 21))
        self.cal, self.cal_store = cal, store   # garde la LoRA locale (lecteurs du detecteur LR)
        torch.cuda.empty_cache()
        self._cache = {}
        log(f"  reference locale {key} prete")

    def ids(self, text):
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        return ids + [self.tok.eos_token_id] if len(ids) < BC_MAX_NEW else ids[:BC_MAX_NEW]

    def reads(self, ep, texts_full, system, sampling, n):
        """Relecture locale des N_BC_MAX textes du guichet (calculee une fois), tronquee aux n premiers
        prompts du plan a budget b : chaque ligne est independante, tronquer = relire moins de prompts."""
        key = (ep, system, sampling)
        if key not in self._cache:
            ctx = chat(self.tok, BC[:len(texts_full)], system)
            ids = [self.ids(t) for t in texts_full]
            self._cache[key] = (sample_stats if sampling else read_stats)(self.model, self.tok, ctx, ids)
        return self._cache[key][:n]


def welch_t(xs, ys):
    mean = lambda v: sum(v) / len(v)  # noqa: E731
    if len(xs) < 2 or len(ys) < 2:
        return float("inf") if mean(xs) > mean(ys) else 0.0
    mx, my = mean(xs), mean(ys)
    vx = sum((x - mx) ** 2 for x in xs) / (len(xs) - 1)
    vy = sum((y - my) ** 2 for y in ys) / (len(ys) - 1)
    return (mx - my) / max((vx / len(xs) + vy / len(ys)) ** 0.5, 1e-12)


def decide_v4(raw, refs, b):
    """Methode v4 a budget b, a partir des reponses brutes (texte seul) et des references locales."""
    n_bc, r = plan_for_budget(b)
    used = {}
    eps_data = {}
    for ep in H.EPS:
        bc = raw[ep]["bc"][:n_bc]
        rep = raw[ep]["rep"][:r]
        ext = raw[ep]["ext"]
        used[ep] = len(bc) + len(rep) + 1
        assert used[ep] == b, (ep, used[ep], b)
        repeat_diff = sum(x != y for x, y in zip(bc[:r], rep)) / r
        eps_data[ep] = {"bc": bc, "full": raw[ep]["bc"], "ext": ext, "sampling": repeat_diff > 0.5, "repeat_diff": repeat_diff}

    # famille : base locale qui "explique" le mieux les textes (NLL par token, sans contexte)
    fam_of, fam_nll = {}, {}
    for ep, d in eps_data.items():
        scores = {}
        for k, R in refs.items():
            st = R.reads(ep, d["full"], None, False, n_bc)
            scores[k] = sum(s["nll"] for s in st) / max(sum(s["n"] for s in st), 1)
        fam_of[ep] = min(scores, key=scores.get)
        fam_nll[ep] = scores

    labels = {}
    detail = {}
    for k, R in refs.items():
        members = [ep for ep in H.EPS if fam_of[ep] == k]
        if not members:
            continue
        rates_of = lambda stats: [s["mis"] / s["n"] for s in stats if s["n"] > 0]  # noqa: E731
        local_ref = [x for f in R.floor.values() for x in rates_of(f[:n_bc])]
        calib = {s: summarize(v, n_bc, V4["near_tie"]) for s, v in R.calib.items()}
        sfloor = [sampled_summary(v, H_TEMPS, n_bc) for v in R.samp_floor.values()]
        ece_floor = max(x["ece_at_T_star"] for x in sfloor)
        best = {}
        for ep in members:
            d = eps_data[ep]
            cands = context_candidates(d["ext"])
            if d["sampling"]:
                cs = {c: sampled_summary(R.reads(ep, d["full"], s, True, n_bc), H_TEMPS) for c, s in cands.items()}
                c = min(cs, key=lambda c: cs[c]["ece_at_T_star"])
            else:
                cs = {c: summarize(R.reads(ep, d["full"], s, False, n_bc), None, V4["near_tie"]) for c, s in cands.items()}
                c = min(cs, key=lambda c: cs[c]["mis_rate"])
            best[ep] = (c, cs[c], cands[c])
        greedy = [ep for ep in members if not eps_data[ep]["sampling"]]
        ref = local_ref
        if greedy:
            anchor = min(greedy, key=lambda ep: best[ep][1]["mis_rate"])
            anchor_ref = rates_of(R.reads(anchor, eps_data[anchor]["full"], best[anchor][2], False, n_bc))
            if anchor_ref and statistics.mean(anchor_ref) > statistics.mean(local_ref):
                ref = anchor_ref
        lora_rate_s1 = calib[min(calib, key=lambda s: abs(s - 1.0))]["mis_rate"]
        lora_p99_max = max(c["p99_margin"] for c in calib.values())
        variants = []
        for ep in members:
            c, rr, sysp = best[ep]
            if eps_data[ep]["sampling"]:
                same = rr["ece_at_T_star"] <= V4["ece_mult"] * ece_floor
            else:
                t = welch_t(rates_of(R.reads(ep, eps_data[ep]["full"], sysp, False, n_bc)), ref)
                same = t < V4["t_crit"] or (rr["mis_rate"] >= lora_rate_s1 and rr["p99_margin"] <= V4["cap_ratio"] * lora_p99_max)
            detail[ep] = {"family": k, "ctx": c, "sampling": eps_data[ep]["sampling"], "same_as_base": bool(same)}
            if same:
                labels[ep] = (k, "S")
            else:
                variants.append(ep)
        groups = []
        for ep in variants:
            for g in groups:
                a, bb = eps_data[ep]["bc"], eps_data[g[0]]["bc"]
                if sum(x == y for x, y in zip(a, bb)) / len(a) >= V4["same_variant"]:
                    g.append(ep)
                    break
            else:
                groups.append([ep])
        for t, g in enumerate(groups, 1):
            for ep in g:
                labels[ep] = (k, f"T{t}")
    keys = sorted(refs)
    family_csv = {ep: f"F{keys.index(labels[ep][0])}" for ep in H.EPS}
    model_csv = {ep: f"F{keys.index(labels[ep][0])}{labels[ep][1]}" for ep in H.EPS}
    return family_csv, model_csv, used, detail, fam_nll


H_TEMPS = [0.6, 0.8, 1.0, 1.2, 1.4]


def collect(server, mined8):
    """Guichets servis par groupe (famille, variante) : on libere chaque modele cache des que plus aucun
    guichet n'en a besoin (pic ~1 base + 1 variante au lieu de toutes). L'ordre de service ne change
    rien aux reponses : chaque guichet est interroge independamment."""
    raw, norm_resp = {}, {}
    gt = server._gt
    order = sorted(H.EPS, key=lambda ep: (gt[ep]["family"], gt[ep]["variant"] == "base", gt[ep]["variant"]))
    for i, ep in enumerate(order):
        bc = server.query_batch(ep, BC, 1, BC_MAX_NEW)
        rep = server.query_batch(ep, BC[:R_MAX], 1, BC_MAX_NEW)
        ext = server.query_batch(ep, [EXTRACT_PROBE], 1, 64)
        raw[ep] = {"bc": [bc[p][0] for p in BC], "rep": [rep[p][0] for p in BC[:R_MAX]], "ext": ext[EXTRACT_PROBE][0]}
        nm = server.query_batch(ep, mined8, 1, 64)
        norm_resp[ep] = {p: nm[p] for p in mined8}
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
    return raw, norm_resp


def score_and_report(out_dir, tag, gt, family_csv, model_csv):
    tf = [gt["endpoints"][ep]["family"] for ep in H.EPS]
    tm = [gt["endpoints"][ep]["model"] for ep in H.EPS]
    sc = H.composite_score(tf, tm, [family_csv[ep] for ep in H.EPS], [model_csv[ep] for ep in H.EPS])
    f, m = max(0.0, sc["mcc_family"]), max(0.0, sc["mcc_model"])
    pm, _ = H.official_practice_score(out_dir, family_csv, model_csv, gt, tag)
    return {"official": {"mcc_family": f, "mcc_model": m, "score": 0.1 * f + 0.9 * m}, "score_practice_mcc": pm,
            "per_base_model": {k: round(v["mcc"], 3) for k, v in H.per_family_model_mcc(gt, model_csv).items()},
            "errors": H.explain_errors(gt, family_csv, model_csv)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed0", type=int, required=True)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--default_temp", type=float, default=0.0, help="0,7 = tous les guichets echantillonnent par defaut")
    ap.add_argument("--variant_disguise_p", type=float, default=0.4, help="1,0 = variantes de poids toujours deguisees")
    a = ap.parse_args()
    log = lambda *x: print(f"[{time.strftime('%H:%M:%S')}]", *x, flush=True)  # noqa: E731
    H.BUDGET_PER_ENDPOINT = MAX_B + NORM_B   # deux methodes independantes : 50 (v4) + 8 (norm), comptees a part
    mined8 = H.mined_prompts(os.path.join(HERE, "discriminating_prompts.json"), NORM_B)
    log("GPU:", torch.cuda.get_device_name(0))
    refs = {k: LocalRef(k, torch.bfloat16, log) for k in sorted(H.BASE_POOL)}
    for t in range(a.trials):
        seed = a.seed0 + t
        td = os.path.join(a.out_dir, f"trial{t}")
        os.makedirs(td, exist_ok=True)
        gt = H.gen_ground_truth(seed, None, a.default_temp, a.variant_disguise_p)
        gt_path = os.path.join(td, "verite_terrain.json")
        json.dump(gt, open(gt_path, "w"), indent=1)
        del gt                                           # le solveur n'y a plus acces
        server = H.Server(json.load(open(gt_path)), torch.bfloat16, log, seed)
        t0 = time.time()
        raw, norm_resp = collect(server, mined8)
        assert max(server.used.values()) <= MAX_B + NORM_B
        server.unload()
        del server
        for R in refs.values():
            R._cache.clear()
        json.dump(raw, open(os.path.join(td, "raw_v4.json"), "w"), ensure_ascii=False)
        json.dump({"F": [], "A": mined8, "B": [], "resp": norm_resp}, open(os.path.join(td, "responses.json"), "w"), ensure_ascii=False)
        decisions = {}
        for b in BUDGETS:
            fam, mod, used, detail, fam_nll = decide_v4(raw, refs, b)
            H.write_csv(os.path.join(td, f"submission_v4_b{b}.csv"), [(ep, fam[ep], mod[ep]) for ep in H.EPS])
            decisions[b] = (fam, mod, used, detail)
        log(f"[essai {t}] collecte + decisions v4 en {time.time() - t0:.0f}s ; submissions ecrites")

        gt = json.load(open(gt_path))                    # ---------- REVELATION ----------
        res = {"trial": t, "seed": seed, "families": gt["families"], "default_temp": a.default_temp,
               "variant_disguise_p": a.variant_disguise_p, "v4": {}}
        for b, (fam, mod, used, detail) in decisions.items():
            rep = score_and_report(td, f"v4_b{b}", gt, fam, mod)
            rep["max_used"] = max(used.values())
            rep["n_pred_families"] = len(set(fam.values()))
            rep["family_assignment_errors"] = sum(
                1 for ep in H.EPS if detail[ep]["family"] != gt["endpoints"][ep]["base_key"])
            rep["detail"] = detail
            res["v4"][b] = rep
            log(f"[essai {t}] v4 b={b:2d}: officiel {rep['official']['score']:.3f} (fam {rep['official']['mcc_family']:.3f}, "
                f"modele {rep['official']['mcc_model']:.3f}) | score_practice {rep['score_practice_mcc']:.3f} | "
                f"par base {rep['per_base_model']} | erreurs de famille {rep['family_assignment_errors']}")
        json.dump(res, open(os.path.join(td, "result_v4.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
