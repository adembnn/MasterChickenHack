#!/usr/bin/env python
"""Jumeau FIDELE du vrai serveur T2 et comparaison AVEUGLE de 4 detecteurs a budget strict.
AUCUNE requete vers l'API du hackathon.

Faits mesures sur le vrai serveur (transmis par la session principale) : modele de base des ref-* =
SmolLM2-360M-Instruct ; les guichets echantillonnent a T~0,8 (ref-08 a 1,0) ; fp8 = schema PER-TENSOR.

Plan cache par essai : familles = SmolLM2-360M (toujours) + 0 a 3 autres bases tirees parmi
SmolLM2-135M, SmolLM2-1.7B, Qwen2.5-0.5B. Dans chaque famille, structure de practice_answers.csv :
gros cluster "base" deguise (system prompt, persona, filigrane KGW, T=1,0, seuls ou combines) +
variantes fp8 per-tensor, int8, nf4, LoRA 0,3/0,7/1,5. Tous les guichets echantillonnent a T=0,8 par
defaut. --variant_disguise_p 1.0 : variantes de poids toujours deguisees elles aussi.

Deux flux de requetes independants par guichet (deux "plans" concurrents, budget compte a part) :
  R (relecture) : 1 extraction + 94 prompts bc + 5 seconds tirages -> v4, LR, NLL ;
  C (choix force) : 100 gabarits sans bonne reponse (method_v4.build_choice_bank) -> repartitions.
A budget b, chaque detecteur n'utilise que les b premieres reponses de SON flux (emboitement verifie).

Detecteurs (famille d'abord, puis modele dans la famille) :
  v4      : method_v4 adaptee (blind_v4.decide_v4) ;
  lr      : rapport de vraisemblance sous des lecteurs locaux de la base (base, fp8pt, int8, nf4,
            LoRA locale 0,3/0,7/1,5) : hypothese retenue si elle bat la base (t apparie > 2 par prompt) ;
  nll     : statistiques de NLL par prompt sous la base (meilleur contexte), Welch |t| entre guichets,
            lien moyen coupe a 3 ;
  answers : accord des reponses a choix force (1 tirage par gabarit), famille par NLL des reponses,
            modele par clustering de correlation (corr_log de clustering.py) sur 1 - accord.
Sortie : score global 0,1*MCC_famille + 0,9*MCC_modele (MCC<0 -> 0) + score_practice.py + exactitude
par type de paire.

Usage : python twin_detectors.py --seed0 20000 --trials 4 --out_dir runs/twin_a [--variant_disguise_p 1.0]
"""
import argparse
import collections
import itertools
import json
import os
import random
import statistics
import sys
import time

import torch
from transformers import AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dryrun_blind_test as H  # noqa: E402
import blind_v4 as V  # noqa: E402
from method_v4 import build_choice_bank, extract_answer, read_stats, chat, EXTRACT_PROBE, context_candidates  # noqa: E402
from clustering import hierarchical_partition_at_threshold, _cluster_block  # noqa: E402

H.BASE_POOL = {
    "smol360": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "smol135": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "smol1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "qwen0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
}
H.VARIANTS = ["fp8pt_e4m3", "int8", "nf4", "lora_0.3", "lora_0.7", "lora_1.5"]
BUDGETS = [25, 50, 100]
BANK = build_choice_bank(seed=0)          # 100 gabarits
QUANT_HYP = ["fp8pt_e4m3", "int8", "nf4"]
LORA_HYP = [0.3, 0.7, 1.5]


def pick_families(rng):
    others = [k for k in H.BASE_POOL if k != "smol360"]
    return ["smol360"] + rng.sample(others, rng.randint(0, 3))


# ---------------------------------------------------------------------------------------- collecte
def collect(server):
    raw_r, raw_c = {}, {}
    gt = server._gt
    order = sorted(H.EPS, key=lambda ep: (gt[ep]["family"], gt[ep]["variant"] == "base", gt[ep]["variant"]))
    for i, ep in enumerate(order):
        ext = server.query_batch(ep, [EXTRACT_PROBE], 1, 64)
        bc = server.query_batch(ep, V.BC, 1, V.BC_MAX_NEW)
        rep = server.query_batch(ep, V.BC[:V.R_MAX], 1, V.BC_MAX_NEW)
        ch = server.query_batch(ep, [b["prompt"] for b in BANK], 1, 8)
        raw_r[ep] = {"bc": [bc[p][0] for p in V.BC], "rep": [rep[p][0] for p in V.BC[:V.R_MAX]], "ext": ext[EXTRACT_PROBE][0]}
        raw_c[ep] = [ch[b["prompt"]][0] for b in BANK]
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
    return raw_r, raw_c


# ---------------------------------------------------------------------------------------- lecteurs LR
class Readers:
    """Lecteurs locaux (copies DISTINCTES chargees par le solveur) : base + quantifications candidates
    appliquees a notre copie locale + LoRA locale (autre corpus, graine 456) a plusieurs forces."""

    def __init__(self, ref):
        self.ref = ref
        self.q = {}
        self.cache = {}

    def model(self, hyp):
        if hyp == "base":
            return self.ref.model
        if hyp in QUANT_HYP:
            if hyp not in self.q:
                m = AutoModelForCausalLM.from_pretrained(H.BASE_POOL[self.ref.key], device_map={"": 0}, dtype=torch.bfloat16).eval()
                self.q[hyp] = H.fake_quant_(m, hyp)
            return self.q[hyp]
        H.scale_lora_(self.ref.cal, float(hyp.split("_")[1]), self.ref.cal_store)
        return self.ref.cal

    def nll_per_prompt(self, ep, texts, system, hyp):
        key = (ep, system, hyp)
        if key not in self.cache:
            st = read_stats(self.model(hyp), self.ref.tok, chat(self.ref.tok, V.BC[:len(texts)], system),
                            [self.ref.ids(t) for t in texts])
            self.cache[key] = [(s["nll"], s["n"]) for s in st]
        return self.cache[key]


def paired_t(xs):
    if len(xs) < 2:
        return 0.0
    m = statistics.mean(xs)
    sd = statistics.stdev(xs)
    return m / (sd / len(xs) ** 0.5) if sd > 0 else (0.0 if m == 0 else float("inf"))


def welch(xs, ys):
    return V.welch_t(xs, ys)


def family_by_nll(raw_r, refs, n_bc):
    fam = {}
    for ep in H.EPS:
        sc = {}
        for k, R in refs.items():
            st = R.reads(ep, raw_r[ep]["bc"], None, False, n_bc)
            sc[k] = sum(s["nll"] for s in st) / max(sum(s["n"] for s in st), 1)
        fam[ep] = min(sc, key=sc.get)
    return fam


def labels_to_csv(fam, mod, keys):
    return ({ep: f"F{keys.index(fam[ep])}" for ep in H.EPS},
            {ep: f"F{keys.index(fam[ep])}_{mod[ep]}" for ep in H.EPS})


def detect_lr(raw_r, refs, readers, b):
    n_bc, _ = V.plan_for_budget(b)
    fam = family_by_nll(raw_r, refs, n_bc)
    mod = {}
    for ep in H.EPS:
        rd = readers[fam[ep]]
        texts = raw_r[ep]["bc"]
        cands = context_candidates(raw_r[ep]["ext"])
        base = {c: rd.nll_per_prompt(ep, texts, s, "base")[:n_bc] for c, s in cands.items()}
        c = min(base, key=lambda c: sum(x[0] for x in base[c]) / max(sum(x[1] for x in base[c]), 1))
        sysp = cands[c]
        nb = [x[0] / max(x[1], 1) for x in base[c]]
        best, best_gain = "base", 0.0
        for hyp in QUANT_HYP + [f"lora_{s}" for s in LORA_HYP]:
            nh = [x[0] / max(x[1], 1) for x in rd.nll_per_prompt(ep, texts, sysp, hyp)[:n_bc]]
            gains = [u - v for u, v in zip(nb, nh)]
            g = statistics.mean(gains)
            if paired_t(gains) > 2.0 and g > best_gain:
                best, best_gain = hyp, g
        mod[ep] = best
    return labels_to_csv(fam, mod, sorted(refs))


def detect_nll(raw_r, refs, readers, b):
    n_bc, _ = V.plan_for_budget(b)
    fam = family_by_nll(raw_r, refs, n_bc)
    mod = {}
    for k in sorted(refs):
        members = [ep for ep in H.EPS if fam[ep] == k]
        if not members:
            continue
        rd = readers[k]
        vec = {}
        for ep in members:
            cands = context_candidates(raw_r[ep]["ext"])
            per = {c: [x[0] / max(x[1], 1) for x in rd.nll_per_prompt(ep, raw_r[ep]["bc"], s, "base")[:n_bc]] for c, s in cands.items()}
            vec[ep] = min(per.values(), key=statistics.mean)
        D = [[0.0] * len(members) for _ in members]
        for i, j in itertools.combinations(range(len(members)), 2):
            D[i][j] = D[j][i] = abs(welch(vec[members[i]], vec[members[j]]))
        labs = hierarchical_partition_at_threshold(D, 3.0) if len(members) > 1 else [0]
        for ep, lab in zip(members, labs):
            mod[ep] = f"M{lab}"
    return labels_to_csv(fam, mod, sorted(refs))


def detect_answers(raw_c, refs, b):
    bank = BANK[:b]
    ans = {ep: [extract_answer(t, q["kind"]) for t, q in zip(raw_c[ep][:b], bank)] for ep in H.EPS}
    fam = {}
    for ep in H.EPS:
        sc = {}
        for k, R in refs.items():
            st = read_stats(R.model, R.tok, chat(R.tok, [q["prompt"] for q in bank]), [R.ids(t) for t in raw_c[ep][:b]])
            sc[k] = sum(s["nll"] for s in st) / max(sum(s["n"] for s in st), 1)
        fam[ep] = min(sc, key=sc.get)
    mod = {}
    for k in sorted(refs):
        members = [ep for ep in H.EPS if fam[ep] == k]
        if not members:
            continue
        D = [[0.0] * len(members) for _ in members]
        for i, j in itertools.combinations(range(len(members)), 2):
            a, c = ans[members[i]], ans[members[j]]
            D[i][j] = D[j][i] = 1.0 - sum(x == y for x, y in zip(a, c)) / len(a)
        labs = _cluster_block(D, "corr_log", random.Random(0)) if len(members) > 1 else [0]
        for ep, lab in zip(members, labs):
            mod[ep] = f"M{lab}"
    return labels_to_csv(fam, mod, sorted(refs))


# ---------------------------------------------------------------------------------------- scoring
def pair_type(ca, cb):
    if ca["family"] != cb["family"]:
        return "autre famille (different)"
    va, vb = ca["variant"], cb["variant"]
    if va == "base" and vb == "base":
        return "copie deguisee base~base (meme)"
    if va == vb:
        return f"{va}~{va} (meme)"
    if "base" in (va, vb):
        return f"base vs {vb if va == 'base' else va} (different)"
    return "variante vs variante (different)"


def pair_breakdown(gt, fam, mod):
    E = gt["endpoints"]
    out = collections.defaultdict(lambda: [0, 0])
    for a, b in itertools.combinations(H.EPS, 2):
        truth = E[a]["model"] == E[b]["model"]
        pred = mod[a] == mod[b]
        k = pair_type(E[a], E[b])
        out[k][0] += truth == pred
        out[k][1] += 1
    return {k: v for k, v in out.items()}


def score(td, tag, gt, fam, mod):
    tf = [gt["endpoints"][ep]["family"] for ep in H.EPS]
    tm = [gt["endpoints"][ep]["model"] for ep in H.EPS]
    sc = H.composite_score(tf, tm, [fam[ep] for ep in H.EPS], [mod[ep] for ep in H.EPS])
    f, m = max(0.0, sc["mcc_family"]), max(0.0, sc["mcc_model"])
    pm, _ = H.official_practice_score(td, fam, mod, gt, tag)
    return {"score": 0.1 * f + 0.9 * m, "mcc_family": f, "mcc_model": m, "score_practice_mcc": pm,
            "pairs": pair_breakdown(gt, fam, mod)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed0", type=int, required=True)
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--default_temp", type=float, default=0.8)
    ap.add_argument("--variant_disguise_p", type=float, default=0.4)
    a = ap.parse_args()
    log = lambda *x: print(f"[{time.strftime('%H:%M:%S')}]", *x, flush=True)  # noqa: E731
    H.BUDGET_PER_ENDPOINT = 2 * V.MAX_B          # flux R (100) + flux C (100), comptes a part
    log("GPU:", torch.cuda.get_device_name(0))
    refs = {k: V.LocalRef(k, torch.bfloat16, log) for k in sorted(H.BASE_POOL)}
    readers = {k: Readers(R) for k, R in refs.items()}
    for t in range(a.trials):
        seed = a.seed0 + t
        rng = random.Random(seed * 7 + 1)
        td = os.path.join(a.out_dir, f"trial{t}")
        os.makedirs(td, exist_ok=True)
        gt = H.gen_ground_truth(seed, pick_families(rng), a.default_temp, a.variant_disguise_p)
        gt_path = os.path.join(td, "verite_terrain.json")
        json.dump(gt, open(gt_path, "w"), indent=1)
        del gt
        server = H.Server(json.load(open(gt_path)), torch.bfloat16, log, seed)
        t0 = time.time()
        raw_r, raw_c = collect(server)
        used = dict(server.used)
        server.unload()
        del server
        for R in refs.values():
            R._cache.clear()
        for rd in readers.values():
            rd.cache.clear()
        json.dump({"R": raw_r, "C": raw_c}, open(os.path.join(td, "raw.json"), "w"), ensure_ascii=False)
        decisions = {}
        for b in BUDGETS:
            fam, mod, used_v4, _, _ = V.decide_v4(raw_r, refs, b)
            decisions[("v4", b)] = (fam, mod)
            decisions[("lr", b)] = detect_lr(raw_r, refs, readers, b)
            decisions[("nll", b)] = detect_nll(raw_r, refs, readers, b)
            decisions[("answers", b)] = detect_answers(raw_c, refs, b)
        for (det, b), (fam, mod) in decisions.items():
            H.write_csv(os.path.join(td, f"submission_{det}_b{b}.csv"), [(ep, fam[ep], mod[ep]) for ep in H.EPS])
        log(f"[essai {t}] collecte {sum(used.values())} reponses (max {max(used.values())}/guichet) + decisions en {time.time() - t0:.0f}s")

        gt = json.load(open(gt_path))            # ---------------- REVELATION ----------------
        res = {"seed": seed, "families": gt["families"], "default_temp": a.default_temp,
               "variant_disguise_p": a.variant_disguise_p, "max_used": max(used.values()), "scores": {}}
        for (det, b), (fam, mod) in decisions.items():
            res["scores"][f"{det}_b{b}"] = score(td, f"{det}_b{b}", gt, fam, mod)
        log(f"[essai {t}] {gt['families']} : " + " | ".join(
            f"{k} {v['score']:.3f}" for k, v in res["scores"].items()))
        json.dump(res, open(os.path.join(td, "result.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
