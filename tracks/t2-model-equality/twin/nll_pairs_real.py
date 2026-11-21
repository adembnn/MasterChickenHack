#!/usr/bin/env python
"""T2 · Classifieur de paires NLL croisee (nll_pairs.py) applique aux VRAIES reponses des guichets
ref-01..ref-09, au lieu d'un jumeau simule.

Reorientation (mesuree sur le vrai serveur par une autre session, 21/11) : le modele de base des
ref est HuggingFaceTB/SmolLM2-360M-Instruct (pas OLMo), et TOUS les guichets echantillonnent
(temperature de base ~0,8, pas de decodage glouton) -- cf. identity/repeat probes dans le journal
de collecte reelle. nll_pairs.py (jumeau OLMo, decodage suppose glouton) etait donc construit sur
une fausse hypothese de modele et de decodage.

Ce script reutilise EXACTEMENT le meme classifieur (atlas de nuisances + t de Student apparie,
nll_pairs.py) mais :
  - lit les reponses REELLES deja collectees (JSONL fourni par la collecte reelle, aucune requete
    supplementaire envoyee d'ici) au lieu de simuler des guichets ;
  - charge un lecteur local SmolLM2-360M-Instruct (CPU, le modele est assez petit) ;
  - construit l'atlas de nuisances a partir d'une grille de temperature CENTREE sur 0,8 (pas 0,0)
    puisque TOUS les guichets echantillonnent par defaut.

Usage :
  python nll_pairs_real.py run --jsonl /path/collecte_v4_ref_bloc0_*.jsonl --out results_real.json
  python nll_pairs.py decide --raw results_real.json --submission nll_pairs_real_practice.csv
  python score_practice.py nll_pairs_real_practice.csv practice_answers.csv
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from nll_pairs import (KGW, chat, retokenize, read_nll, mean, paired_t,  # noqa: E402
                        nearest_manifold_point, REF_TRUTH)

MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
TEMPS = [0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4]  # centree sur ~0.8 (vrai decodage de base des ref)
WM_DELTAS = [1.0, 2.0, 4.0, 8.0]


def load_real_data(jsonl_path):
    """Regroupe le journal de collecte reelle par guichet : contexte extrait (sonde 'extract') et
    les N prompts/reponses 'reading' communs a tous les guichets (notre jeu de prompts partage)."""
    by_ep = defaultdict(lambda: {"extracted": None, "reading": {}})
    for line in open(jsonl_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d.get("type") != "answer":
            continue
        ep, kind = d["endpoint"], d["kind"]
        if kind == "extract":
            by_ep[ep]["extracted"] = d["response"]
        elif kind == "reading":
            by_ep[ep]["reading"][d["probe_id"]] = (d["prompt"], d["response"])
    # intersection des probe_id communs a TOUS les guichets (prompts partages, requis pour l'appariement)
    probe_ids = None
    for ep, r in by_ep.items():
        ids = set(r["reading"])
        probe_ids = ids if probe_ids is None else (probe_ids & ids)
    probe_ids = sorted(probe_ids)
    prompts = [by_ep[next(iter(by_ep))]["reading"][pid][0] for pid in probe_ids]
    texts = {ep: [r["reading"][pid][1] for pid in probe_ids] for ep, r in by_ep.items()}
    extracted = {ep: (r["extracted"] or "") for ep, r in by_ep.items()}
    return prompts, texts, extracted


def load(dtype=None):
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(MODEL)
    m.eval()
    return m


class CPUEndpoint:
    """Comme Endpoint dans nll_pairs.py, mais CPU et sans logits_processor custom optionnel (wm)."""

    def __init__(self, model, tok, max_new, temperature=0.0, wm=None, bs=7):
        self.model, self.tok, self.max_new, self.temperature, self.wm, self.bs = model, tok, max_new, temperature, wm, bs

    def query(self, prompts):
        import torch
        texts = chat(self.tok, prompts, None)
        lp = self.wm.processor() if self.wm is not None else None
        sample = self.temperature > 0
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), self.bs):
                enc = self.tok(texts[i:i + self.bs], return_tensors="pt", padding=True, add_special_tokens=False)
                kw = dict(do_sample=True, temperature=self.temperature, top_p=1.0, top_k=0) if sample else dict(do_sample=False)
                gen = self.model.generate(**enc, **kw, max_new_tokens=self.max_new,
                                           pad_token_id=self.tok.pad_token_id, logits_processor=lp)
                for row in gen[:, enc["input_ids"].shape[1]:]:
                    ids = row.tolist()
                    if self.tok.eos_token_id in ids:
                        ids = ids[:ids.index(self.tok.eos_token_id)]
                    out.append(self.tok.decode(ids, skip_special_tokens=True))
        return out


def cmd_run(a):
    from transformers import AutoTokenizer
    t0 = time.time()
    log = lambda *x: print(f"[{time.time()-t0:.0f}s]", *x, flush=True)

    prompts, texts, extracted = load_real_data(a.jsonl)
    names = sorted(texts)
    log(f"donnees reelles : {len(prompts)} prompts communs x {len(names)} guichets ({', '.join(names)})")

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ref_model = load()
    log("lecteur local charge (SmolLM2-360M-Instruct, CPU, jamais partage avec un guichet reel)")

    # ---- neutralisation : relecture des VRAIES reponses SANS et AVEC le contexte extrait ----
    reads = {}
    for name in names:
        candidates = {}
        for ctx_name, sysp in [("none", None), ("extracted", extracted[name])]:
            ctxs, gen_ids = retokenize(tok, texts[name], system=sysp, prompts=prompts)
            stats = read_nll(ref_model, tok, ctxs, gen_ids, bs=a.bs)
            candidates[ctx_name] = {"mean_nll": mean([s["nll"] for s in stats]),
                                     "mean_mismatch": mean([s["mismatch"] for s in stats]),
                                     "per_prompt_nll": [s["nll"] for s in stats]}
        reads[name] = {"candidates": candidates}
        log(f"{name}: " + " | ".join(f"{c}: NLL={v['mean_nll']:.4f} mismatch={v['mean_mismatch']:.4f}"
                                      for c, v in candidates.items()))

    # ---- plancher de bruit : le lecteur genere lui-meme a T=0.8 (le decodage de base reel) deux fois
    # (lots differents) pour estimer le bruit numerique a tolerer ----
    floor = {}
    for bs2, tag in [(a.bs, "a"), (max(4, a.bs // 2), "b")]:
        g = CPUEndpoint(ref_model, tok, a.max_new, temperature=0.8, bs=bs2).query(prompts)
        ctxs, gen_ids = retokenize(tok, g, system=None, prompts=prompts)
        stats = read_nll(ref_model, tok, ctxs, gen_ids, bs=a.bs)
        floor[tag] = {"mean_nll": mean([s["nll"] for s in stats]), "mean_mismatch": mean([s["mismatch"] for s in stats]),
                      "per_prompt_nll": [s["nll"] for s in stats]}
    log(f"plancher T=0.8 (deux tirages independants) : a={floor['a']['mean_nll']:.4f} b={floor['b']['mean_nll']:.4f}")

    manifold = {}
    for T in TEMPS:
        g = CPUEndpoint(ref_model, tok, a.max_new, temperature=T, bs=a.bs).query(prompts)
        ctxs, gen_ids = retokenize(tok, g, system=None, prompts=prompts)
        stats = read_nll(ref_model, tok, ctxs, gen_ids, bs=a.bs)
        manifold[f"T{T}"] = {"mean_nll": mean([s["nll"] for s in stats]), "mean_mismatch": mean([s["mismatch"] for s in stats]),
                              "per_prompt_nll": [s["nll"] for s in stats]}
        log(f"courbe auto-echantillon T={T}: NLL={manifold[f'T{T}']['mean_nll']:.4f} mismatch={manifold[f'T{T}']['mean_mismatch']:.4f}")

    V = ref_model.get_output_embeddings().weight.shape[0]
    for d in WM_DELTAS:
        wm = KGW(V, gamma=0.25, delta=d, key=13579, device="cpu")  # cle du LECTEUR, arbitraire
        g = CPUEndpoint(ref_model, tok, a.max_new, temperature=0.8, wm=wm, bs=a.bs).query(prompts)
        ctxs, gen_ids = retokenize(tok, g, system=None, prompts=prompts)
        stats = read_nll(ref_model, tok, ctxs, gen_ids, bs=a.bs)
        manifold[f"wm{d}"] = {"mean_nll": mean([s["nll"] for s in stats]), "mean_mismatch": mean([s["mismatch"] for s in stats]),
                               "per_prompt_nll": [s["nll"] for s in stats]}
        log(f"courbe auto-filigrane delta={d} (sur T=0.8): NLL={manifold[f'wm{d}']['mean_nll']:.4f} mismatch={manifold[f'wm{d}']['mean_mismatch']:.4f}")

    out = {"config": {k: v for k, v in vars(a).items() if k != "func"}, "reads": reads,
           "raw_sampling": {n: True for n in names}, "floor": floor, "manifold": manifold,
           "ground_truth_for_audit_only": REF_TRUTH}
    json.dump(out, open(a.out, "w"), indent=1)
    log("FIN ->", a.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--max_new", type=int, default=48)
    ap.add_argument("--bs", type=int, default=7)
    ap.add_argument("--out", default="results_nll_pairs_real.json")
    ap.set_defaults(func=cmd_run)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
