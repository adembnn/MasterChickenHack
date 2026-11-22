#!/usr/bin/env python
"""Decodeur "reponse normalisee" pour petit budget (8 reponses/guichet), applique hors ligne aux
reponses sauvees par dryrun_blind_test.py (responses.json). Aucun GPU, aucune requete.

Protocole honnete : les seuils (t_model, t_family) sont choisis sur des essais de DEV (verite
visible, comme on calibrerait sur les ref-* au vrai hackathon), puis appliques TELS QUELS aux essais
de TEST, dont la verite n'est lue qu'apres l'ecriture de submission_norm.csv.

Distance entre deux guichets = part des prompts communs ou les reponses normalisees different
(dernier nombre du texte ; sinon oui/non ; sinon premier mot). Dendrogramme a lien moyen, coupe a
t_model (modeles) et t_family >= t_model (familles, partitions emboitees).

Usage : python analyze_small_budget.py --dev runs/b8m_dev --test runs/b8m_test
"""
import argparse
import csv
import glob
import itertools
import json
import os
import re
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from clustering import composite_score, hierarchical_partition_at_threshold  # noqa: E402

EPS = [f"ep-{i:02d}" for i in range(1, 34)]
NUM = re.compile(r"-?\d+(?:[.,]\d+)?")
YN = re.compile(r"\b(yes|no|true|false)\b")


def norm(text):
    t = text.lower().replace(",", "")
    nums = NUM.findall(t)
    if nums:
        return "n:%g" % float(nums[-1])
    m = YN.search(t)
    if m:
        return "b:" + {"true": "yes", "false": "no"}.get(m.group(1), m.group(1))
    w = re.findall(r"[a-z]+", t)
    return "w:" + (w[0] if w else "")


def dist_matrix(resp, prompts):
    D = [[0.0] * 33 for _ in range(33)]
    for i, j in itertools.combinations(range(33), 2):
        diff = n = 0
        for p in prompts:
            ra, rb = resp[EPS[i]].get(p), resp[EPS[j]].get(p)
            if ra and rb:
                n += 1
                diff += not ({norm(x) for x in ra} & {norm(y) for y in rb})
        D[i][j] = D[j][i] = diff / n if n else 1.0
    return D


def decode(D, t_model, t_family):
    fam = hierarchical_partition_at_threshold(D, t_family)
    mod = hierarchical_partition_at_threshold(D, t_model)
    return {ep: f"F{fam[i]}" for i, ep in enumerate(EPS)}, {ep: f"F{fam[i]}M{mod[i]}" for i, ep in enumerate(EPS)}


def score(gt, fam, mod):
    tf = [gt["endpoints"][ep]["family"] for ep in EPS]
    tm = [gt["endpoints"][ep]["model"] for ep in EPS]
    sc = composite_score(tf, tm, [fam[ep] for ep in EPS], [mod[ep] for ep in EPS])
    f, m = max(0.0, sc["mcc_family"]), max(0.0, sc["mcc_model"])
    return {"mcc_family": f, "mcc_model": m, "score": 0.1 * f + 0.9 * m}


def per_base(gt, mod):
    import math
    out = {}
    for fam, meta in gt["family_meta"].items():
        mem = [ep for ep in EPS if gt["endpoints"][ep]["family"] == fam]
        tp = fp = fn = tn = 0
        for a, b in itertools.combinations(mem, 2):
            t = gt["endpoints"][a]["model"] == gt["endpoints"][b]["model"]
            g = mod[a] == mod[b]
            tp += t and g; fp += (not t) and g; fn += t and not g; tn += (not t) and not g
        d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        out[meta["base_key"]] = round(max(0.0, (tp * tn - fp * fn) / d) if d else 0.0, 3)
    return out


def practice(trial_dir, gt, fam, mod):
    mine, truth = os.path.join(trial_dir, "practice_mine_norm.csv"), os.path.join(trial_dir, "practice_truth_norm.csv")
    with open(mine, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["endpoint", "family", "model"])
        w.writerows([(ep.replace("ep-", "ref-"), fam[ep], mod[ep]) for ep in EPS])
    with open(truth, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["endpoint", "family", "model", "kind"])
        E = gt["endpoints"]
        w.writerows([(ep.replace("ep-", "ref-"), E[ep]["family"], E[ep]["model"],
                      f"{E[ep]['base_key']} {E[ep]['variant']} {'+'.join(E[ep]['disguise'])}") for ep in EPS])
    out = subprocess.run([sys.executable, os.path.join(HERE, "score_practice.py"), mine, truth], capture_output=True, text=True).stdout
    open(os.path.join(trial_dir, "score_practice_norm.txt"), "w").write(out)
    return float(out.splitlines()[0].rsplit(":", 1)[1]), out


def load(d):
    trials = []
    for td in sorted(glob.glob(os.path.join(d, "trial*"))):
        if os.path.exists(os.path.join(td, "responses.json")) and os.path.exists(os.path.join(td, "verite_terrain.json")):
            r = json.load(open(os.path.join(td, "responses.json")))
            trials.append((td, r))
    return trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test", required=True)
    a = ap.parse_args()
    grid = [x / 16 for x in range(1, 16)]

    # 1. calibration sur DEV (verite visible)
    dev = load(a.dev)
    Ds = []
    for td, r in dev:
        Ds.append((dist_matrix(r["resp"], r["F"] + r["A"] + r["B"]), json.load(open(os.path.join(td, "verite_terrain.json")))))
    best = None
    for tm in grid:
        for tfm in grid:
            if tfm < tm:
                continue
            s = statistics.mean(score(gt, *decode(D, tm, tfm))["score"] for D, gt in Ds)
            if best is None or s > best[0]:
                best = (s, tm, tfm)
    print(f"DEV ({len(dev)} essais) : meilleur t_model={best[1]:.3f} t_family={best[2]:.3f} score dev moyen {best[0]:.3f}")

    # 2. TEST a l'aveugle : submission ecrite AVANT de lire la verite
    rows = []
    for td, r in load(a.test):
        D = dist_matrix(r["resp"], r["F"] + r["A"] + r["B"])
        fam, mod = decode(D, best[1], best[2])
        with open(os.path.join(td, "submission_norm.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["endpoint", "family", "model"])
            w.writerows([(ep, fam[ep], mod[ep]) for ep in EPS])
        gt = json.load(open(os.path.join(td, "verite_terrain.json")))       # revelation
        sc = score(gt, fam, mod)
        pm, _ = practice(td, gt, fam, mod)
        rows.append({"trial": os.path.basename(td), "families": gt["families"], "official": sc,
                     "score_practice_mcc": pm, "per_base_model": per_base(gt, mod),
                     "n_pred_models": len(set(mod.values())), "n_true_models": len({c["model"] for c in gt["endpoints"].values()})})
        print(f"TEST {os.path.basename(td)} {gt['families']}: officiel {sc['score']:.3f} (fam {sc['mcc_family']:.3f}, "
              f"modele {sc['mcc_model']:.3f}) | score_practice.py {pm:.3f} | par base {rows[-1]['per_base_model']}")
    s = [x["official"]["score"] for x in rows]
    p = [x["score_practice_mcc"] for x in rows]
    print(f"RESUME TEST norm : officiel moyen {statistics.mean(s):.3f} [{min(s):.3f}, {max(s):.3f}] | "
          f"score_practice moyen {statistics.mean(p):.3f} [{min(p):.3f}, {max(p):.3f}]")
    json.dump({"thresholds": {"t_model": best[1], "t_family": best[2], "dev_score": best[0], "n_dev": len(dev)},
               "test": rows}, open(os.path.join(a.test, "norm_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
