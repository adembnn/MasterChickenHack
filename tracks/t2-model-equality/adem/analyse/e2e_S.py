#!/usr/bin/env python
"""Bout en bout en régime « tout échantillonné » : monde worldS3 -> plan tout_echantillonne -> t2_pipeline.py -> barème.

Chaque graine tire, par famille, 9 endpoints de structure identique au corrigé (base en clair, un des 3 prompts système
des organisateurs, persona, filigrane, T 1,0, une quantification, LoRA 0,3 / 0,7 / 1,5), les renomme ep-01..ep-18 et ne
garde que les réponses du plan (7 lots de 10 questions + 2 lots de 5, 10 tirages chacun ; le monde worldS3 n'a ni les
2 questions chiffrées de la ronde 0 ni les sondes d'extraction, soit 90 réponses par endpoint au lieu de 100).
Usage : python e2e_S.py --world ../../worldS3 --seeds 20
"""
import argparse, glob, json, os, random, statistics, sys, tempfile
from collections import defaultdict
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, ".."))
from score_local import score          # noqa: E402
import t2_pipeline                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True)
    ap.add_argument("--fams", default="olmo,qwen")
    ap.add_argument("--seeds", type=int, default=20)
    a = ap.parse_args()
    probes = os.path.join(here, "..", "probes", "probes_gelees.jsonl")
    plan = json.load(open(os.path.join(here, "..", "probes", "plan.json")))["tout_echantillonne"]
    data = defaultdict(lambda: defaultdict(dict))
    for f in glob.glob(os.path.join(a.world, "*_S_*.jsonl")):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            data[r["endpoint"]][r["id"]][r["repeat"]] = r
    res = {"familles estimées": [], "familles vraies": [], "repli (familles estimées)": []}
    per_fam = defaultdict(list)
    for seed in range(a.seeds):
        rng = random.Random(seed)
        chosen = []
        for fam in a.fams.split(","):
            chosen += [f"{fam}-base-plain", f"{fam}-base-{rng.choice(['sysacme', 'syshelp', 'sysspace'])}", f"{fam}-base-persona",
                       f"{fam}-base-wm", f"{fam}-base-t1", f"{fam}-{rng.choice(['fp8', 'int8', 'nf4'])}-plain",
                       f"{fam}-lora0.3-plain", f"{fam}-lora0.7-plain", f"{fam}-lora1.5-plain"]
        chosen = [e for e in chosen if e in data]
        rng.shuffle(chosen)
        d = tempfile.mkdtemp()
        ans = os.path.join(d, "answers.jsonl")
        truth, n = {}, []
        with open(ans, "w", encoding="utf-8") as fo:
            for i, src in enumerate(chosen):
                ep = f"ep-{i + 1:02d}"
                fam, w = src.split("-")[0], src.split("-")[1]
                truth[ep] = (fam, f"{fam}-{w}")
                k = 0
                for pid, reps in plan.items():
                    for rep in range(reps):
                        if pid in data[src] and rep in data[src][pid]:
                            r = dict(data[src][pid][rep])
                            r["endpoint"] = ep
                            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
                            k += 1
                n.append(k)
        rows, repli, diag = t2_pipeline.run([ans], probes)
        res["familles estimées"].append(score({r[0]: (r[1], r[2]) for r in rows}, truth)[:3])
        res["repli (familles estimées)"].append(score({r[0]: (r[1], r[2]) for r in repli}, truth)[:3])
        rows2, _, _ = t2_pipeline.run([ans], probes, families={e: t[0] for e, t in truth.items()})
        res["familles vraies"].append(score({r[0]: (r[1], r[2]) for r in rows2}, truth)[:3])
        for fam in a.fams.split(","):
            sub = {e: t for e, t in truth.items() if t[0] == fam}
            per_fam[fam].append(score({r[0]: (r[1], r[2]) for r in rows2 if r[0] in sub}, sub)[2])
        if seed == 0:
            print(f"{len(chosen)} endpoints, {min(n)} à {max(n)} réponses par endpoint ; diagnostic : {[l for l in diag if l.startswith('!!')][:2]}")
    print(f"{'cas':28s} {'score':>7s} {'[min ; max]':>15s} {'MCC famille':>12s} {'MCC modèle':>11s}")
    for v, r in res.items():
        s = [x[0] for x in r]
        print(f"{v:28s} {statistics.mean(s):7.3f} {'[%.2f ; %.2f]' % (min(s), max(s)):>15s} "
              f"{statistics.mean(x[1] for x in r):12.3f} {statistics.mean(x[2] for x in r):11.3f}")
    print("MCC modèle par famille (familles vraies) : " + "  ".join(f"{f} {statistics.mean(v):.3f}" for f, v in per_fam.items()))


if __name__ == "__main__":
    main()
