#!/usr/bin/env python
"""Vérifie le mode dégradé de bout en bout avec les SEULES réponses de la ronde 0 (4 sondes x 2 = 8 par endpoint).

Tire 33 endpoints dans un monde local, les renomme ep-01..ep-33, ne garde que la ronde 0, lance
t2_pipeline.py --variante repli et note le CSV au barème officiel. Écrit le CSV de la graine 0 dans --keep.
Usage : python e2e_ronde0.py --world ../../world3 --regime G --seeds 20 [--keep dossier]
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
    ap.add_argument("--regime", default="G")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--keep", default=None)
    a = ap.parse_args()
    probes = os.path.join(here, "..", "probes", "probes_gelees.jsonl")
    ronde0 = json.load(open(os.path.join(here, "..", "probes", "plan.json")))["ronde0"]
    data = defaultdict(lambda: defaultdict(dict))
    for f in glob.glob(os.path.join(a.world, f"*_{a.regime}_*.jsonl")):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            if r["id"] in ronde0 and r["repeat"] < 2:
                data[r["endpoint"]][r["id"]][r["repeat"]] = r
    names = sorted(e for e, d in data.items() if all(len(d.get(p, {})) == 2 for p in ronde0))
    res, nfam = [], []
    for seed in range(a.seeds):
        rng = random.Random(seed)
        chosen = rng.sample(names, 33)
        d = a.keep if (a.keep and seed == 0) else tempfile.mkdtemp()
        os.makedirs(d, exist_ok=True)
        ans = os.path.join(d, "answers.jsonl")
        truth = {}
        with open(ans, "w", encoding="utf-8") as fo:
            for i, src in enumerate(chosen):
                ep = f"ep-{i + 1:02d}"
                truth[ep] = (src.split("-")[0], src)
                for p in ronde0:
                    for k in (0, 1):
                        r = dict(data[src][p][k])
                        r["endpoint"] = ep
                        fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        rows, _, _ = t2_pipeline.run([ans], probes, "repli")
        s = score({r[0]: (r[1], r[2]) for r in rows}, truth)
        res.append(s[:3])
        nfam.append(len(set(r[1] for r in rows)))
        if a.keep and seed == 0:
            t2_pipeline.write_csv(os.path.join(d, "premier.csv"), rows)
    print(f"régime {a.regime}, {len(names)} endpoints disponibles, {a.seeds} tirages de 33, ronde 0 seule (8 réponses par endpoint)")
    print(f"score {statistics.mean(x[0] for x in res):.3f} [{min(x[0] for x in res):.3f} ; {max(x[0] for x in res):.3f}]  "
          f"MCC famille {statistics.mean(x[1] for x in res):.3f} [{min(x[1] for x in res):.3f} ; {max(x[1] for x in res):.3f}]  "
          f"MCC modèle {statistics.mean(x[2] for x in res):.3f}  familles estimées : {min(nfam)} à {max(nfam)} (vraies : 2)")


if __name__ == "__main__":
    main()
