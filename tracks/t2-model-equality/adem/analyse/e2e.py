#!/usr/bin/env python
"""Test de bout en bout sur des endpoints simulés : simulate_ep.py -> t2_pipeline.py -> score_local.py.

Pour chaque graine : tire des endpoints dans les mondes locaux, les renomme ep-01..ep-N, n'en garde que les 100
réponses du plan gelé, lance le pipeline et note le CSV au barème officiel (0,1 MCC famille + 0,9 MCC modèle).
Usage : python e2e.py --world ../../world3 --structure ref|mix --n 33 --seeds 20 [--variante principal]
"""
import argparse, os, statistics, subprocess, sys, tempfile
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, ".."))
from score_local import read, score          # noqa: E402
import t2_pipeline                             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True)
    ap.add_argument("--regime", default="G")
    ap.add_argument("--fams", default="olmo,qwen")
    ap.add_argument("--structure", default="mix")
    ap.add_argument("--n", type=int, default=33)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--variantes", default="principal,centre,ancre,prudent")
    ap.add_argument("--oracle_fam", action="store_true", help="impose les vraies familles (isole l'effet du décodeur)")
    ap.add_argument("--keep", default=None, help="dossier où garder les fichiers de la graine 0")
    a = ap.parse_args()
    probes = os.path.join(here, "..", "probes", "probes_gelees.jsonl")
    plan = os.path.join(here, "..", "probes", "plan.json")
    res = {v: [] for v in a.variantes.split(",")}
    res["repli"] = []
    for seed in range(a.seeds):
        d = a.keep if (a.keep and seed == 0) else tempfile.mkdtemp()
        os.makedirs(d, exist_ok=True)
        ans, tr = os.path.join(d, "sim_answers.jsonl"), os.path.join(d, "sim_truth.csv")
        subprocess.run([sys.executable, os.path.join(here, "simulate_ep.py"), "--world", a.world, "--regime", a.regime,
                        "--fams", a.fams, "--n", str(a.n), "--seed", str(seed), "--plan", plan, "--structure",
                        a.structure, "--out", ans, "--truth", tr], check=True, capture_output=True)
        truth = read(tr)
        fams = {e: t[0] for e, t in truth.items()} if a.oracle_fam else None
        for v in a.variantes.split(","):
            rows, repli, diag = t2_pipeline.run([ans], probes, v, families=fams)
            res[v].append(score({r[0]: (r[1], r[2]) for r in rows}, truth)[:3])
            if v == a.variantes.split(",")[0]:
                res["repli"].append(score({r[0]: (r[1], r[2]) for r in repli}, truth)[:3])
                if a.keep and seed == 0:
                    t2_pipeline.write_csv(os.path.join(d, "soumission.csv"), rows)
                    t2_pipeline.write_csv(os.path.join(d, "soumission_repli.csv"), repli)
                    open(os.path.join(d, "soumission_diag.txt"), "w", encoding="utf-8").write("\n".join(diag) + "\n")
    print(f"structure {a.structure}, {a.n if a.structure == 'mix' else '9 par famille'} endpoints, familles "
          f"{'vraies (oracle)' if a.oracle_fam else 'estimées'}, {a.seeds} graines, régime {a.regime}")
    print(f"{'variante':12s} {'score':>7s} {'[min ; max]':>15s} {'MCC famille':>12s} {'MCC modèle':>11s}")
    for v, r in res.items():
        s = [x[0] for x in r]
        print(f"{v:12s} {statistics.mean(s):7.3f} {'[%.2f ; %.2f]' % (min(s), max(s)):>15s} "
              f"{statistics.mean(x[1] for x in r):12.3f} {statistics.mean(x[2] for x in r):11.3f}")


if __name__ == "__main__":
    main()
