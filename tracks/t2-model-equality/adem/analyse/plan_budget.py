#!/usr/bin/env python
"""Après la ronde 0 : dit quels endpoints sont déterministes et lesquels échantillonnent (stdlib, aucune requête).

Un endpoint échantillonne si, pour au moins une sonde de la ronde 0, ses deux réponses au même message diffèrent.
Écrit endpoints_deterministes.txt et endpoints_echantillonnes.txt (un endpoint par ligne) et affiche le reste à payer.
Usage : python plan_budget.py --answers answers.jsonl [--plan ../probes/plan.json] [--prefix ep-]
"""
import argparse, json, os
from collections import defaultdict


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", nargs="+", required=True)
    ap.add_argument("--plan", default=os.path.join(here, "..", "probes", "plan.json"))
    ap.add_argument("--prefix", default="")
    a = ap.parse_args()
    plan = json.load(open(a.plan))
    reps = defaultdict(lambda: defaultdict(dict))
    for p in a.answers:
        for l in open(p, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                if r["endpoint"].startswith(a.prefix):
                    reps[r["endpoint"]][str(r["id"])][int(r.get("repeat", 0))] = r["response"].strip()
    det, samp, missing = [], [], []
    for ep in sorted(reps):
        seen = [reps[ep][p] for p in plan["ronde0"] if len(reps[ep].get(p, {})) >= 2]
        if len(seen) < len(plan["ronde0"]):
            missing.append(ep)
        (samp if any(len(set(d.values())) > 1 for d in seen) else det).append(ep)
    open("endpoints_deterministes.txt", "w").write("\n".join(det) + ("\n" if det else ""))
    open("endpoints_echantillonnes.txt", "w").write("\n".join(samp) + ("\n" if samp else ""))
    print(f"{len(det)} déterministes : {' '.join(det)}")
    print(f"{len(samp)} échantillonnés : {' '.join(samp)}")
    if samp and not det:
        print("TOUS les endpoints échantillonnent : utiliser le plan « tout échantillonné » = tout_echantillonne_x10.jsonl "
              "x10 (86 nouvelles réponses) + tout_echantillonne_x2.jsonl x2 (6) pour chacun, à la place du plan ci-dessous.")
    if missing:
        print("ATTENTION, ronde 0 incomplète (moins de 2 réponses sur une sonde) :", " ".join(missing))
    print("reste à payer : déterministe = deterministe.jsonl x1 (92) ; échantillonné = echantillonne_x2.jsonl x2 (32) "
          "+ echantillonne_x4.jsonl x4 (60 nouvelles, 4 déjà payées en ronde 0)")


if __name__ == "__main__":
    main()
