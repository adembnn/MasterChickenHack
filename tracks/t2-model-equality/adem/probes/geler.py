#!/usr/bin/env python
"""Gèle la liste de sondes et le plan de dépense des 100 réponses par endpoint.

Plan (par endpoint, 100 réponses exactement) :
  ronde 0        : 4 sondes fermées x 2 répétitions = 8 réponses. Si deux réponses au même message diffèrent,
                   l'endpoint échantillonne ; sinon il est déterministe.
  déterministe   : 92 autres sondes x 1 = 92  (16 lp, 12 meta, 14 ab5, 18 q1, 12 nb5, 12 word, 8 open)
  échantillonné  : 16 sondes libres x 2 (12 meta, 4 open) = 32, pour le test à deux échantillons sur les textes ;
                   16 sondes fermées x 4 (6 ab5 dont les 2 de la ronde 0, 4 nb5, 6 lp) = 64 dont 4 déjà payées = 60.
                   Total 8 + 32 + 60 = 100.
Fichiers écrits (format accepté par task_template.py : .jsonl avec id et prompt) :
  probes_gelees.jsonl, plan.json, ronde0.jsonl (REPEATS=2), deterministe.jsonl (REPEATS=1),
  echantillonne_x2.jsonl (REPEATS=2), echantillonne_x4.jsonl (REPEATS=4 ; task_template saute ce qui est déjà fait)
Usage : python geler.py probes_v3.jsonl probes_meta2.jsonl
"""
import json, sys, os
from collections import defaultdict

DET = {"lp": 16, "meta": 12, "ab5": 16, "q1": 20, "nb5": 12, "word": 12, "open": 8}
SAMP4 = {"ab5": 6, "nb5": 4, "lp": 6}
SAMP2 = {"meta": 12, "open": 4}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    rows = [json.loads(l) for f in sys.argv[1:] for l in open(f, encoding="utf-8") if l.strip()]
    by = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["id"]):
        by[r["kind"]].append(r)
    keep = {k: by[k][:n] for k, n in DET.items()}
    assert all(len(keep[k]) == n for k, n in DET.items()), {k: len(v) for k, v in keep.items()}
    ronde0 = [r["id"] for r in keep["ab5"][:2] + keep["q1"][:2]]
    allp = [r for k in DET for r in keep[k]]
    det = [r["id"] for r in allp if r["id"] not in ronde0]
    x2 = [r["id"] for k, n in SAMP2.items() for r in keep[k][:n]]
    x4 = [r["id"] for k, n in SAMP4.items() for r in keep[k][:n]]
    samp = {p: 2 for p in ronde0}
    samp.update({p: 2 for p in x2})
    samp.update({p: 4 for p in x4})
    assert len(det) == 92 and sum(samp.values()) == 100, sum(samp.values())
    byid = {r["id"]: r for r in allp}

    def dump(name, ids):
        with open(os.path.join(here, name), "w", encoding="utf-8") as f:
            for i in ids:
                r = byid[i]
                f.write(json.dumps({"id": r["id"], "kind": r["kind"], "prompt": r["prompt"]}, ensure_ascii=False) + "\n")

    dump("probes_gelees.jsonl", [r["id"] for r in allp])
    dump("ronde0.jsonl", ronde0)
    dump("deterministe.jsonl", det)
    dump("echantillonne_x2.jsonl", x2)
    dump("echantillonne_x4.jsonl", x4)
    for old in ("echantillonne_x1.jsonl", "echantillonne_x3.jsonl"):
        if os.path.exists(os.path.join(here, old)):
            os.remove(os.path.join(here, old))
    json.dump({"ronde0": ronde0, "deterministe": det, "echantillonne": samp},
              open(os.path.join(here, "plan.json"), "w"), indent=1)
    print(f"{len(allp)} sondes gelées ; ronde 0 = {ronde0} ; déterministe 8 + {len(det)} = 100 ; "
          f"échantillonné : {sum(samp.values())} réponses ({len(x2)} libres x 2, {len(x4)} fermées x 4)")


if __name__ == "__main__":
    main()
