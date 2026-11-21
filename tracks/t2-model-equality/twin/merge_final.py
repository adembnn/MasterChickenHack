#!/usr/bin/env python
"""CSV final du test 1 = familles de run_ep.sh (assign) + étiquettes de modèle de la session fp8 (ep_readers.py).
Étiquettes acceptées : JSON {"ep-01": "label", ...} (ou {"labels": {...}}), ou CSV endpoint,family,model / endpoint,model.
Le modèle final est préfixé par la famille (règle du validateur : FiXXX) ; deux endpoints de familles différentes ne
peuvent donc jamais partager un modèle. Endpoint sans étiquette -> repli (CSV de la règle pré-enregistrée).
Usage : python merge_final.py --ep_dir ep_bloc0 --labels labels_fp8.json --out submission_final.csv"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from family_table import SHORT


def read_labels(path):
    p = Path(path)
    if p.suffix == ".json":
        d = json.load(open(p))
        d = d.get("labels", d)
        return {e: str(v) for e, v in d.items() if e.startswith("ep-")}
    rows = list(csv.reader(open(p)))
    if rows and rows[0][0] == "endpoint":
        rows = rows[1:]
    return {r[0].strip(): r[-1].strip() for r in rows if r and r[0].startswith("ep-")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep_dir", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="submission_final.csv")
    a = ap.parse_args()
    d = Path(a.ep_dir)
    fam = json.load(open(d / "assign.json"))["family_twin"]
    fbp = d / "submission_ep_fallback.csv"
    fb = {r["endpoint"]: r["model"] for r in csv.DictReader(open(fbp))} if fbp.exists() else {}
    lab = read_labels(a.labels)
    # étiquette « base » de chaque famille côté fp8 (ex. "qwen3b_base"), pour qu'un endpoint sans étiquette la rejoigne
    base_of = {}
    for e, l in lab.items():
        if l.endswith("_base") and e in fam:
            base_of.setdefault(fam[e], l)
    out, src = {}, {}
    for e in sorted(fam):
        f = SHORT.get(fam[e].split("/")[-1], fam[e].split("/")[-1].lower()).replace(".", "")
        if e in lab:
            out[e] = (f, f + "_" + lab[e].replace(",", "_").replace(" ", "_"))
            src[e] = "fp8"
        else:
            out[e] = (f, f + ("_repli_" + fb[e] if e in fb else "_" + base_of.get(fam[e], "base")))
            src[e] = "repli"
    with open(a.out, "w") as fh:
        fh.write("endpoint,family,model\n")
        for e in sorted(out):
            fh.write(f"{e},{out[e][0]},{out[e][1]}\n")
    print(f"{len(out)} endpoints ; étiquettes fp8 : {sum(v == 'fp8' for v in src.values())} ; repli : "
          f"{[e for e, v in src.items() if v == 'repli']}")
    for f in sorted({v[0] for v in out.values()}):
        groups = {}
        for e, (ff, m) in out.items():
            if ff == f:
                groups.setdefault(m, []).append(e)
        print(f"  {f} : " + " | ".join(f"{m}: {','.join(es)}" for m, es in sorted(groups.items())))
    print("-> " + a.out)
    sys.exit(subprocess.run([sys.executable, str(Path(__file__).with_name("validate_t2.py")), a.out]).returncode)


if __name__ == "__main__":
    main()
