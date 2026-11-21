#!/usr/bin/env python
"""CSV minimal pour entrer au classement : familles de `assign` (NLL minimale), un seul modèle par famille.
Ne vise que le terme famille (10 %) ; le terme modèle vaut 0 (aucune paire « différent » dans une famille).
Usage : python families_only_csv.py --assign ep_bloc0/assign.json --out submission_familles.csv"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from family_table import SHORT

ap = argparse.ArgumentParser()
ap.add_argument("--assign", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
fam = json.load(open(a.assign))["family_twin"]
with open(a.out, "w") as f:
    f.write("endpoint,family,model\n")
    for e in sorted(fam):
        s = SHORT.get(fam[e].split("/")[-1], fam[e].split("/")[-1].lower()).replace(".", "")
        f.write(f"{e},{s},{s}_base\n")
print("-> " + a.out)
sys.exit(subprocess.run([sys.executable, str(Path(__file__).with_name("validate_t2.py")), a.out]).returncode)
