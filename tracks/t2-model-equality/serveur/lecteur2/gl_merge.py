"""Fusionne les sorties de gl_reader.py lancees en parallele sur des sous-ensembles de directions. Usage : gl_merge.py sortie.json part0.json part1.json ..."""
import sys, json
out, parts = sys.argv[1], sys.argv[2:]
R = json.load(open(parts[0]))
for p in parts[1:]:
    Q = json.load(open(p))
    R["dir"].update(Q.get("dir", {})); R["tau"].update(Q.get("tau", {})); R["self"].update(Q.get("self", {}))
    for sc, v in Q.get("fp8", {}).items(): R["fp8"].setdefault(sc, v)
json.dump(R, open(out, "w")); print("fusion :", len(R["dir"]), "directions,", len(R["base"]), "profils de base,", list(R["fp8"]))
