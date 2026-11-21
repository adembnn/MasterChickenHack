#!/usr/bin/env python3
"""Validateur local T1 (Python standard). Usage :
   python3 validate_t1.py submission.pt [--names test_names.txt | --meta test/metadata.json]
Verifie : extension .pt, taille <= 20 Mo (plafond du PDF ; le README du kit dit 25 Mo : on vise 20), racine dict avec
exactement les cles image_names et masks, masks = tenseur bool/uint8 [N,128,128] valeurs 0/1, N = nombre de noms,
noms str uniques (sans .png), et, si on donne la liste de test, ensemble de noms identique (1020)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ptlite import PtFile

LIMIT = 20 * 1000 * 1000   # on prend 20 Mo decimaux : le cas le plus strict


def main(argv):
    if not argv: sys.exit(__doc__)
    path, errs = argv[0], []
    size = os.path.getsize(path)
    if not path.lower().endswith(".pt"): errs.append("extension != .pt")
    if size > LIMIT: errs.append(f"taille {size:,} o > 20 Mo")
    try:
        f = PtFile(path)
    except Exception as e:
        sys.exit(f"ECHEC lecture : {e}")
    o = f.obj
    if not isinstance(o, dict) or set(o) != {"image_names", "masks"}:
        sys.exit(f"ECHEC : racine {type(o).__name__} cles={list(o) if isinstance(o, dict) else '-'} ; attendu exactement image_names, masks")
    names, m = o["image_names"], o["masks"]
    if not isinstance(names, list) or not all(type(n) is str for n in names): errs.append("image_names doit etre list[str] (pas numpy/Path)")
    else:
        if len(set(names)) != len(names): errs.append(f"noms dupliques : {len(names) - len(set(names))}")
        if any(n.lower().endswith(".png") or "/" in n for n in names): errs.append("noms avec .png ou chemin (attendu : nom sans .png)")
    if not hasattr(m, "storage"): sys.exit("ECHEC : masks n'est pas un tenseur")
    if m.dtype not in ("bool", "uint8"): errs.append(f"dtype masks {m.dtype} (bool/uint8 recommandes ; float = 67 Mo)")
    if m.size[1:] != (128, 128): errs.append(f"forme {m.size} != (N,128,128)")
    if isinstance(names, list) and m.size[0] != len(names): errs.append(f"{m.size[0]} masques pour {len(names)} noms")
    if m.dtype in ("bool", "uint8") and f.storage_bytes(m).strip(b"\x00\x01"): errs.append("valeurs hors {0,1}")
    exp = None
    if "--names" in argv: exp = [l.strip() for l in open(argv[argv.index("--names") + 1]) if l.strip()]
    if "--meta" in argv: exp = [r["image_name"] for r in json.load(open(argv[argv.index("--meta") + 1]))]
    if exp is not None and isinstance(names, list):
        miss, extra = set(exp) - set(names), set(names) - set(exp)
        if miss or extra: errs.append(f"noms: {len(miss)} manquants ex.{sorted(miss)[:3]}, {len(extra)} inconnus ex.{sorted(extra)[:3]}")
        if len(exp) != 1020: print(f"AVERTISSEMENT liste de reference de {len(exp)} noms (attendu 1020)")
    elif exp is None and len(names) != 1020: errs.append(f"{len(names)} images != 1020")
    print(f"{len(names)} noms, masks {m.dtype}{list(m.size)}, {size:,} octets")
    for e in errs: print("ERREUR", e)
    print("RESULTAT:", "INVALIDE" if errs else "VALIDE (format) ; mIoU non verifie ici")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
