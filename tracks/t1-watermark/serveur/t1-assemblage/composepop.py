"""Compose un fichier de soumission par filigrane ET par population predite (sans verite : justesse du decodeur sur l'image entiere, seuil de
asm.populations regle hors reserve). Chaque cle "filigrane/pop" prend les masques de test d'un assemblage (assemblage_<k>.pt ou submission_candidat_<k>.pt).
Usage : python composepop.py 46 pixelseal/pleines=45 pixelseal/partielles=42 trustmark=44 stegastamp=38"""
import sys, numpy as np, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); from asm import *
n = int(sys.argv[1]); src = dict(kv.split("=") for kv in sys.argv[2:])
for m in METHODS:
    if m in src: src[f"{m}/pleines"] = src[f"{m}/partielles"] = src.pop(m)
tr, te, S, notes = discover(); P = populations(tr, te)
pick = lambda k: ME / f"assemblage_{k}.pt" if (ME / f"assemblage_{k}.pt").exists() else ME / f"submission_candidat_{k}.pt"
subs = {k: torch.load(pick(k), weights_only=True) for k in set(src.values())}; idx = {k: {nm: i for i, nm in enumerate(s["image_names"])} for k, s in subs.items()}
pop = {}
for m in METHODS:
    for r, f in zip(te[m], P[m]["te"]): pop[r["image_name"]] = (m, "pleines" if f else "partielles")
order = [r["image_name"] for r in load_test()]; cnt = {}
def get(nm):
    k = src["/".join(pop[nm])]; cnt[("/".join(pop[nm]), k)] = cnt.get(("/".join(pop[nm]), k), 0) + 1; return subs[k]["masks"][idx[k][nm]]
masks = torch.stack([get(nm) for nm in order]).to(torch.uint8); out = ME / f"submission_candidat_{n}.pt"; torch.save({"image_names": order, "masks": masks}, out)
chk = torch.load(out, weights_only=True); assert chk["masks"].dtype == torch.uint8 and tuple(chk["masks"].shape) == (1020, 128, 128) and len(set(chk["image_names"])) == 1020 and int(chk["masks"].max()) <= 1
print("ecrit", out, out.stat().st_size, "octets |", {f"{a} <- {b}": c for (a, b), c in sorted(cnt.items())})
