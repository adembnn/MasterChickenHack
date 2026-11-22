"""Fichiers sonde : le candidat 35 dont UN filigrane est remplace par celui du fichier 27 envoye (pour savoir lequel n'a pas transfere au public)."""
import sys, torch
sys.path.insert(0, "."); from t1lib import *
a = torch.load(ME / "submission_candidat_35.pt", weights_only=True); b = torch.load(ME / "submission_candidat_27.pt", weights_only=True)
order = [r["image_name"] for r in load_test()]; meth = {r["image_name"]: r["method"] for r in load_test()}; assert a["image_names"] == order
ib = {n: i for i, n in enumerate(b["image_names"])}; assert set(ib) == set(order)
for m, tag in (("pixelseal", "ps"), ("trustmark", "tm"), ("stegastamp", "ss")):
    M = a["masks"].clone(); k = [i for i, n in enumerate(order) if meth[n] == m]
    for i in k: M[i] = b["masks"][ib[order[i]]].to(M.dtype)
    out = ME / f"sonde_35_{tag}27.pt"; torch.save({"image_names": list(order), "masks": M.to(torch.uint8)}, out)
    c = torch.load(out, weights_only=True); diff = (c["masks"] != a["masks"]).flatten(1).any(1); other = [i for i in range(1020) if meth[order[i]] != m]
    print(f"{out.name} : {len(k)} masques {m} pris au 27 | images differentes du 35 : {int(diff.sum())} (toutes {m} : {bool(all(meth[order[i]] == m for i in torch.where(diff)[0].tolist()))}) | autres filigranes identiques au 35 : {bool((c['masks'][other] == a['masks'][other]).all())} | {m} identique au 27 : {bool(all((c['masks'][i] == b['masks'][ib[order[i]]]).all() for i in k))}")
