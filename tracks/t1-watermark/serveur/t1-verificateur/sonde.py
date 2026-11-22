"""Contrôle des fichiers sonde : chaque sonde = candidat 35 sauf un filigrane, repris du candidat 27. Lecture seule."""
import json, hashlib, torch
from verif import HERE
A = HERE / "t1-assemblage"; tm = json.load(open(HERE / "release" / "test" / "metadata.json")); meth = {m["image_name"]: m["method"] for m in tm}; order = [m["image_name"] for m in tm]
L = lambda n: torch.load(A / n, weights_only=True); s35 = L("submission_candidat_35.pt"); s27 = L("submission_candidat_27.pt")
d35 = dict(zip(s35["image_names"], s35["masks"])); d27 = dict(zip(s27["image_names"], s27["masks"]))
for tag, me in (("ps27", "pixelseal"), ("tm27", "trustmark"), ("ss27", "stegastamp")):
    p = A / f"sonde_35_{tag}.pt"; s = L(p.name); M = s["masks"]; names = list(s["image_names"]); d = dict(zip(names, M))
    ok_fmt = list(s.keys()) == ["image_names", "masks"] and tuple(M.shape) == (1020, 128, 128) and M.dtype == torch.uint8 and sorted(torch.unique(M).tolist()) == [0, 1] and len(set(names)) == 1020 and set(names) == set(order)
    eq27 = sum(bool((d[n] == d27[n]).all()) for n in names if meth[n] == me); eq35 = sum(bool((d[n] == d35[n]).all()) for n in names if meth[n] != me); diff35 = sum(not bool((d[n] == d35[n]).all()) for n in names); wrong = sum(not bool((d[n] == d35[n]).all()) for n in names if meth[n] != me)
    a0 = sum(not bool(m.any()) for m in M); a1 = sum(bool(m.all()) for m in M)
    print(f"sonde_35_{tag}.pt sha256 {hashlib.sha256(open(p, 'rb').read()).hexdigest()} ; {p.stat().st_size} octets ; format conforme {ok_fmt} ; ordre du test {names == order} ; {me} égal au 27 : {eq27}/340 ; autres filigranes égaux au 35 : {eq35}/680 ; images différentes du 35 : {diff35} (hors filigrane annoncé : {wrong}) ; tout-0 {a0}, tout-1 {a1}")
