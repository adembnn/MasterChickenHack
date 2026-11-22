"""Compose un candidat a partir de plusieurs assemblages : un numero d'assemblage par filigrane (ex. PixelSeal du candidat 6 deja verifie,
TrustMark et StegaStamp du 10). Ecrit submission_candidat_<n>.pt, candidat_<n>_honnete_<filigrane>.npz et re-note sur le train.
Usage : python compose.py 11 pixelseal=6 trustmark=10 stegastamp=10 [ref=5]"""
import sys, numpy as np, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); from t1lib import *
n = int(sys.argv[1]); src = dict(kv.split("=") for kv in sys.argv[2:]); ref = src.pop("ref", None)
test_order = [r["image_name"] for r in load_test()]; meth_te = {r["image_name"]: r["method"] for r in load_test()}
pick = lambda k: ME / f"assemblage_{k}.pt" if (ME / f"assemblage_{k}.pt").exists() else ME / f"submission_candidat_{k}.pt"
subs = {k: torch.load(pick(k), weights_only=True) for k in set(src.values())}; idx = {k: {nm: i for i, nm in enumerate(s["image_names"])} for k, s in subs.items()}
masks = torch.stack([subs[src[meth_te[nm]]]["masks"][idx[src[meth_te[nm]]][nm]] for nm in test_order]).to(torch.uint8)
out = ME / f"submission_candidat_{n}.pt"; torch.save({"image_names": test_order, "masks": masks}, out)
chk = torch.load(out, weights_only=True); assert chk["masks"].dtype == torch.uint8 and tuple(chk["masks"].shape) == (1020, 128, 128) and len(set(chk["image_names"])) == 1020 and int(chk["masks"].max()) <= 1
print("ecrit", out, out.stat().st_size, "octets | sources :", src)
tot = []; totref = []
for m in METHODS:
    z = np.load(ME / f"candidat_{src[m]}_honnete_{m}.npz"); rows = {r["image_name"]: r for r in load_train(m)}; names = z["names"].tolist(); M = z["masks"]; G = np.stack([rows[k]["gt"] for k in names]); rr = [rows[k] for k in names]
    np.savez_compressed(ME / f"candidat_{n}_honnete_{m}.npz", names=z["names"], masks=M, reserve=z["reserve"]); sc = miou(M, G); rep = report(sc, rr); line = f"{m:10s} candidat {src[m]:>2s} : reserve {rep['reserve']:.4f} (pleines {rep['res_pleines']:.4f}, partielles {rep['res_partielles']:.4f}) | hors reserve {rep['hors_reserve']:.4f}"
    tot.append(rep["reserve"])
    if ref:
        zr = np.load(ME / f"candidat_{ref}_honnete_{m}.npz") if (ME / f"candidat_{ref}_honnete_{m}.npz").exists() else None
        if zr is not None:
            scr = miou(zr["masks"], np.stack([rows[k]["gt"] for k in zr["names"].tolist()])); res = np.array([r["reserve"] for r in rr]); d = sc - scr; totref.append(float(scr[res].mean()))
            line += f" | contre candidat {ref} : reserve {d[res].mean():+.4f} ({int((d[res] > 1e-6).sum())} images mieux, {int((d[res] < -1e-6).sum())} pires), hors reserve {d[~res].mean():+.4f}"
    print(line)
print(f"moyenne des trois filigranes, reserve : {np.mean(tot):.4f}" + (f" (reference : {np.mean(totref):.4f}, ecart {np.mean(tot) - np.mean(totref):+.4f})" if totref else ""))
