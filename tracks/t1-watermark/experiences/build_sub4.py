"""Soumission T1 v3 : StegaStamp et TrustMark = moyenne des 5 plis de t1-reel (run mixr) ; PixelSeal = v2 (tronc décodeur)."""
import sys, json, glob, numpy as np, torch
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "kit"))
from wmloc.data import LocalisationDataset
from wmloc.utils import miou_image
train = LocalisationDataset(HERE / "release", "train"); meta = train.meta
gt = {m["image_name"]: train.m128[m["image_name"]] for m in meta}
reserve = {m["image_name"] for i, m in enumerate(meta) if i % 5 == 0}
def avg(files):
    acc = {}
    for f in files:
        z = np.load(f)
        for n, l in zip(z["names"], z["logits"].astype(np.float32)): acc.setdefault(str(n), []).append(l)
    return {n: np.mean(v, 0) for n, v in acc.items()}
rep = {}; pred = {}
for meth in ("stegastamp", "trustmark"):
    d = HERE / "adem" / "t1-reel" / "runs" / "mixr" / meth
    res = avg(sorted(glob.glob(str(d / "fold*" / "res.npz")))); tst = avg(sorted(glob.glob(str(d / "fold*" / "test.npz"))))
    sc = [miou_image(l > 0, gt[n]) for n, l in res.items() if n in reserve]
    rep[meth] = {"source": "t1-reel mixr, 5 plis moyennés", "reserve": round(float(np.mean(sc)), 4), "n_reserve": len(sc), "n_test": len(tst)}
    pred.update({n: l > 0 for n, l in tst.items()})
z = np.load(HERE / "adem" / "t1-pixelseal" / "run_D_res512" / "pred_train.npz")
sc = [miou_image(l > 0, gt[str(n)]) for n, l in zip(z["names"], z["logits"].astype(np.float32)) if str(n) in reserve]
z = np.load(HERE / "adem" / "t1-pixelseal" / "run_D_res512" / "pred_test.npz")
pred.update({str(n): l > 0 for n, l in zip(z["names"], z["logits"].astype(np.float32))})
rep["pixelseal"] = {"source": "t1-pixelseal run_D_res512", "reserve": round(float(np.mean(sc)), 4), "n_reserve": len(sc)}
test = LocalisationDataset(HERE / "release", "test"); names = [m["image_name"] for m in test.meta]
assert len(names) == 1020 and len(set(names)) == 1020 and all(n in pred for n in names), "noms manquants"
masks = torch.from_numpy(np.stack([pred[n] for n in names])).bool()
out = HERE / "adem" / "submission_v4.pt"; torch.save({"image_names": names, "masks": masks}, out)
chk = torch.load(out, weights_only=True)
rep["fichier"] = {"chemin": str(out), "Mo": round(out.stat().st_size / 1e6, 2), "forme": list(chk["masks"].shape), "dtype": str(chk["masks"].dtype), "part_filigranee": round(float(chk["masks"].float().mean()), 3)}
rep["reserve_moyenne"] = round(float(np.mean([rep[m]["reserve"] for m in ("pixelseal", "trustmark", "stegastamp")])), 4)
print(json.dumps(rep, indent=1))
