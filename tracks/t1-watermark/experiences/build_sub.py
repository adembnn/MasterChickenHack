"""Assemble la soumission T1 à partir des prédictions par filigrane, vérifie le score local, écrit submission.pt."""
import sys, json, numpy as np, torch
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "kit"))
from wmloc.data import LocalisationDataset
from wmloc.utils import miou_image
CHOIX = {"pixelseal": "seg_v2dec_pixelseal", "trustmark": sys.argv[1] if len(sys.argv) > 1 else "seg_v2dec_trustmark", "stegastamp": "seg_v2unet_stegastamp"}
train = LocalisationDataset(HERE / "release", "train"); meta = train.meta
gt = {m["image_name"]: (train.m128[m["image_name"]], i % 5 == 0, m["method"]) for i, m in enumerate(meta)}
pred_test = {}; rep = {}
for meth, d in CHOIX.items():
    z = np.load(HERE / "adem" / d / f"pred_train_{meth}.npz"); sc = [(miou_image(l > 0, gt[n][0]), gt[n][1]) for n, l in zip(z["names"], z["logits"].astype(np.float32))]
    rep[meth] = {"source": d, "train_vu_en_partie": round(float(np.mean([s for s, _ in sc])), 4), "reserve_34": round(float(np.mean([s for s, r in sc if r])), 4)}
    z = np.load(HERE / "adem" / d / f"pred_test_{meth}.npz")
    for n, l in zip(z["names"], z["logits"].astype(np.float32)): pred_test[str(n)] = (l > 0)
test = LocalisationDataset(HERE / "release", "test"); names = [m["image_name"] for m in test.meta]
assert len(names) == 1020 and len(set(names)) == 1020 and all(n in pred_test for n in names)
masks = torch.from_numpy(np.stack([pred_test[n] for n in names])).bool()
out = HERE / "adem" / "submission_v2.pt"; torch.save({"image_names": names, "masks": masks}, out)
chk = torch.load(out, weights_only=True)
rep["fichier"] = {"chemin": str(out), "Mo": round(out.stat().st_size / 1e6, 2), "forme": list(chk["masks"].shape), "dtype": str(chk["masks"].dtype), "part_filigranee": round(float(chk["masks"].float().mean()), 3)}
rep["reserve_moyenne_3_filigranes"] = round(float(np.mean([rep[m]["reserve_34"] for m in CHOIX])), 4)
print(json.dumps(rep, indent=1))
