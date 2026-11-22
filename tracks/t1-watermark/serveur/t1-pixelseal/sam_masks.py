"""Masques SAM (segmentation générique) des images classées PARTIELLES (bits justes de l'image entière <= 0,80) d'un filigrane.
Usage : python sam_masks.py <split> <methode> [...autres couples split methode]
Sortie : sam_<split>_<methode>.npz ; par image, clé <nom> = masques K x 128 x 128 (bool, compressés par packbits) et <nom>__n = K."""
import os, sys, json, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
HERE = Path(os.environ["WMLOC_ROOT"]); D = Path(os.environ.get("SAM_DIR", HERE / "adem" / "t1-pixelseal"))
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
sam = sam_model_registry["vit_h"](checkpoint=str(D / "sam_vit_h_4b8939.pth")).to("cuda").eval()
gen = SamAutomaticMaskGenerator(sam, points_per_side=24, points_per_batch=int(os.environ.get("SAM_PPB", 64)), pred_iou_thresh=0.84, stability_score_thresh=0.90, crop_n_layers=0, min_mask_region_area=64)
jobs = list(zip(sys.argv[1::2], sys.argv[2::2])); SI, SN = (int(v) for v in os.environ.get("SHARD", "0/1").split("/")); SUF = f".s{SI}of{SN}" if SN > 1 else ""
for split, method in jobs:
    f = np.load(D / f"feat_{split}.npz"); acc = dict(zip(f["names"].tolist(), f["acc"].tolist()))
    meta = [m for m in json.loads((HERE / "release" / split / "metadata.json").read_text()) if m["method"] == method and acc[m["image_name"]] <= 0.80]
    meta = meta[SI::SN]; out = {}; t0 = time.time()
    for n, m in enumerate(meta):
        im = np.asarray(Image.open(HERE / "release" / split / "images" / f"{m['image_name']}.png").convert("RGB"))
        with torch.no_grad(): ms = gen.generate(im)
        ms = sorted(ms, key=lambda r: -r["area"])[:160]
        a = np.stack([r["segmentation"].reshape(128, 4, 128, 4).mean((1, 3)) >= 0.5 for r in ms]) if ms else np.zeros((0, 128, 128), bool)
        a = a[a.reshape(len(a), -1).sum(1) >= 4]; out[m["image_name"]] = np.packbits(a, axis=None); out[m["image_name"] + "__n"] = np.array(len(a))
        if n % 25 == 0:
            print(split, method, n, "/", len(meta), len(a), "masques", round(time.time() - t0), "s", flush=True)
            if n: np.savez_compressed(D / f"sam_{split}_{method}{SUF}.part.npz", **out)      # sauvegarde partielle
    np.savez_compressed(D / f"sam_{split}_{method}{SUF}.npz", **out); print("FINI", split, method, SUF, len(meta), round(time.time() - t0), "s", flush=True)
