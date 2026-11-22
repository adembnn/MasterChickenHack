"""Masques SAM (segmentation générique, sans rapport avec le filigrane) pour les images PixelSeal : candidats de contours d'objets.
Sortie : sam_<split>.npz, une clé par image = masques K x 128 x 128 (bool compressé par packbits), règle officielle 512 -> 128."""
import os, sys, json, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
HERE = Path(os.environ["WMLOC_ROOT"]); D = HERE / "adem" / "t1-pixelseal"; split = sys.argv[1]
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
sam = sam_model_registry["vit_h"](checkpoint=str(D / "sam_vit_h_4b8939.pth")).to("cuda").eval()
gen = SamAutomaticMaskGenerator(sam, points_per_side=24, points_per_batch=128, pred_iou_thresh=0.84, stability_score_thresh=0.90, crop_n_layers=0, min_mask_region_area=64)
meta = [m for m in json.loads((HERE / "release" / split / "metadata.json").read_text()) if m["method"] == "pixelseal"]
out = {}; t0 = time.time()
for n, m in enumerate(meta):
    im = np.asarray(Image.open(HERE / "release" / split / "images" / f"{m['image_name']}.png").convert("RGB"))
    with torch.no_grad(): ms = gen.generate(im)
    ms = sorted(ms, key=lambda r: -r["area"])[:160]
    a = np.stack([r["segmentation"].reshape(128, 4, 128, 4).mean((1, 3)) >= 0.5 for r in ms]) if ms else np.zeros((0, 128, 128), bool)
    a = a[a.reshape(len(a), -1).sum(1) >= 4]; out[m["image_name"]] = np.packbits(a, axis=None); out[m["image_name"] + "__n"] = np.array(len(a))
    if n % 25 == 0: print(split, n, len(a), "masques", round(time.time() - t0), "s", flush=True)
np.savez_compressed(D / f"sam_{split}.npz", **out); print("FINI", split, len(meta), round(time.time() - t0), "s")
