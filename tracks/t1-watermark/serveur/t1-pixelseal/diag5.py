"""Diagnostic 5 : les vraies pleines PixelSeal ont-elles été filigranées sur une toile plus grande (marge p px, miroir) puis rognées ?
On compare réel et fabriqué (p = 0..24) sur : bits justes, marge du décodeur, accord par case 8x8 au centre et au bord."""
import os, sys, json, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
HERE = Path(os.environ.get("WMLOC_ROOT", "/p/scratch/training2666/destors1/MasterChickenHack/Hackathon_Watermark_Localisation_Task")); sys.path.insert(0, str(HERE / "kit"))
from wmloc.watermarks import load_watermark
from wmloc.utils import to_tensor
dev = "cuda"; wm = load_watermark("pixelseal", models_dir=HERE / "kit" / "models", device=dev); rng = np.random.default_rng(0)
meta = json.loads((HERE / "release" / "train" / "metadata.json").read_text()); img = lambda m: np.asarray(Image.open(HERE / "release" / "train" / "images" / f"{m['image_name']}.png").convert("RGB"))
z = np.load(HERE / "release" / "train" / "masks_128.npz"); G = dict(zip(z["names"].tolist(), z["masks"]))
bord = np.ones((8, 8), bool); bord[1:-1, 1:-1] = False
def stats(x, bits):
    sgn = torch.tensor(np.asarray(bits)).float().to(dev) * 2 - 1; t = to_tensor(x).to(dev)
    with torch.no_grad(): lg = wm.logits(t)[0]; pp = wm.features(t)["per_position_logits"][0].float()
    a = (pp * sgn[:, None, None]).mean(0).cpu().numpy()
    return [float(((lg > 0).float() * 2 - 1 == sgn).float().mean()), float((lg * sgn).mean()), float(a[~bord].mean()), float(a[bord].mean()), float((lg * sgn).min())]
real = [stats(img(m), [int(c) for c in m["message"]]) for m in meta if m["method"] == "pixelseal" and m.get("full_watermark") and G[m["image_name"]].mean() > 0.9]
print("colonnes : bits justes | marge moyenne | accord par case au centre | au bord | marge du pire bit")
print(f"RÉEL pleines (>90 % filigrané, n={len(real)}) :", np.round(np.mean(real, 0), 3).tolist(), " écart-type de la marge", round(float(np.std([r[1] for r in real])), 3))
bgs = [img(m) for i, m in enumerate(meta) if m["method"] != "pixelseal" and i % 5 != 0][::6]
for p in (0, 4, 8, 12, 16, 24, 32):
    out = []
    for b in bgs:
        bits = rng.integers(0, 2, 256); c = np.pad(b, ((p, p), (p, p), (0, 0)), mode="reflect") if p else b
        out.append(stats(np.ascontiguousarray(wm.embed(c, bits)[p:p + 512, p:p + 512]), bits))
    print(f"FABRIQUÉ marge {p:2d} px (n={len(out)}) :", np.round(np.mean(out, 0), 3).tolist())
out = []
for b in bgs:                                         # témoin : toile agrandie par zoom (image 536 redimensionnée) au lieu du miroir
    bits = rng.integers(0, 2, 256); c = np.asarray(Image.fromarray(b).resize((536, 536), Image.Resampling.LANCZOS))
    out.append(stats(np.asarray(Image.fromarray(wm.embed(c, bits)).resize((512, 512), Image.Resampling.LANCZOS)), bits))
print("FABRIQUÉ filigrané à 536 puis réduit à 512 :", np.round(np.mean(out, 0), 3).tolist())
