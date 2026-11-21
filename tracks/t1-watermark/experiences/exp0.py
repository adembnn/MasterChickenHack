"""Expérience 0 (T1) : où est le signal ? Mesures sur le train, par filigrane.
- stats des masques (part filigranée, rectangle plein ou troué, full_watermark)
- bases : tout-0, tout-1, boîte englobante vraie (oracle),
- précision de bits : image entière / rognage de la boîte vraie / rognage hors boîte
- carte brute du décodeur (per_position_logits) seuillée : mIoU
Usage : python exp0.py [--limit N]"""
import sys, json, argparse, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "kit"))
from wmloc.watermarks import load_watermark
from wmloc.data import LocalisationDataset
from wmloc.utils import downsample_mask, miou_image, to_tensor

ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
out = {}
for method in ("pixelseal", "trustmark", "stegastamp"):
    t0 = time.time()
    ds = LocalisationDataset(HERE / "release", "train", methods=[method])
    n = len(ds) if not a.limit else min(a.limit, len(ds))
    wm = load_watermark(method, models_dir=HERE / "kit" / "models", device=dev)
    R = {k: [] for k in ("frac", "fill", "fullwm", "all0", "all1", "bbox", "acc_full", "acc_box", "acc_out", "cam", "cam_box")}
    for i in range(n):
        it = ds[i]; img, m, g = it["image"], it["mask"], it["mask128"]
        msg = torch.tensor(it["message"]).float()
        sgn = (2 * msg - 1)
        R["frac"].append(m.mean()); R["fullwm"].append(float(bool(it.get("full_watermark", False))))
        R["all0"].append(miou_image(np.zeros_like(g), g)); R["all1"].append(miou_image(np.ones_like(g), g))
        x = to_tensor(img)
        def acc(t):
            return float(((wm.logits(t).cpu()[0] > 0).float() == msg).float().mean())
        R["acc_full"].append(acc(x))
        ys, xs = np.where(m > 0)
        if len(ys) == 0:
            R["fill"].append(np.nan); R["bbox"].append(miou_image(np.zeros_like(g), g)); R["acc_box"].append(np.nan); R["acc_out"].append(np.nan)
            y0 = x0 = 0; y1 = x1 = 512
        else:
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            R["fill"].append(m[y0:y1, x0:x1].mean())
            bb = np.zeros_like(m); bb[y0:y1, x0:x1] = 1
            R["bbox"].append(miou_image(downsample_mask(bb), g))
            R["acc_box"].append(acc(x[:, :, y0:y1, x0:x1]))
            # plus grande bande hors boîte
            bands = [(0, y0, 0, 512), (y1, 512, 0, 512), (0, 512, 0, x0), (0, 512, x1, 512)]
            b = max(bands, key=lambda q: (q[1] - q[0]) * (q[3] - q[2]))
            R["acc_out"].append(acc(x[:, :, b[0]:b[1], b[2]:b[3]]) if (b[1] - b[0]) >= 32 and (b[3] - b[2]) >= 32 else np.nan)
        if method != "stegastamp":
            def cam(t, size):
                pp = wm.features(t)["per_position_logits"].float().cpu()[0]          # nbits x h x w
                v = (sgn[:, None, None] * pp).mean(0)[None, None]
                return F.interpolate(v, size=size, mode="bilinear", align_corners=False)[0, 0].numpy()
            v = cam(x, (512, 512))
            R["cam"].append(miou_image(downsample_mask((v > 0).astype(np.uint8)), g))
            vb = np.full((512, 512), -1.0, np.float32)
            vb[y0:y1, x0:x1] = cam(x[:, :, y0:y1, x0:x1], (y1 - y0, x1 - x0))
            R["cam_box"].append(miou_image(downsample_mask((vb > 0).astype(np.uint8)), g))
    s = {k: (float(np.nanmean(v)) if len(v) else None) for k, v in R.items()}
    s["n"] = n; s["sec"] = round(time.time() - t0, 1)
    s["fill_lt_095"] = float(np.nanmean(np.array(R["fill"]) < 0.95))
    out[method] = s
    print(method, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}), flush=True)
    del wm; torch.cuda.empty_cache()
(HERE / "exp0_result.json").write_text(json.dumps(out, indent=1))
