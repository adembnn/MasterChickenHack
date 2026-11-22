"""Affinage des rectangles fiables : quel critère donne la boîte la plus proche de la vraie ? (marge brute, marge saturée, variantes)"""
import os, sys, json
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
HERE = Path(os.environ["WMLOC_ROOT"]); sys.path.insert(0, str(HERE / "kit"))
from wmloc.watermarks import load_watermark
dev = "cuda"; wm = load_watermark("pixelseal", models_dir=HERE / "kit" / "models", device=dev); det = wm.det; D = HERE / "adem" / "t1-pixelseal"
meta = {m["image_name"]: m for m in json.loads((HERE / "release" / "train" / "metadata.json").read_text())}
R = [r for r in json.loads((D / "box_train.json").read_text()) if "box" in r and r["bits"] >= 0.85 and r["bits"] - r["glob"] >= 0.35]
@torch.no_grad()
def score(x, sgn, boxes, exact=False):
    b = torch.as_tensor(boxes, dtype=torch.float32, device=dev); n = len(b)
    if exact: crop = torch.cat([F.interpolate(x[:, :, int(y):int(y + h), int(xx):int(xx + w)], size=(256, 256), mode="bilinear", align_corners=False, antialias=True) for y, xx, h, w in b.tolist()])
    else:
        th = torch.zeros(n, 2, 3, device=dev); th[:, 0, 0] = b[:, 3] / 512; th[:, 0, 2] = (b[:, 1] + b[:, 3] / 2) / 256 - 1; th[:, 1, 1] = b[:, 2] / 512; th[:, 1, 2] = (b[:, 0] + b[:, 2] / 2) / 256 - 1
        crop = F.grid_sample(x.expand(n, -1, -1, -1), F.affine_grid(th, (n, 3, 256, 256), align_corners=False), mode="bilinear", padding_mode="border", align_corners=False)
    lg = det(crop)[:, 1:].float(); return (lg * sgn).mean(1), (torch.tanh(lg) * sgn).mean(1), (lg * sgn).min(1).values, (lg * sgn).median(1).values
def refine(x, sgn, box, which, exact):
    best = list(box); bm = float(score(x, sgn, [best], exact)[which])
    for d in (32, 16, 8, 4, 2, 1):
        for _ in range(8):
            y, xx, h, w = best; y1, x1 = y + h, xx + w; c = []
            for dy0, dx0, dy1, dx1 in ((-d, 0, 0, 0), (d, 0, 0, 0), (0, -d, 0, 0), (0, d, 0, 0), (0, 0, -d, 0), (0, 0, d, 0), (0, 0, 0, -d), (0, 0, 0, d)):
                ny0, nx0, ny1, nx1 = max(0, y + dy0), max(0, xx + dx0), min(512, y1 + dy1), min(512, x1 + dx1)
                if ny1 - ny0 >= 16 and nx1 - nx0 >= 16: c.append([ny0, nx0, ny1 - ny0, nx1 - nx0])
            m = score(x, sgn, c, exact)[which]; j = int(m.argmax())
            if float(m[j]) > bm + 1e-5: best = c[j]; bm = float(m[j])
            else: break
    return best, bm
def iou(a, b):
    ya, xa = max(a[0], b[0]), max(a[1], b[1]); yb, xb = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3]); i = max(0, yb - ya) * max(0, xb - xa); return i / (a[2] * a[3] + b[2] * b[3] - i)
res = {k: [] for k in ("départ", "brute", "brute exact", "saturée exact", "pire bit exact", "médiane exact")}
for r in R:
    n = r["name"]; x = torch.from_numpy(np.asarray(Image.open(HERE / "release" / "train" / "images" / f"{n}.png").convert("RGB"))).permute(2, 0, 1)[None].float().div(255).to(dev)
    sgn = torch.tensor([int(c) for c in meta[n]["message"]], device=dev).float() * 2 - 1
    mk = np.asarray(Image.open(HERE / "release" / "train" / "masks" / f"{n}.png")) > 127; ys, xs = np.where(mk); tb = [int(ys.min()), int(xs.min()), int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1)]
    s_true = score(x, sgn, [tb], True); s_found = score(x, sgn, [r["box"]], True)
    line = f"vrai {tb} marge brute {float(s_true[0]):.2f} pire {float(s_true[2]):.2f} | trouvé {r['box']} brute {float(s_found[0]):.2f} pire {float(s_found[2]):.2f} |"
    res["départ"].append(iou(r["box"], tb))
    for nm, which, ex in (("brute", 0, False), ("brute exact", 0, True), ("saturée exact", 1, True), ("pire bit exact", 2, True), ("médiane exact", 3, True)):
        b, m = refine(x, sgn, r["box"], which, ex); res[nm].append(iou(b, tb)); line += f" {nm}: {b} {iou(b, tb):.2f}"
    print(line, flush=True)
print({k: round(float(np.mean(v)), 3) for k, v in res.items()})
