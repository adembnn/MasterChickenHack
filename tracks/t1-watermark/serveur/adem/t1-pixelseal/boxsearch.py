"""Recherche de rectangle par décodage (sans entraînement). Pour une image partielle : on essaie des centaines de boîtes, on rogne chacune,
on la remet à 256 et on demande au décodeur PixelSeal d'origine à quel point elle relit le message connu ; on garde la meilleure boîte,
puis on ajuste ses quatre bords. Mesuré sur les vraies partielles (rectangles et objets)."""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "8")
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
HERE = Path(os.environ.get("WMLOC_ROOT", "/p/scratch/training2666/destors1/MasterChickenHack/Hackathon_Watermark_Localisation_Task"))
sys.path.insert(0, str(HERE / "kit"))
from wmloc.watermarks import load_watermark
from wmloc.utils import miou_image, downsample_mask
dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"; split = sys.argv[2] if len(sys.argv) > 2 else "train"; t0 = time.time()
OUT = HERE / "adem" / "t1-pixelseal"
wm = load_watermark("pixelseal", models_dir=HERE / "kit" / "models", device=dev); det = wm.det
meta = [(i, m) for i, m in enumerate(json.loads((HERE / "release" / split / "metadata.json").read_text())) if m["method"] == "pixelseal"]
SZ = [96, 144, 208, 304, 432, 512]
CAND = []
for h in SZ:
    for w in SZ:
        if max(h, w) / min(h, w) > 6.5: continue
        st = max(32, min(h, w) // 2)
        ys = sorted(set(list(range(0, 512 - h + 1, st)) + [512 - h])); xs = sorted(set(list(range(0, 512 - w + 1, st)) + [512 - w]))
        CAND += [(y, x, h, w) for y in ys for x in xs]
CAND = torch.tensor(CAND, dtype=torch.float32, device=dev); print("boîtes candidates", len(CAND), flush=True)
@torch.no_grad()
def score(x, sgn, boxes, bs=192):                    # x 1x3x512x512 ; boxes n x 4 (y, x, h, w) ; renvoie marge et bits justes
    ms, accs = [], []
    for k in range(0, len(boxes), bs):
        b = boxes[k:k + bs]; n = len(b); th = torch.zeros(n, 2, 3, device=dev)
        th[:, 0, 0] = b[:, 3] / 512; th[:, 0, 2] = (b[:, 1] + b[:, 3] / 2) / 256 - 1; th[:, 1, 1] = b[:, 2] / 512; th[:, 1, 2] = (b[:, 0] + b[:, 2] / 2) / 256 - 1
        crop = F.grid_sample(x.expand(n, -1, -1, -1), F.affine_grid(th, (n, 3, 256, 256), align_corners=False), mode="bilinear", padding_mode="border", align_corners=False)
        with torch.autocast("cuda", dtype=torch.float16): lg = det(crop)[:, 1:].float()
        ms.append((torch.tanh(lg) * sgn).mean(1)); accs.append(((lg > 0).float() * 2 - 1 == sgn).float().mean(1))
    return torch.cat(ms), torch.cat(accs)
def search(x, sgn):
    m, a_ = score(x, sgn, CAND); j = int(m.argmax()); best = CAND[j].clone(); bm = float(m[j]); ba = float(a_[j])
    for d in (32, 16, 8, 4, 2):
        for _ in range(3):
            y, xx, h, w = best.tolist(); y1, x1 = y + h, xx + w; c = []
            for dy0, dx0, dy1, dx1 in ((-d, 0, 0, 0), (d, 0, 0, 0), (0, -d, 0, 0), (0, d, 0, 0), (0, 0, -d, 0), (0, 0, d, 0), (0, 0, 0, -d), (0, 0, 0, d),
                                       (-d, 0, -d, 0), (d, 0, d, 0), (0, -d, 0, -d), (0, d, 0, d)):
                ny0, nx0, ny1, nx1 = max(0, y + dy0), max(0, xx + dx0), min(512, y1 + dy1), min(512, x1 + dx1)
                if ny1 - ny0 >= 16 and nx1 - nx0 >= 16: c.append((ny0, nx0, ny1 - ny0, nx1 - nx0))
            c = torch.tensor(c, dtype=torch.float32, device=dev); m, a_ = score(x, sgn, c); j = int(m.argmax())
            if float(m[j]) > bm + 1e-4: best = c[j].clone(); bm = float(m[j]); ba = float(a_[j])
            else: break
    return [int(v) for v in best.tolist()], bm, ba
rows = []
for n, (i, m) in enumerate(meta):
    x = torch.from_numpy(np.asarray(Image.open(HERE / "release" / split / "images" / f"{m['image_name']}.png").convert("RGB"))).permute(2, 0, 1)[None].float().div(255).to(dev)
    sgn = torch.tensor([int(c) for c in m["message"]], device=dev).float() * 2 - 1
    with torch.no_grad(): ga = float(((wm.logits(x)[0] > 0).float() * 2 - 1 == sgn).float().mean())
    if ga > 0.75 and split != "train": rows.append({"i": i, "name": m["image_name"], "glob": ga}); continue          # pleine : pas de recherche
    if split == "train" and m.get("full_watermark"): rows.append({"i": i, "name": m["image_name"], "glob": ga}); continue
    box, bm, ba = search(x, sgn); rows.append({"i": i, "name": m["image_name"], "glob": ga, "box": box, "marge": bm, "bits": ba})
    if len(rows) % 20 == 0: print(len(rows), round(time.time() - t0), "s", flush=True)
(OUT / f"box_{split}.json").write_text(json.dumps(rows)); print("temps", round(time.time() - t0), "s")
if split == "train":
    z = np.load(HERE / "release" / "train" / "masks_128.npz"); G = dict(zip(z["names"].tolist(), z["masks"]))
    R = [r for r in rows if "box" in r]; rect, iou, bits, sc_box, sc_zero, res = [], [], [], [], [], []
    for r in R:
        mk = np.asarray(Image.open(HERE / "release" / "train" / "masks" / f"{r['name']}.png")) > 127; ys, xs = np.where(mk)
        rect.append(mk.sum() / ((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)) >= 0.98)
        y, x, h, w = r["box"]; p = np.zeros((512, 512), bool); p[y:y + h, x:x + w] = True; iou.append((p & mk).sum() / (p | mk).sum()); bits.append(r["bits"])
        g = G[r["name"]]; sc_box.append(miou_image(downsample_mask(p), g)); sc_zero.append(miou_image(np.zeros((128, 128)), g)); res.append(r["i"] % 5 == 0)
    rect, iou, bits, sc_box, sc_zero, res = map(np.array, (rect, iou, bits, sc_box, sc_zero, res))
    print("rectangles réels : IoU boîte trouvée / vraie, rangé par bits justes de la boîte trouvée")
    for lo, hi in ((0, .56), (.56, .6), (.6, .65), (.65, .75), (.75, 1.01)):
        for nm, sel in (("rect", rect), ("objet", ~rect)):
            s = sel & (bits >= lo) & (bits < hi)
            if s.sum(): print(f"  bits {lo:.2f}-{hi:.2f} {nm:5s} n={s.sum():2d} IoU moyen {iou[s].mean():.3f} | mIoU si on rend la boîte {sc_box[s].mean():.3f} contre tout-à-0 {sc_zero[s].mean():.3f}")
    print("règle : rendre la boîte si bits >= tau, sinon tout à 0 ; tau réglé hors réserve")
    for tau in (0.56, 0.58, 0.60, 0.62, 0.65, 0.70):
        sc = np.where(bits >= tau, sc_box, sc_zero)
        print(f"  tau {tau:.2f} : hors réserve rect {sc[rect & ~res].mean():.4f} objets {sc[~rect & ~res].mean():.4f} toutes {sc[~res].mean():.4f} | RÉSERVE partielles {sc[res].mean():.4f} (tout-à-0 : {sc_zero[res].mean():.4f} ; hors réserve tout-à-0 {sc_zero[~res].mean():.4f})")
