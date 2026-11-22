"""PixelSeal, rectangles : la lecture du message par le décodeur sur la boîte (z = écart au hasard en écarts-types : (bits justes − 0,5) × 32 pour
256 bits) suffit-elle à reconnaître un rectangle bien en dessous du seuil actuel (z >= 12,8, soit 90 % des bits) ?
(a) z à la vraie boîte et z maximal en la décalant de quelques pixels ; (b) niveau de bruit : z maximal sur 600 boîtes tirées hors de la vraie zone ;
(c) recherche amorcée par la CARTE (pas par la vérité) : boîtes autour de la tache de la carte, meilleure boîte, son z et son recouvrement avec la vérité."""
import sys, json
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from scipy import ndimage as ndi
from verif import HERE, miou_vec, load_truth
sys.path.insert(0, str(HERE / "kit")); from wmloc.watermarks import load_watermark
TAG = sys.argv[1] if len(sys.argv) > 1 else "ps_pop_x2+x2b+x1+dx__st+x2"; wm = load_watermark("pixelseal", models_dir=HERE / "kit" / "models", device="cuda"); rng = np.random.default_rng(0)
meta, g = load_truth(check_png=False); idx = {m["image_name"]: i for i, m in enumerate(meta)}; zz = np.load(HERE / "t1-reel" / TAG / "pred_train_pixelseal.npz"); LG = {str(a): l.astype(np.float32) for a, l in zip(zz["names"], zz["logits"])}
def zbox(x, boxes, sg):                                        # x : 1x3x512x512 sur GPU ; boîtes (y0, x0, y1, x1) en pixels
    out = []
    for i in range(0, len(boxes), 256):
        b = torch.cat([F.interpolate(x[:, :, y0:y1, x0:x1], size=(256, 256), mode="bilinear", align_corners=False, antialias=True) for y0, x0, y1, x1 in boxes[i:i + 256]])
        with torch.no_grad(): lg = wm.logits(b)
        out.append((((lg > 0).float() * 2 - 1) * sg).mean(1) * 16)                                   # (bits justes − 0,5) × 32
    return torch.cat(out).cpu().numpy()
def bm(b):
    m = np.zeros((512, 512), bool); m[b[0]:b[2], b[1]:b[3]] = True; return m.reshape(128, 4, 128, 4).mean((1, 3)) >= 0.5
def clip(b): y0, x0, y1, x1 = b; y0 = max(0, min(y0, 500)); x0 = max(0, min(x0, 500)); return (y0, x0, max(y0 + 8, min(y1, 512)), max(x0 + 8, min(x1, 512)))
rows = []
for m in meta:
    if m["method"] != "pixelseal" or m["full_watermark"]: continue
    G = g[m["image_name"]]; ys, xs = np.where(G); y0, y1, x0, x1 = ys.min() * 4, (ys.max() + 1) * 4, xs.min() * 4, (xs.max() + 1) * 4; isrect = G.sum() >= 0.95 * (y1 - y0) * (x1 - x0) / 16
    x = torch.from_numpy(np.asarray(Image.open(HERE / "release" / "train" / "images" / f"{m['image_name']}.png").convert("RGB"))).permute(2, 0, 1)[None].float().div(255).cuda(); sg = torch.tensor([1.0 if c == "1" else -1.0 for c in m["message"]], device="cuda")
    # (b) bruit : 600 boîtes de tailles variées hors de la vraie zone
    nb = []
    while len(nb) < 600:
        h = int(np.exp(rng.uniform(np.log(16), np.log(300)))); w = int(np.exp(rng.uniform(np.log(16), np.log(300)))); a = int(rng.integers(0, 512 - h)); c = int(rng.integers(0, 512 - w))
        if not G[a // 4:(a + h) // 4 + 1, c // 4:(c + w) // 4 + 1].any(): nb.append((a, c, a + h, c + w))
        elif len(nb) == 0 and rng.random() < 0.002: break
    znoise = zbox(x, nb, sg).max() if nb else np.nan
    # (a) vraie boîte et petits décalages
    jit = [clip((y0 + a, x0 + c, y1 + e, x1 + f_)) for a in (-4, 0, 4) for c in (-4, 0, 4) for e in (-4, 0, 4) for f_ in (-4, 0, 4)]; zt = zbox(x, [(y0, x0, y1, x1)], sg)[0]; ztj = zbox(x, jit, sg).max()
    # (c) recherche amorcée par la carte : tache = composante du maximum de la carte lissée, au seuil max − 2
    v = ndi.gaussian_filter(LG[m["image_name"]], 1); lab, _ = ndi.label(v >= v.max() - 2); bl = lab == lab[np.unravel_index(v.argmax(), v.shape)]; by, bx = np.where(bl); cy, cx = (by.min() + by.max() + 1) * 2, (bx.min() + bx.max() + 1) * 2; bh, bw = (by.max() - by.min() + 1) * 4, (bx.max() - bx.min() + 1) * 4
    cand = []
    for sh in (0.5, 0.75, 1, 1.5, 2, 3, 4):
        for sw in (0.5, 0.75, 1, 1.5, 2, 3, 4):
            for dy in (-1, -0.5, 0, 0.5, 1):
                for dx in (-1, -0.5, 0, 0.5, 1):
                    h = max(12, bh * sh); w = max(12, bw * sw); yy = cy + dy * h * 0.5; xx = cx + dx * w * 0.5; cand.append(clip((int(yy - h / 2), int(xx - w / 2), int(yy + h / 2), int(xx + w / 2))))
    cand = list(dict.fromkeys(cand)); zc = zbox(x, cand, sg); best = cand[int(zc.argmax())]; io = float((bm(best) & G).sum() / max((bm(best) | G).sum(), 1))
    rows.append((m["image_name"], isrect, min(y1 - y0, x1 - x0), zt, ztj, znoise, float(zc.max()), io, float(miou_vec(bm(best)[None], G[None])[0]), float(miou_vec((LG[m["image_name"]] > 0)[None], G[None])[0]), idx[m["image_name"]] % 5 == 0))
R = np.array([r[1:] for r in rows], dtype=float); rect = R[:, 0] == 1; ms = R[:, 1]
print(f"PixelSeal partielles : {len(R)} images, dont {int(rect.sum())} rectangles. z = (bits justes − 0,5) × 32 ; seuil actuel des boîtes : z >= 12,8")
print(f"  bruit (z maximal de 600 boîtes hors de la vraie zone), toutes partielles : médiane {np.nanmedian(R[:, 4]):.1f}, 95e centile {np.nanpercentile(R[:, 4], 95):.1f}, max {np.nanmax(R[:, 4]):.1f}")
for lo, hi in ((0, 32), (32, 64), (64, 128), (128, 513)):
    k = rect & (ms >= lo) & (ms < hi)
    if k.sum(): print(f"  rectangles, petit côté {lo}-{hi} px (n={k.sum()}) : z à la vraie boîte médiane {np.median(R[k, 2]):.1f} ; avec décalage de ±4 px {np.median(R[k, 3]):.1f} ; z >= 6 pour {np.sum(R[k, 3] >= 6)}, z >= 8 pour {np.sum(R[k, 3] >= 8)}, z >= 12,8 pour {np.sum(R[k, 3] >= 12.8)} | recherche amorcée par la carte : z max médian {np.median(R[k, 5]):.1f}, z >= 6 pour {np.sum(R[k, 5] >= 6)}, dont bonne boîte (IoU >= 0,5) {np.sum((R[k, 5] >= 6) & (R[k, 6] >= .5))}")
print("  recherche amorcée par la carte, selon le seuil z d'acceptation (toutes les partielles ; sinon on garde la carte au seuil 0) :")
for z0 in (5, 6, 7, 8, 10, 12.8):
    acc = R[:, 5] >= z0; s = np.where(acc, R[:, 7], R[:, 8]); print(f"     z >= {z0:4.1f} : boîtes rendues {int(acc.sum()):2d} (rectangles {int((acc & rect).sum())}, objets {int((acc & ~rect).sum())}) ; IoU boîte / vérité des rendues : médiane {np.median(R[acc, 6]) if acc.any() else float('nan'):.2f} ; mIoU des 85 partielles {s.mean():.4f} (carte seule {R[:, 8].mean():.4f}) ; rectangles {s[rect].mean():.4f} (carte seule {R[rect, 8].mean():.4f}) ; objets {s[~rect].mean():.4f}")
json.dump([dict(name=r[0], rect=bool(r[1]), petit_cote=int(r[2]), z_vraie=float(r[3]), z_vraie_decal=float(r[4]), z_bruit=float(r[5]), z_carte=float(r[6]), iou=float(r[7])) for r in rows], open("boitez.json", "w"))
