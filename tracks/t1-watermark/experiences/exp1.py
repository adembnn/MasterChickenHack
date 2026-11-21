"""Expérience 1 (T1) : quelle carte fine localise le mieux ? Mesuré sur le train, par filigrane.
Variantes de carte (toutes ramenées à 128x128) :
  cam        : vote par position du décodeur (8x8), agrandi
  mask{W}    : on ne garde qu'une fenêtre WxW (reste = couleur moyenne), on décode, marge vs message connu
  crop{W}    : on rogne la fenêtre et on la décode seule
  occl{W}    : on efface la fenêtre, on mesure la chute de marge
  grad       : gradient de la marge par rapport aux pixels (pas StegaStamp)
Plafonds : masque vrai dégradé en blocs de k pixels (dit quelle finesse il faut).
Sorties : exp1_<methode>.json (scores) et maps_<methode>.npz (cartes, pour entraîner un réseau ensuite)."""
import sys, json, argparse, time
from pathlib import Path
import os; os.environ.setdefault('OMP_NUM_THREADS','8')
import numpy as np, torch, torch.nn.functional as F
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "kit"))
from wmloc.watermarks import load_watermark
from wmloc.data import LocalisationDataset
from wmloc.utils import downsample_mask, miou_image, to_tensor

ap = argparse.ArgumentParser(); ap.add_argument("--method", required=True); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--split", default="train"); a = ap.parse_args()
dev = "cuda"; method = a.method; torch.set_num_threads(8)
ds = LocalisationDataset(HERE / "release", a.split, methods=[method])
n = len(ds) if not a.limit else min(a.limit, len(ds))
wm = load_watermark(method, models_dir=HERE / "kit" / "models", device=dev)
torch_model = method != "stegastamp"

def raw_logits(x):                                   # avec gradient possible
    if method == "pixelseal": return wm.det(wm._resize(x.to(dev)))[:, 1:].float()
    if method == "trustmark": return wm.net(wm._prep(x)).float()
    return wm.logits(x)

def margins(batch, sgn, bs=256):
    out_m, out_a = [], []
    for i in range(0, len(batch), bs):
        with torch.no_grad():
            lg = wm.logits(batch[i:i + bs]).cpu()
        out_m.append((lg * sgn).mean(1)); out_a.append(((lg > 0).float() * 2 - 1 == sgn).float().mean(1))
    return torch.cat(out_m), torch.cat(out_a)

def window_scores(xd, mc, sgn, grid, win, kind, bs=96):
    """xd : image 1x3x512x512 sur GPU ; construit les lots sur GPU, par paquets."""
    ms, accs = [], []
    for i in range(0, len(grid), bs):
        g = grid[i:i + bs]
        if kind == "crop":
            b = torch.stack([xd[0, :, y:y + win, x:x + win] for y, x in g])
        else:
            b = (mc if kind == "mask" else xd).expand(len(g), 3, 512, 512).clone()
            for j, (y, x) in enumerate(g):
                b[j, :, y:y + win, x:x + win] = xd[0, :, y:y + win, x:x + win] if kind == "mask" else mc[0]
        m, a_ = margins(b, sgn, bs=bs); ms.append(m); accs.append(a_)
    return torch.cat(ms), torch.cat(accs)

def win_grid(win, stride):
    r = list(range(0, 512 - win + 1, stride)); return [(y, x) for y in r for x in r]

def splat(vals, grid, win):                          # moyenne des fenêtres qui couvrent chaque cellule
    acc = np.zeros((128, 128), np.float32); cnt = np.zeros((128, 128), np.float32)
    for v, (y, x) in zip(vals, grid):
        acc[y // 4:(y + win) // 4, x // 4:(x + win) // 4] += float(v); cnt[y // 4:(y + win) // 4, x // 4:(x + win) // 4] += 1
    return acc / np.maximum(cnt, 1)

WINS = [(128, 32), (64, 32)] if method == "stegastamp" else [(192, 32), (128, 32), (96, 32), (64, 32)]
maps = {}; gts = []; names = []; fullwm = []; ceil = {k: [] for k in (8, 16, 32, 64)}
t0 = time.time()
for i in range(n):
    it = ds[i]; names.append(it["image_name"])
    x = to_tensor(it["image"]); sgn = torch.tensor(it["message"]).float() * 2 - 1
    mean_col = x.mean(dim=(2, 3), keepdim=True)
    if "mask128" in it:
        g = it["mask128"]; gts.append(g); fullwm.append(bool(it.get("full_watermark", False)))
        m = it["mask"].astype(np.float32)
        for k in ceil:
            blk = (m.reshape(512 // k, k, 512 // k, k).mean(axis=(1, 3)) >= 0.5).astype(np.uint8)
            ceil[k].append(miou_image(downsample_mask(np.kron(blk, np.ones((k, k), np.uint8))), g))
    def put(key, arr): maps.setdefault(key, []).append(arr.astype(np.float16))
    full_m, _ = margins(x, sgn)
    if torch_model:
        with torch.no_grad():
            pp = wm.features(x)["per_position_logits"].float().cpu()[0]
        v = (sgn[:, None, None] * pp).mean(0)[None, None]
        put("cam", F.interpolate(v, size=(128, 128), mode="bilinear", align_corners=False)[0, 0].numpy())
        xg = x.clone().to(dev).requires_grad_(True)
        (raw_logits(xg) * sgn.to(dev)).mean().backward()
        gmap = xg.grad.abs().sum(1, keepdim=True)
        put("grad", F.avg_pool2d(F.avg_pool2d(gmap, 4), 5, 1, 2)[0, 0].cpu().numpy())
        gx = (xg.grad * (x.to(dev) - mean_col.to(dev))).sum(1, keepdim=True)          # gradient x entrée
        put("gradxin", F.avg_pool2d(F.avg_pool2d(gx, 4), 5, 1, 2)[0, 0].cpu().numpy())
    xd = x.to(dev); mc = mean_col.to(dev)
    for win, stride in WINS:
        grid = win_grid(win, stride)
        mm, ma = window_scores(xd, mc, sgn, grid, win, "mask"); put(f"mask{win}", splat(mm, grid, win)); put(f"maskacc{win}", splat(ma, grid, win))
        cm, ca = window_scores(xd, mc, sgn, grid, win, "crop"); put(f"crop{win}", splat(cm, grid, win)); put(f"cropacc{win}", splat(ca, grid, win))
        om, _ = window_scores(xd, mc, sgn, grid, win, "occl"); put(f"occl{win}", splat(float(full_m) - om, grid, win))
    if i % 20 == 0: print(method, i, round(time.time() - t0), "s", flush=True)

maps = {k: np.stack(v) for k, v in maps.items()}
np.savez_compressed(HERE / "adem" / f"maps_{a.split}_{method}.npz", names=np.array(names), **maps)
res = {"n": n, "sec": round(time.time() - t0)}
if gts:
    gts = np.stack(gts); fullwm = np.array(fullwm)
    def otsu(v):
        h, e = np.histogram(v, 64); p = h / h.sum(); w = np.cumsum(p); mu = np.cumsum(p * (e[:-1] + e[1:]) / 2)
        s = (mu[-1] * w - mu) ** 2 / np.maximum(w * (1 - w), 1e-9); return e[np.argmax(s) + 1]
    def evaluate(M):
        M = M.astype(np.float32); ths = np.quantile(M, np.linspace(0.02, 0.98, 49)); best = (-1, None, None)
        for t in ths:
            sc = np.array([miou_image(M[j] > t, gts[j]) for j in range(len(M))])
            if sc.mean() > best[0]: best = (float(sc.mean()), float(t), sc)
        ot = float(np.mean([miou_image(M[j] > otsu(M[j]), gts[j]) for j in range(len(M))]))
        return {"best": round(best[0], 4), "th": best[1], "otsu": round(ot, 4),
                "best_fullwm": round(float(best[2][fullwm].mean()), 4), "best_partial": round(float(best[2][~fullwm].mean()), 4)}
    res["maps"] = {k: evaluate(v) for k, v in maps.items()}
    # moyenne simple de cartes centrées-réduites (fusion naïve)
    def z(M): M = M.astype(np.float32); return (M - M.mean()) / (M.std() + 1e-6)
    keys = [k for k in maps if k.startswith(("mask1", "mask9", "mask6", "crop1", "occl1"))]
    res["maps"]["fusion_z"] = evaluate(sum(z(maps[k]) for k in keys) / len(keys)); res["fusion_keys"] = keys
    res["plafond_blocs_px"] = {k: round(float(np.mean(v)), 4) for k, v in ceil.items()}
(HERE / "adem" / f"exp1_{a.split}_{method}.json").write_text(json.dumps(res, indent=1))
print("RESULT", method, json.dumps(res), flush=True)
