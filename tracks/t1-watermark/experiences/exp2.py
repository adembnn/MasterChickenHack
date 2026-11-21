"""Expérience 2 (T1) : filtre adapté. On compare la haute fréquence de l'image au motif que le tampon poserait
pour NOTRE message (gabarit), et à K messages au hasard (témoins). Carte de gain g = <h, r_m> / <r_m, r_m> (lissés).
Sorties : exp2_<methode>.json, maps2_train_<methode>.npz (gain, z)."""
import os, sys, json, argparse, time
os.environ.setdefault("OMP_NUM_THREADS", "8")
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "kit"))
from wmloc.watermarks import load_watermark
from wmloc.data import LocalisationDataset
from wmloc.utils import miou_image

ap = argparse.ArgumentParser(); ap.add_argument("--method", required=True); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--split", default="train"); ap.add_argument("--K", type=int, default=6); a = ap.parse_args()
torch.set_num_threads(8); dev = "cuda"; method = a.method; rng = np.random.default_rng(0)
ds = LocalisationDataset(HERE / "release", a.split, methods=[method]); n = len(ds) if not a.limit else min(a.limit, len(ds))
wm = load_watermark(method, models_dir=HERE / "kit" / "models", device=dev)

def gauss(x, sigma):
    k = int(4 * sigma) | 1; t = torch.arange(k, device=x.device) - k // 2; w = torch.exp(-t ** 2 / (2 * sigma ** 2)); w = (w / w.sum()).float()
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), w.view(1, 1, 1, k).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), w.view(1, 1, k, 1).repeat(c, 1, 1, 1), groups=c)

def template(img, bits):                               # motif que le tampon ajouterait (float, H x W x 3 -> 1x3xHxW)
    out = wm.embed(img, bits).astype(np.float32) - img.astype(np.float32)
    return torch.from_numpy(out).permute(2, 0, 1)[None].to(dev)

SIG_H = {"pixelseal": 1.5, "trustmark": 1.5, "stegastamp": 3.0}[method]
maps = {k: [] for k in ("gain8", "gain16", "z8", "z16", "energy")}; gts = []; fullwm = []; names = []; t0 = time.time()
for i in range(n):
    it = ds[i]; img = it["image"]; names.append(it["image_name"])
    x = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)[None].to(dev)
    h = x - gauss(x, SIG_H)
    r = template(img, it["message"]); rs = [template(img, rng.integers(0, 2, wm.nbits)) for _ in range(a.K)]
    for sg in (8, 16):
        S = gauss((h * r).sum(1, keepdim=True), sg); E = gauss((r * r).sum(1, keepdim=True), sg) + 1e-3
        Sk = torch.cat([gauss((h * q).sum(1, keepdim=True), sg) for q in rs])
        g = S / E; z = (S - Sk.mean(0, keepdim=True)) / (Sk.std(0, keepdim=True) + 1e-3)
        maps[f"gain{sg}"].append(F.avg_pool2d(g, 4)[0, 0].cpu().numpy().astype(np.float16)); maps[f"z{sg}"].append(F.avg_pool2d(z, 4)[0, 0].cpu().numpy().astype(np.float16))
        if sg == 16: maps["energy"].append(F.avg_pool2d(E, 4)[0, 0].cpu().numpy().astype(np.float16))
    if "mask128" in it: gts.append(it["mask128"]); fullwm.append(bool(it.get("full_watermark", False)))
    if i % 40 == 0: print(method, i, round(time.time() - t0), "s", flush=True)
maps = {k: np.stack(v) for k, v in maps.items()}
np.savez_compressed(HERE / "adem" / f"maps2_{a.split}_{method}.npz", names=np.array(names), **maps)
res = {"n": n, "sec": round(time.time() - t0)}
if gts:
    gts = np.stack(gts); fullwm = np.array(fullwm)
    def evaluate(M, fixed=None):
        M = M.astype(np.float32); best = (-1, None, None)
        for t in ([fixed] if fixed is not None else np.quantile(M, np.linspace(0.02, 0.98, 49))):
            sc = np.array([miou_image(M[j] > t, gts[j]) for j in range(len(M))])
            if sc.mean() > best[0]: best = (float(sc.mean()), float(t), sc)
        return {"score": round(best[0], 4), "seuil": round(best[1], 4), "pleines": round(float(best[2][fullwm].mean()), 4), "partielles": round(float(best[2][~fullwm].mean()), 4)}
    res["maps"] = {k: evaluate(v) for k, v in maps.items() if k != "energy"}
    res["maps"]["gain16_seuil0.5"] = evaluate(maps["gain16"], 0.5); res["maps"]["gain8_seuil0.5"] = evaluate(maps["gain8"], 0.5)
    # valeurs typiques du gain dedans / dehors (médianes), par population
    g = maps["gain16"].astype(np.float32); inm = gts.astype(bool)
    res["gain_median"] = {"dedans_pleines": float(np.median(g[fullwm][inm[fullwm]])), "dehors_pleines": float(np.median(g[fullwm][~inm[fullwm]])),
                          "dedans_partielles": float(np.median(g[~fullwm][inm[~fullwm]])), "dehors_partielles": float(np.median(g[~fullwm][~inm[~fullwm]]))}
(HERE / "adem" / f"exp2_{a.split}_{method}.json").write_text(json.dumps(res, indent=1))
print("RESULT", method, json.dumps(res), flush=True)
