"""Inférence zoomée (mesure du vérificateur, sans réentraînement) sur les modèles existants d'un run à plis de t1-reel.
Pour chaque image : logits normaux L0 ; Z2 = tuiles de 256 px (pas 128) agrandies à 512, logits ramenés à la grille ; Z4 = tuiles de 128 px (pas 64).
Recouvrements : moyenne. Images hors-pli : modèle du pli qui ne les a pas vues ; réserve : moyenne des 5 modèles (comme au test).
Usage : python zoom.py <methode> <tag>   -> écrit zoom_<methode>_<tag>.npz (names, grp, L0, Z2, Z4)"""
import sys, json, argparse, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
HERE = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(HERE / "t1-reel")); import reel3 as R
me, tag = sys.argv[1], sys.argv[2]; dev = "cuda"; torch.set_num_threads(4)
trm, tem = R.metas(); RT = np.load(R.CACHE / "REAL_train.npy", mmap_mode="r"); idx, res, rest, fold = R.splits(trm, me)

@torch.no_grad()
def tiles(run, X, size, stride, bs=48):
    """X : B x 3 x 512 x 512 (0..255). Rend B x 128 x 128 : moyenne des logits des tuiles agrandies à 512."""
    B = X.shape[0]; acc = torch.zeros(B, 128, 128, device=dev); cnt = torch.zeros(128, 128, device=dev); pos = list(range(0, 512 - size + 1, stride)); c = size // 4; k = 512 // size
    for y in pos:
        for x in pos:
            t = F.interpolate(X[:, :, y:y + size, x:x + size], size=(512, 512), mode="bilinear", align_corners=False)
            lg = torch.cat([run.fwd(run.net, t[j:j + bs], None) for j in range(0, B, bs)])             # B x 128 x 128 sur la tuile
            lg = F.avg_pool2d(lg[:, None], k)[:, 0]                                                    # ramené à c x c cases de l'image
            acc[:, y // 4:y // 4 + c, x // 4:x // 4 + c] += lg; cnt[y // 4:y // 4 + c, x // 4:x // 4 + c] += 1
    return (acc / cnt).cpu().numpy()

out = {}; t0 = time.time()
for f in range(5):
    p = R.ROOT / "runs" / tag / me / f"fold{f}" / "ema.pt"
    if not p.exists(): print("pli manquant", f); continue
    ck = torch.load(p, map_location=dev); args = argparse.Namespace(**ck["args"]); args.pretrained = 0
    assert not args.flag, "modèle à drapeau : non géré ici"
    run = R.Runner(args, dev); run.net.load_state_dict(ck["state"]); run.net.eval()
    va = [i for i in rest if fold[i] == f]; ids = va + list(res)
    X = torch.from_numpy(np.ascontiguousarray(RT[ids])).to(dev).permute(0, 3, 1, 2).float()
    L0 = run.predict(run.net, np.ascontiguousarray(RT[ids]), np.zeros(len(ids), np.float32), tta=2); Z2 = tiles(run, X, 256, 128); Z4 = tiles(run, X, 128, 64)
    # contrôle : mes logits normaux = ceux sauvés à l'entraînement
    z = np.load(p.with_name("oof.npz")); d = np.abs(L0[:len(va)] - z["logits"].astype(np.float32)).max() if list(z["idx"]) == va else float("nan")
    print(f"pli {f} : {len(va)} hors-pli + {len(res)} réserve, écart max de mes logits normaux aux logits sauvés {d:.3f}, {round(time.time() - t0)} s", flush=True)
    for j, i in enumerate(ids):
        n = trm[i]["image_name"]; o = out.setdefault(n, {"grp": "res" if i % 5 == 0 else "oof", "L0": 0, "Z2": 0, "Z4": 0, "k": 0})
        o["L0"] = o["L0"] + L0[j]; o["Z2"] = o["Z2"] + Z2[j]; o["Z4"] = o["Z4"] + Z4[j]; o["k"] += 1
    del run; torch.cuda.empty_cache()
names = sorted(out)
np.savez_compressed(Path(__file__).with_name(f"zoom_{me}_{tag}.npz"), names=np.array(names), grp=np.array([out[n]["grp"] for n in names]), nmod=np.array([out[n]["k"] for n in names]),
                    **{k: np.stack([out[n][k] / out[n]["k"] for n in names]).astype(np.float16) for k in ("L0", "Z2", "Z4")})
print("écrit", f"zoom_{me}_{tag}.npz", len(names), "images")
