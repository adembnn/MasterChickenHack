"""Test X1 du critique (tour 3) : le trou des pleines est-il limité par la force du signal ? Sur un jeu fabriqué FIXE (mêmes fonds, mêmes trous réels des autres
filigranes), on multiplie le résidu du filigrane par alpha et on mesure l'IoU du trou avec le même réseau (g_s1, 8000 pas). Si le score grimpe vite avec alpha,
c'est bien le rapport signal sur bruit qui limite ; s'il reste bas même à alpha = 2, c'est la règle de décision du réseau."""
import sys, json, argparse, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); import reel6 as R
method, tag, suf, N = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]); dev = "cuda"; rng = np.random.default_rng(5)
trm, tem = R.metas(); MT = np.load(R.CACHE / "MASK_train.npy", mmap_mode="r"); D = R.ROOT / f"data_{method}{suf}"; BASE = np.load(D / "BASE.npy", mmap_mode="r"); WFULL = np.load(D / "WFULL.npy", mmap_mode="r")
ck = torch.load(R.ROOT / "runs" / tag / method / "fold-1" / "ema.pt", map_location=dev); a = argparse.Namespace(**ck["args"]); a.pretrained = 0
for k_, v_ in (("s1", 0), ("st", 0), ("echan", 0), ("hp", 0), ("flag", 0), ("stem1", 0)): setattr(a, k_, getattr(a, k_, v_))
run = R.Runner(a, dev); run.net.load_state_dict(ck["state"])
holes = [~np.array(MT[i]).astype(bool) for i, m in enumerate(trm) if m["method"] != method and m["full_watermark"] and i % 5 != 0]
bi = rng.choice(len(BASE), N, replace=False); H = [R.dih_np(holes[int(rng.integers(len(holes)))], int(rng.integers(8))) for _ in range(N)]
B = np.stack([np.asarray(BASE[int(b)]) for b in bi]).astype(np.float32); W = np.stack([np.asarray(WFULL[int(b)]) for b in bi]).astype(np.float32); Hm = np.stack(H)
G = np.stack([R.downsample_mask((~h).astype(np.uint8)) for h in H]).astype(bool); frac = Hm.mean((1, 2))
def iou0(p, g): u = ((~p) | (~g)).sum(); return 1.0 if u == 0 else ((~p) & (~g)).sum() / u
print(f"## {method} {tag} : {N} pleines fabriquées fixes, résidu moyen {np.abs(W - B).mean():.2f} niveau(x)")
for al in (0.5, 0.7, 1.0, 1.5, 2.0, 3.0):
    X = np.clip(np.round(np.where(Hm[..., None], B, B + al * (W - B))), 0, 255).astype(np.uint8); lg = run.predict(run.net, X, np.ones(N), tta=1, bs=8); v = np.array([iou0(l > 0, g) for l, g in zip(lg, G)])
    print(f"  alpha = {al:<4} : IoU du trou {v.mean():.3f} | trous < 2 % : {v[frac < .02].mean():.3f} | 2-8 % : {v[(frac >= .02) & (frac < .08)].mean():.3f} | > 8 % : {v[frac >= .08].mean():.3f} | part prédite propre {np.mean([(l <= 0).mean() for l in lg]):.3f} (vraie {frac.mean():.3f})")
