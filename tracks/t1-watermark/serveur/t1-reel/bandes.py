"""Question : les petits trous sont-ils aussi mal lus sur le FABRIQUÉ que sur le RÉEL ? IoU du trou par tranche de taille, même modèle (pré-entraîné sur fabriqué seul),
sur le jeu fabriqué fixe (fabval) et sur les vraies pleines hors réserve. Si le fabriqué est aussi bas, l'écart fabriqué/réel n'est pas la cause.
usage : bandes.py <filigrane> <tag pré-entraîné> <suffixe du jeu fabriqué>"""
import sys, json, numpy as np, torch, argparse
from pathlib import Path
sys.argv, me, tag, suf = sys.argv[:1], sys.argv[1], sys.argv[2], sys.argv[3]
import reel9 as R9
ROOT = R9.ROOT; CACHE = R9.CACHE; a = argparse.Namespace(**json.loads((ROOT / "runs" / tag / me / "fold-1" / "args.json").read_text()))
for k, v in dict(npl=0, pl="", hsmall=0.0, hlo=0.005, hhi=0.04).items(): setattr(a, k, getattr(a, k, v))
R = R9.Runner(a, "cuda"); net = R.net; net.load_state_dict(torch.load(ROOT / "runs" / tag / me / "fold-1" / "ema.pt", map_location="cuda")["state"]); net.eval()
B = [(0, .01), (.01, .02), (.02, .04), (.04, .08), (.08, .2), (.2, 1.01)]
def table(nom, L, Y):
    Y = Y.astype(bool); P = L > 0; fr = 1 - Y.reshape(len(Y), -1).mean(1); print(f"## {nom} : {len(Y)} pleines, part propre médiane {np.median(fr):.3f}, moins de 2 % : {np.mean(fr < .02):.2f}")
    for lo, hi in B:
        s = (fr >= lo) & (fr < hi)
        if not s.sum(): continue
        ic = np.array([((~p) & (~y)).sum() / max(((~p) | (~y)).sum(), 1) for p, y in zip(P[s], Y[s])]); iw = np.array([(p & y).sum() / max((p | y).sum(), 1) for p, y in zip(P[s], Y[s])])
        print(f"   trou {lo*100:4.1f} à {hi*100:5.1f} % : {int(s.sum()):4d} images | IoU du trou {ic.mean():.3f} | part des trous jamais vus (IoU < 0,05) {np.mean(ic < .05):.2f} | mIoU {((ic + iw) / 2).mean():.3f}")
z = np.load(ROOT / f"fabval_{me}{suf}.npz"); SX, SY, SF = z["X"], z["Y"], z["F"]; full = SY.reshape(len(SY), -1).mean(1) > 0.5
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16): Lf = R.predict(net, SX[full], SF[full], tta=2)
table(f"FABRIQUÉ ({me}, {tag})", Lf, SY[full] > 0.5)
trm, tem = R9.metas(); RT = np.load(CACHE / "REAL_train.npy", mmap_mode="r"); Y128 = np.load(CACHE / "Y128_train.npy")
idx = [i for i, m in enumerate(trm) if m["method"] == me and m["full_watermark"] and i % 5 != 0]
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16): Lr = R.predict(net, np.stack([RT[i] for i in idx]), [0.0] * len(idx), tta=2)
table(f"RÉEL hors réserve ({me}, {tag})", Lr, Y128[idx] > 0.5)
