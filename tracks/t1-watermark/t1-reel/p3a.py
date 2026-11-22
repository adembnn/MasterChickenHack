"""P3a (critique PixelSeal) : IoU de la zone filigranée sur des rectangles FABRIQUÉS, par petit côté du rectangle, avec les 5 modèles affinés d'un tag (moyenne des logits, image + image retournée).
Réserve : ces rectangles viennent de la réserve d'exemples d'entraînement (vus ~8 fois chacun) : le chiffre est un majorant.
usage : p3a.py <filigrane> <tag affiné> <suffixe de la réserve d'exemples> [nombre par tranche]"""
import sys, json, numpy as np, torch, argparse
me, tag, suf = sys.argv[1:4]; N = int(sys.argv[4]) if len(sys.argv) > 4 else 150; sys.argv = sys.argv[:1]
import reel10 as R9
ROOT = R9.ROOT; D = ROOT / f"data_{me}{suf}"; RECT = np.load(D / "RECT.npy", mmap_mode="r"); RBOX = np.load(D / "RBOX.npy"); rng = np.random.default_rng(0)
a = argparse.Namespace(**json.loads((ROOT / "runs" / tag / me / "fold0" / "args.json").read_text()))
for k, v in dict(npl=0, pl="", hsmall=0.0, hlo=0.005, hhi=0.04, cf=0.0).items(): setattr(a, k, getattr(a, k, v))
R = R9.Runner(a, "cuda"); nets = []
for f in range(5):
    R_ = R9.Runner(a, "cuda"); R_.net.load_state_dict(torch.load(ROOT / "runs" / tag / me / f"fold{f}" / "ema.pt", map_location="cuda")["state"]); R_.net.eval(); nets.append(R_)
small = np.minimum(RBOX[:, 2], RBOX[:, 3]); ratio = np.maximum(RBOX[:, 2], RBOX[:, 3]) / small; area = RBOX[:, 2] * RBOX[:, 3] / 512 ** 2
print(f"{me} {tag} réserve {suf} : {len(RBOX)} rectangles ; petit côté < 64 px : {np.mean(small < 64):.2f} ; aire < 3 % : {np.mean(area < .03):.2f} ; rapport >= 8,7 : {np.mean(ratio >= 8.7):.2f}", flush=True)
def run(sel, nom):
    if len(sel) == 0: return
    sel = np.sort(rng.choice(sel, min(N, len(sel)), replace=False)); X = np.stack([RECT[i] for i in sel]); Y = np.zeros((len(sel), 512, 512), np.float32)
    for k, i in enumerate(sel): y0, x0, h, w = RBOX[i]; Y[k, y0:y0 + h, x0:x0 + w] = 1
    Y = Y.reshape(len(sel), 128, 4, 128, 4).mean((2, 4)) >= 0.5
    with torch.no_grad(): L = np.mean([r.predict(r.net, X, [0.0] * len(X), tta=2) for r in nets], 0)
    P = L > 0; iw = np.array([(p & y).sum() / max((p | y).sum(), 1) for p, y in zip(P, Y)]); ic = np.array([((~p) & (~y)).sum() / max(((~p) | (~y)).sum(), 1) for p, y in zip(P, Y)])
    print(f"   {nom:34s} : {len(sel):4d} rectangles | IoU de la zone filigranée {iw.mean():.3f} | jamais vus (IoU < 0,05) {np.mean(iw < .05):.2f} | mIoU {((iw + ic) / 2).mean():.3f}", flush=True)
for lo, hi in ((10, 32), (32, 64), (64, 128), (128, 513)): run(np.where((small >= lo) & (small < hi))[0], f"petit côté {lo} à {hi} px")
run(np.where((ratio >= 8) & (small < 64))[0], "bandes (rapport >= 8, côté < 64)")
run(np.where((ratio < 3) & (small < 64))[0], "compacts (rapport < 3, côté < 64)")
