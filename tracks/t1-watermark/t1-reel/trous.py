"""Taille des vrais trous (zone propre des pleines) par filigrane, et IoU du trou par tranche de taille pour une carte livrée (images hors réserve, prédites hors pli)."""
import sys, json, numpy as np
from pathlib import Path
REL = Path(r"E:/EAC-lab/t1/release/train"); meta = json.loads((REL / "metadata.json").read_text()); z = np.load(REL / "masks_128.npz"); Y = dict(zip(z["names"].tolist(), z["masks"].astype(bool)))
P = {"trustmark": "preds/livre/tm_f_x1+f_c5+f_r50", "pixelseal": "preds/livre/ps_f_x1+f_s1+f_deca", "stegastamp": "preds/livre/ss_f_x1+f_c4"}
B = [(0, .005), (.005, .01), (.01, .02), (.02, .04), (.04, .08), (.08, .2), (.2, 1.01)]
for me in ("trustmark", "pixelseal", "stegastamp"):
    fu = [(i, m["image_name"]) for i, m in enumerate(meta) if m["method"] == me and m.get("full_watermark")]
    fr = np.array([1 - Y[n].mean() for _, n in fu]); q = np.percentile(fr, [10, 25, 50, 75, 90])
    print(f"## {me} : {len(fu)} pleines ; part propre (trou) : p10 {q[0]:.3f} p25 {q[1]:.3f} médiane {q[2]:.3f} p75 {q[3]:.3f} p90 {q[4]:.3f} ; moins de 2 % : {np.mean(fr < .02):.2f} ; moins de 4 % : {np.mean(fr < .04):.2f} ; sans trou : {np.mean(fr == 0):.2f}")
    z2 = np.load(Path(P[me]) / f"pred_train_{me}.npz"); L = dict(zip(z2["names"].tolist(), z2["logits"].astype(np.float32)))
    for lo, hi in B:
        hs, ms, n = [], [], 0
        for (i, nm), f in zip(fu, fr):
            if i % 5 == 0 or not (lo <= f < hi): continue
            p = L[nm] > 0; y = Y[nm]; n += 1
            ic = ((~p) & (~y)).sum() / max(((~p) | (~y)).sum(), 1) if ((~p) | (~y)).sum() else 1.0; iw = (p & y).sum() / max((p | y).sum(), 1)
            hs.append(ic); ms.append((ic + iw) / 2)
        if n: print(f"   trou {lo*100:4.1f} à {hi*100:5.1f} % : {n:3d} images | IoU du trou {np.mean(hs):.3f} | mIoU {np.mean(ms):.3f} | perte totale sur la moyenne des pleines {(1 - np.mean(ms)) * n:.1f}")
