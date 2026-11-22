"""Même mesure que p5.py (vignette 64 px avec un vrai rectangle contre vignette propre de la même image), mais avec les cartes hors pli d'un segmenteur :
score d'une vignette = logit maximal (et moyen) de ses 16 x 16 cases. Dit si le détecteur local de P5 voit quelque chose que le segmenteur ne voit pas.
usage : p5cmp.py <filigrane> <dossier de prédictions> [...]"""
import sys, json, numpy as np
from pathlib import Path
from PIL import Image
REL = Path(r"E:/EAC-lab/t1/release/train"); meta = json.loads((REL / "metadata.json").read_text()); me = sys.argv[1]; rng = np.random.default_rng(0); W = 16
def auc(s, t):
    s = np.asarray(s); t = np.asarray(t); o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1); n1 = t.sum(); n0 = len(t) - n1; return (r[t == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)
real = []
for i, m in enumerate(meta):
    if m["method"] != me or m.get("full_watermark"): continue
    M = np.array(Image.open(REL / "masks" / (Path(m["image_name"]).stem + ".png")).convert("L")) > 127
    if not M.any(): continue
    ys, xs = np.where(M); h, w = ys.max() + 1 - ys.min(), xs.max() + 1 - xs.min()
    if M.sum() >= 0.98 * h * w: real.append((m["image_name"], ys.min() // 4, xs.min() // 4, max(h // 4, 1), max(w // 4, 1), min(h, w)))
for d in sys.argv[2:]:
    z = np.load(Path(d) / f"pred_train_{me}.npz"); L = dict(zip(z["names"].tolist(), z["logits"].astype(np.float32))); out = []
    for lo, hi in ((0, 32), (32, 64), (64, 128)):
        S = {"max": [], "moy": []}; T = []
        for n, y0, x0, h, w, sm in real:
            if not (lo <= sm < hi): continue
            for _ in range(8):
                cy = int(rng.integers(y0, y0 + h)); cx = int(rng.integers(x0, x0 + w)); py = int(np.clip(cy - W // 2, 0, 128 - W)); px = int(np.clip(cx - W // 2, 0, 128 - W)); p = L[n][py:py + W, px:px + W]; S["max"].append(p.max()); S["moy"].append(p.mean()); T.append(1)
                for _ in range(50):
                    py = int(rng.integers(0, 128 - W + 1)); px = int(rng.integers(0, 128 - W + 1))
                    if py + W <= y0 or py >= y0 + h or px + W <= x0 or px >= x0 + w: p = L[n][py:py + W, px:px + W]; S["max"].append(p.max()); S["moy"].append(p.mean()); T.append(0); break
        out.append(f"{lo}-{hi} px : max {auc(S['max'], T):.3f}, moyenne {auc(S['moy'], T):.3f}")
    print(f"   {d:40s} | " + " | ".join(out))
