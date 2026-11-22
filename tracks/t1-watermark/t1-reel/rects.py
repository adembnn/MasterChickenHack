"""IoU de la zone filigranée des VRAIS rectangles (partielles dont le masque remplit sa boîte), par petit côté, images hors réserve prédites hors pli.
usage : rects.py <filigrane> <dossier de prédictions> [...]"""
import sys, json, numpy as np
from pathlib import Path
from PIL import Image
REL = Path(r"E:/EAC-lab/t1/release/train"); meta = json.loads((REL / "metadata.json").read_text()); z = np.load(REL / "masks_128.npz"); Y = dict(zip(z["names"].tolist(), z["masks"].astype(bool)))
me = sys.argv[1]; info = []
for i, m in enumerate(meta):
    if m["method"] != me or m.get("full_watermark") or i % 5 == 0: continue
    M = np.array(Image.open(REL / "masks" / (Path(m["image_name"]).stem + ".png")).convert("L")) > 127
    if not M.any(): continue
    ys, xs = np.where(M); h, w = ys.max() + 1 - ys.min(), xs.max() + 1 - xs.min()
    if M.sum() >= 0.98 * h * w: info.append((m["image_name"], min(h, w), max(h, w) / min(h, w)))
print(f"{me} : {len(info)} vrais rectangles hors réserve ; petit côté < 64 px : {np.mean([s < 64 for _, s, _ in info]):.2f} ; rapport >= 8 : {np.mean([r >= 8 for _, _, r in info]):.2f}")
for d in sys.argv[2:]:
    z2 = np.load(Path(d) / f"pred_train_{me}.npz"); L = dict(zip(z2["names"].tolist(), z2["logits"].astype(np.float32))); out = []
    for lo, hi in ((0, 32), (32, 64), (64, 128), (128, 513)):
        v = [((L[n] > 0) & Y[n]).sum() / max(((L[n] > 0) | Y[n]).sum(), 1) for n, s, _ in info if lo <= s < hi and Y[n].any()]
        out.append(f"{lo}-{hi} px : {np.mean(v):.3f} ({len(v)})" if v else f"{lo}-{hi} px : aucun")
    v = [((L[n] > 0) & Y[n]).sum() / max(((L[n] > 0) | Y[n]).sum(), 1) for n, s, _ in info if s < 64 and Y[n].any()]
    print(f"   {d:44s} | " + " | ".join(out) + f" | tous < 64 px : {np.mean(v):.3f} ({len(v)})")
