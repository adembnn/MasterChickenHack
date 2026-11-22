"""Question du vérificateur : la chaîne traite-t-elle les images hors réserve autrement que la réserve ? Piste : la banque de formes (masques des DEUX AUTRES filigranes, hors réserve,
gardés à leur place 85 % du temps). Si un masque d'une image du filigrane évalué existe à l'identique chez un autre filigrane, le réseau a vu cette forme à cette place.
Mesure : pour chaque image d'un filigrane, meilleure IoU de son masque avec un masque de la banque (autres filigranes, index % 5 != 0)."""
import json, numpy as np
from pathlib import Path
REL = Path(r"E:/EAC-lab/t1/release/train"); meta = json.loads((REL / "metadata.json").read_text()); z = np.load(REL / "masks_128.npz"); Y = dict(zip(z["names"].tolist(), z["masks"].astype(bool)))
M = np.stack([Y[m["image_name"]] for m in meta]).reshape(len(meta), -1).astype(np.float32); me = np.array([m["method"] for m in meta]); res = np.arange(len(meta)) % 5 == 0; full = np.array([bool(m.get("full_watermark")) for m in meta])
inter = M @ M.T; area = M.sum(1); iou = inter / np.maximum(area[:, None] + area[None] - inter, 1)
for m in ("trustmark", "pixelseal", "stegastamp"):
    bank = (me != m) & ~res
    for nom, sel in (("hors réserve", (me == m) & ~res), ("réserve", (me == m) & res)):
        b = iou[np.ix_(sel, bank)].max(1); print(f"{m:10s} {nom:12s} : {int(sel.sum()):3d} images | masque identique dans la banque (IoU > 0,98) {np.mean(b > .98):.2f} | proche (IoU > 0,90) {np.mean(b > .90):.2f} | meilleure IoU médiane {np.median(b):.2f} | pleines identiques {np.mean(b[full[sel]] > .98):.2f} | partielles identiques {np.mean(b[~full[sel]] > .98):.2f}")
same = (iou > .98) & (me[:, None] != me[None]); i, j = np.where(same); print("paires identiques entre filigranes :", len(i) // 2, "; exemples :", [(meta[a]["image_name"], meta[b]["image_name"], int(a), int(b)) for a, b in list(zip(i, j))[:4]])
print("clés des métadonnées :", sorted(meta[0].keys()))
