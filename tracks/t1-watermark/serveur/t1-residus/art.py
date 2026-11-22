"""Les vraies images portent-elles une trace de traitement (JPEG, changement de taille) différente dedans / dehors ?
Local, sans GPU. Par filigrane et population : profil des différences par phase modulo 8 (grille JPEG), énergie fine, dedans vs dehors."""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
R = Path("E:/EAC-lab/t1/release/train"); meta = json.loads((R / "metadata.json").read_text())

def blockiness(img, mask=None):
    """Rapport entre la différence moyenne à cheval sur une frontière de bloc 8x8 et ailleurs (1,0 = pas de JPEG). Retourne (rapport, phase)."""
    y = img.astype(np.float32).mean(2); best = (0, 0)
    out = []
    for axis in (0, 1):
        d = np.abs(np.diff(y, axis=axis)); w = None
        if mask is not None:
            w = (mask[1:, :] & mask[:-1, :]) if axis == 0 else (mask[:, 1:] & mask[:, :-1])
        prof = np.zeros(8);
        for p in range(8):
            sl = d[p::8, :] if axis == 0 else d[:, p::8]
            if w is not None:
                ws = w[p::8, :] if axis == 0 else w[:, p::8]; prof[p] = sl[ws].mean() if ws.sum() > 50 else np.nan
            else: prof[p] = sl.mean()
        out.append(prof)
    prof = np.nanmean(out, 0); med = np.nanmedian(prof)
    return float(np.nanmax(prof) / (med + 1e-9)), int(np.nanargmax(prof)), prof

def fine(img, mask):
    y = img.astype(np.float32).mean(2); lap = np.abs(y[1:-1, 1:-1] * 4 - y[:-2, 1:-1] - y[2:, 1:-1] - y[1:-1, :-2] - y[1:-1, 2:]); m = mask[1:-1, 1:-1]
    return float(np.median(lap[m])) if m.sum() > 50 else float("nan")

for method in ("pixelseal", "trustmark", "stegastamp"):
    for pop in (True, False):
        es = [e for e in meta if e["method"] == method and bool(e.get("full_watermark")) == pop]
        rows = []
        for e in es:
            img = np.asarray(Image.open(R / "images" / f"{e['image_name']}.png").convert("RGB")); mk = np.asarray(Image.open(R / "masks" / f"{e['image_name']}.png")) > 127
            if mk.mean() < 0.03 or mk.mean() > 0.97: continue
            bi, pi, _ = blockiness(img, mk); bo, po, _ = blockiness(img, ~mk); ba, pa, prof = blockiness(img)
            rows.append((bi, bo, ba, fine(img, mk), fine(img, ~mk), pi, po))
        A = np.array(rows)
        print("%-10s %-10s n=%3d | grille JPEG (1,0 = aucune) dedans %.3f  dehors %.3f  image %.3f | détail fin médian dedans %.2f  dehors %.2f | phases dedans %s" % (
            method, "pleines" if pop else "partielles", len(A), np.median(A[:, 0]), np.median(A[:, 1]), np.median(A[:, 2]), np.nanmedian(A[:, 3]), np.nanmedian(A[:, 4]), np.bincount(A[:, 5].astype(int), minlength=8)))
