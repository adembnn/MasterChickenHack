"""Test court : les zones propres / filigranées portent-elles une trace de grille JPEG 8x8 (source photo décodée d'un JPEG) ?
Statistique par bloc 8x8 : part des coefficients DCT de haute fréquence presque nuls (|c| < 0,75). Un bloc décodé d'un JPEG, jamais
retouché, en a beaucoup SUR LA BONNE PHASE de grille ; une phase décalée ou un bloc retouché en a moins."""
import json
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent.parent; REL = HERE / "release"
meta = json.load(open(REL / "train" / "metadata.json"))
k = np.arange(8); D = np.sqrt(2 / 8) * np.cos(np.pi * (2 * k[None] + 1) * k[:, None] / 16); D[0] /= np.sqrt(2)
u, v = np.meshgrid(k, k, indexing="ij"); HF = (u + v) >= 7

def zmap(Y, dy, dx):
    nb = (512 - 8) // 8; B = Y[dy:dy + 8 * nb, dx:dx + 8 * nb].reshape(nb, 8, nb, 8).transpose(0, 2, 1, 3)
    C = D @ B @ D.T
    return (np.abs(C[..., HF]) < 0.75).mean(-1)

def auc(s, l):
    o = np.argsort(s, kind="stable"); r = np.empty(len(o)); r[o] = np.arange(1, len(o) + 1); n1 = l.sum(); n0 = len(l) - n1
    return float((r[l].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))

rng = np.random.default_rng(0); rows = []
for me in ("pixelseal", "trustmark", "stegastamp"):
    for full in (True, False):
        ids = [i for i, m in enumerate(meta) if m["method"] == me and m["full_watermark"] == full]
        for i in rng.choice(ids, 12, replace=False):
            n = meta[i]["image_name"]; im = np.asarray(Image.open(REL / "train" / "images" / f"{n}.png").convert("RGB"), np.float32)
            Y = 0.299 * im[..., 0] + 0.587 * im[..., 1] + 0.114 * im[..., 2] - 128
            M = np.asarray(Image.open(REL / "train" / "masks" / f"{n}.png").convert("L")) > 127
            zs = np.array([[zmap(Y, dy, dx).mean() for dx in range(8)] for dy in range(8)])
            dy, dx = np.unravel_index(zs.argmax(), zs.shape); peak = zs.max() / np.median(zs)
            Z = zmap(Y, dy, dx); nb = Z.shape[0]; Mb = M[dy:dy + 8 * nb, dx:dx + 8 * nb].reshape(nb, 8, nb, 8).mean((1, 3))
            pure = (Mb == 0) | (Mb == 1); lab = Mb[pure] > 0.5
            a = auc(Z[pure], lab) if 5 < lab.sum() < pure.sum() - 5 else float("nan")
            rows.append((me, full, peak, a, Z[pure][lab].mean() if lab.any() else np.nan, Z[pure][~lab].mean() if (~lab).any() else np.nan, (dy, dx)))
print("filigrane   pleine  pic de phase (max/médiane)  AUC bloc filigrané>propre  z filigrané  z propre  phases trouvées")
for me in ("pixelseal", "trustmark", "stegastamp"):
    for full in (True, False):
        r = [x for x in rows if x[0] == me and x[1] == full]
        print(f"{me:10s} {str(full):6s}  pic médian {np.median([x[2] for x in r]):.3f} (max {np.max([x[2] for x in r]):.3f})   AUC médiane {np.nanmedian([x[3] for x in r]):.3f} [min {np.nanmin([x[3] for x in r]):.3f} ; max {np.nanmax([x[3] for x in r]):.3f}]   z filigrané {np.nanmean([x[4] for x in r]):.3f}   z propre {np.nanmean([x[5] for x in r]):.3f}   {[x[6] for x in r][:6]}")
