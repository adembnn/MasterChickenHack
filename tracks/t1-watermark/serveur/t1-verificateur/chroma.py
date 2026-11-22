"""Signal local sans entraînement ni original : le résidu PixelSeal est le même sur R, V, B (gris), donc il ajoute de l'énergie haute fréquence à la
luminance Y = (R+V+B)/3 et aucune aux différences R−V et V−B. Statistique par case : log énergie HF(Y) − log énergie HF(chrominance).
AUC par image (cases filigranées contre cases propres), sur les vraies images, et témoin : mêmes masques posés sur des images sans PixelSeal."""
import sys
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from verif import HERE, load_truth
ME = sys.argv[1] if len(sys.argv) > 1 else "pixelseal"; OTHER = "trustmark" if ME != "trustmark" else "stegastamp"
meta, g = load_truth(check_png=False); img = lambda n: np.asarray(Image.open(HERE / "release" / "train" / "images" / f"{n}.png").convert("RGB")).astype(np.float32)
def hp(x): return x - ndi.uniform_filter(x, 3)                                    # passe-haut : écart à la moyenne 3x3
def stat(a, win):
    Y = a.mean(-1); c1 = a[..., 0] - a[..., 1]; c2 = a[..., 1] - a[..., 2]
    ey = ndi.uniform_filter(hp(Y) ** 2, win); ec = ndi.uniform_filter(hp(c1) ** 2 + hp(c2) ** 2, win) / 6      # /6 : même variance que Y pour un bruit indépendant par canal
    s = np.log(ey + 0.02) - np.log(ec + 0.02); return s.reshape(128, 4, 128, 4).mean((1, 3)), np.log(ey + 0.02).reshape(128, 4, 128, 4).mean((1, 3))
def auc(v, t):
    v = v.ravel(); t = t.ravel(); n1 = t.sum(); n0 = len(t) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = v.argsort().argsort() + 1; return (r[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
own = [m for m in meta if m["method"] == ME]; oth = [m for m in meta if m["method"] == OTHER]
for win in (8, 16, 32):
    A, A0, AE, full, rf, ms = [], [], [], [], [], []
    for k, m in enumerate(own):
        G = g[m["image_name"]]; s, e = stat(img(m["image_name"]), win); s0, _ = stat(img(oth[k % len(oth)]["image_name"]), win)
        A.append(auc(s, G)); A0.append(auc(s0, G)); AE.append(auc(e, G)); full.append(m["full_watermark"]); R = ~G if m["full_watermark"] else G; rf.append(R.mean()); ys, xs = np.where(R); ms.append(min(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1) * 4 if len(ys) else 0)
    A, A0, AE, full, rf, ms = map(np.array, (A, A0, AE, full, rf, ms))
    def line(lab, k): return f"{lab} n={k.sum()} : AUC {np.nanmean(A[k]):.3f} (témoin {np.nanmean(A0[k]):.3f} ; énergie de luminance seule {np.nanmean(AE[k]):.3f}) ; part des images > 0,6 : {np.nanmean(A[k] > .6):.2f}, > 0,7 : {np.nanmean(A[k] > .7):.2f}"
    print(f"fenêtre {win} px | " + line("toutes", np.ones(len(A), bool)))
    for lab, k in (("pleines", full), ("partielles", ~full), ("partielles < 3 %", ~full & (rf < .03)), ("partielles >= 3 %", ~full & (rf >= .03)), ("pleines, trou < 3 %", full & (rf < .03)), ("pleines, trou >= 3 %", full & (rf >= .03))): print("    " + line(lab, k))
