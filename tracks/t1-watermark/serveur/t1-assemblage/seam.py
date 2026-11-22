"""Y a-t-il une couture mesurable au bord exact des rectangles (image source collee dans une toile generee autour) ?
Pour chaque cote de chaque vrai rectangle du train (les trois filigranes) : saut moyen entre deux lignes voisines, le long du cote, pour des
decalages de -12 a +12 px autour du vrai bord. Si le vrai bord (decalage 0) ressort, on peut caler les boites au pixel pres, et meme trouver les
petits rectangles que ni le decodeur ni le reseau ne voient."""
import sys, json, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); from t1lib import *
from PIL import Image
sh = {r["name"]: r for r in json.loads((ME / "shapes_oracle.json").read_text())}; W = 12; out = {}
for r in load_train():
    s = sh[r["image_name"]]
    if r["full"] or s["rect"] < 0.98: continue
    a = load_img("train", r["image_name"]).astype(np.float32); y0, x0, h, w = s["bbox"]; y1, x1 = y0 + h, x0 + w
    gy = np.abs(np.diff(a, axis=0)).sum(2); gx = np.abs(np.diff(a, axis=1)).sum(2)          # gy[y] = saut entre les lignes y et y+1
    gy2 = np.abs(a[2:] - 2 * a[1:-1] + a[:-2]).sum(2); gx2 = np.abs(a[:, 2:] - 2 * a[:, 1:-1] + a[:, :-2]).sum(2)
    for name, edge, lo, hi, G1, ax in (("haut", y0, x0, x1, gy, 0), ("bas", y1, x0, x1, gy, 0), ("gauche", x0, y0, y1, gx, 1), ("droite", x1, y0, y1, gx, 1)):
        if edge - W - 2 < 0 or edge + W + 2 > 511: continue                                # cote colle au bord de l'image
        prof = np.array([(G1[edge - 1 + d, lo:hi] if ax == 0 else G1[lo:hi, edge - 1 + d]).mean() for d in range(-W, W + 1)])
        med = np.median(prof); mad = np.median(np.abs(prof - med)) + 1e-6
        out.setdefault(r["method"], []).append(((prof[W] - med) / mad, int(np.argmax(prof)) - W, prof[W] / (med + 1e-6)))
for m, v in out.items():
    v = np.array(v); print(f"{m:10s} cotes mesures {len(v):3d} | score du vrai bord (ecarts a la mediane, en MAD) : mediane {np.median(v[:, 0]):.2f}, part > 3 : {(v[:, 0] > 3).mean():.2f}, part > 6 : {(v[:, 0] > 6).mean():.2f}"
                     f" | le maximum du profil tombe sur le vrai bord (a 1 px pres) : {(np.abs(v[:, 1]) <= 1).mean():.2f} (hasard : 0,12) | saut au bord / saut median : {np.median(v[:, 2]):.2f}")
