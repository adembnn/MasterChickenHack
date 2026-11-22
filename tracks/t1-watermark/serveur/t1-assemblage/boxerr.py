"""Erreurs des boites du decodeur sur les rectangles du train : de combien chaque bord est-il decale, et l'erreur est-elle systematique (corrigible) ?"""
import sys, glob, json, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); from t1lib import *
m = sys.argv[1] if len(sys.argv) > 1 else "trustmark"
R = []
for f in sorted(glob.glob(str(ME / "boxtm" / f"boxes_{m}_train*.json"))): R += json.load(open(f))
R = [r for r in R if r.get("box") is not None and not r["full"]]
rect = [r for r in R if r["rect"]]; obj = [r for r in R if not r["rect"]]
print(m, "partielles cherchees", len(R), "rectangles", len(rect), "objets", len(obj))
print("z rectangles : quantiles 0/10/50/90", np.quantile([r["z"] for r in rect], [0, .1, .5, .9]).round(2), "| z objets : quantiles 50/90/100", np.quantile([r["z"] for r in obj], [.5, .9, 1]).round(2))
def iou(b, g):
    iy = max(0, min(b[2], g[2]) - max(b[0], g[0])); ix = max(0, min(b[3], g[3]) - max(b[1], g[1])); i = iy * ix
    return i / ((b[2] - b[0]) * (b[3] - b[1]) + (g[2] - g[0]) * (g[3] - g[1]) - i)
ok = [r for r in rect if r["z"] >= 6]; B = np.array([r["box"] for r in ok], float); G = np.array([r["gt_box"] for r in ok], float); D = B - G
h, w = G[:, 2] - G[:, 0], G[:, 3] - G[:, 1]; bh, bw = B[:, 2] - B[:, 0], B[:, 3] - B[:, 1]
print("rectangles avec z >= 6 :", len(ok), "| IoU moyen", np.mean([iou(b, g) for b, g in zip(B, G)]).round(3))
print("decalage par bord (trouve - vrai), px : mediane", np.median(D, 0), "| quartiles", np.quantile(D, .25, 0), np.quantile(D, .75, 0), "| ecart absolu median", np.median(np.abs(D), 0))
print("taille trouvee / vraie : hauteur mediane", np.median(bh / h).round(3), "largeur", np.median(bw / w).round(3), "| centre : dy", np.median((B[:, 0] + B[:, 2] - G[:, 0] - G[:, 2]) / 2).round(1), "dx", np.median((B[:, 1] + B[:, 3] - G[:, 1] - G[:, 3]) / 2).round(1))
print("decalage relatif a la taille (bord/taille) : mediane", np.median(D / np.stack([h, w, h, w], 1), 0).round(3))
for r in sorted(ok, key=lambda r: r["iou"])[:12]: print("   IoU %.2f z %.1f trouve %s vrai %s" % (r["iou"], r["z"], [int(v) for v in r["box"]], r["gt_box"]))
# correction apprise en validation croisee (5 plis sur gidx) : bord_vrai = a * bord_trouve + b * taille + c, par bord
fold = np.array([r["gidx"] % 5 for r in ok]); P = np.zeros_like(B)
for f in range(5):
    tr = fold != f
    for k in range(4):
        X = np.stack([B[:, k], (bh if k % 2 == 0 else bw), np.ones(len(B))], 1); c, *_ = np.linalg.lstsq(X[tr], G[tr, k], rcond=None); P[~tr, k] = X[~tr] @ c
print("apres correction lineaire par bord (validation croisee) : IoU moyen", np.mean([iou(b, g) for b, g in zip(P, G)]).round(3), "| avec seulement le decalage median retire :", np.mean([iou(b - np.median(D, 0), g) for b, g in zip(B, G)]).round(3))
