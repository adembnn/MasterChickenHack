import numpy as np, sys, json
sys.path.insert(0, "."); from t1lib import *
m = "pixelseal"; rows = {r["image_name"]: r for r in load_train(m)}; sh = {r["name"]: r for r in json.loads((ME / "shapes_oracle.json").read_text())}
a = np.load(f"candidat_36_cv_{m}.npz"); b = np.load(f"candidat_35_cv_{m}.npz"); n = a["names"].tolist(); assert n == b["names"].tolist()
G = np.stack([rows[k]["gt"] for k in n]); d = miou(a["masks"], G) - miou(b["masks"], G); fold = a["fold"]; rng = np.random.default_rng(0)
kind = np.array(["pleine" if rows[k]["full"] else ("rectangle" if sh[k]["rect"] > 0.98 else "objet") for k in n])
print("PixelSeal 36 - 35, validation croisee, par pli (pli 0 = reserve) :", {int(f): (round(float(d[fold == f].mean()), 4), int((fold == f).sum())) for f in range(5)})
print("ecart-type de l'ecart par image :", round(float(d.std()), 3), "-> erreur type sur 30 images :", round(float(d.std() / np.sqrt(30)), 3), "| sur 170 :", round(float(d.std() / np.sqrt(170)), 4))
for k in ("pleine", "rectangle", "objet"):
    s = kind == k; print(f"  {k:9s} toutes n={int(s.sum()):3d} ecart {d[s].mean():+.4f} | reserve n={int((s & (fold == 0)).sum()):2d} ecart {d[s & (fold == 0)].mean():+.4f} | hors reserve ecart {d[s & (fold != 0)].mean():+.4f}")
h36 = np.load(f"candidat_36_honnete_{m}.npz"); h35 = np.load(f"candidat_35_honnete_{m}.npz"); dh = miou(h36["masks"], G) - miou(h35["masks"], G); res = fold == 0
print("masques 'honnetes' (recette reglee hors reserve) : reserve", round(float(dh[res].mean()), 4), "| par type sur la reserve :", {k: round(float(dh[res & (kind == k)].mean()), 4) for k in ("pleine", "rectangle", "objet")})
w = np.argsort(dh[res])[:4]; idx = np.where(res)[0][w]
for i in idx: print("   pire image de reserve :", n[i], kind[i], "ecart %+.3f" % dh[i], "part filigranee %.3f" % G[i].mean())
