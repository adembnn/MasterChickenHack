"""Ou le candidat perd-il encore du mIoU, image par image ? (demande de la session relais, 22/11 6 h 13)
Pour chaque filigrane : les 20 pires images du candidat (masques de validation croisee 5 plis) et la perte rangee par cause.
La "perte" d'une image = 1 - son mIoU ; la colonne "points" = ce que la moyenne des trois filigranes gagnerait si la categorie devenait parfaite.
Usage : python pires.py 38"""
import sys, json, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); from t1lib import *
n = sys.argv[1]; sh = {r["name"]: r for r in json.loads((ME / "shapes_oracle.json").read_text())}; ztr = np.load(ME / "feat_train.npz"); ACC = dict(zip(ztr["names"].tolist(), ztr["acc"])); TOT = {}
def iou(a, b): u = (a | b).sum(); return 1.0 if u == 0 else (a & b).sum() / u
for m in METHODS:
    rows = {r["image_name"]: r for r in load_train(m)}; z = np.load(ME / f"candidat_{n}_cv_{m}.npz"); names = z["names"].tolist(); M = z["masks"]; G = np.stack([rows[k]["gt"] for k in names]); sc = miou(M, G); R = []
    for i, k in enumerate(names):
        r = rows[k]; g = G[i]; p = M[i]; iw, ic = iou(p, g), iou(~p, ~g); pred_full = p.mean() > 0.5
        if r["full"]:
            if not pred_full: cat = "pleine rendue comme partielle"
            elif not (~p & ~g).any(): cat = "pleine, trou manque (aucun recouvrement)"
            elif ic < 0.5: cat = "pleine, trou mal dessine (IoU du trou < 0,5)"
            else: cat = "pleine, trou correct (IoU >= 0,5)"
            det = f"trou vrai {100 * (~g).mean():.1f} % / rendu {100 * (~p).mean():.1f} %, IoU trou {ic:.2f}"
        else:
            s = sh[k]; rect = s["rect"] > 0.98; side = min(s["bbox"][2], s["bbox"][3]); forme = (f"rectangle {'< 32' if side < 32 else '32-64' if side < 64 else '>= 64'} px" if rect else "objet")
            if pred_full: cat = "partielle rendue comme pleine"
            elif not (p & g).any(): cat = f"partielle, zone manquee ({forme})"
            elif iw < 0.5: cat = f"partielle, zone mal dessinee ({forme})"
            else: cat = f"partielle, zone correcte ({forme})"
            det = f"{forme}, zone vraie {100 * g.mean():.1f} % / rendue {100 * p.mean():.1f} %, IoU zone {iw:.2f}"
        R.append((float(sc[i]), k, cat, det, float(ACC[k])))
    R.sort(); print(f"\n=== {m} · candidat {n} · mIoU {sc.mean():.4f} sur {len(R)} images ===\n20 pires images :")
    for s_, k, cat, det, acc in R[:20]: print(f"   {k} {s_:.3f} | {cat} | {det} | bits lus sur l'image entiere {acc:.2f}")
    print("perte par cause (nombre d'images, mIoU moyen, points sur la moyenne des trois filigranes) :"); C = {}
    for s_, k, cat, det, acc in R: C.setdefault(cat, []).append(s_)
    for cat, v in sorted(C.items(), key=lambda kv: -sum(1 - x for x in kv[1])): print(f"   {cat:58s} n={len(v):3d} | mIoU {np.mean(v):.3f} | {sum(1 - x for x in v) / len(R) / 3:+.4f}"); TOT[(m, cat)] = sum(1 - x for x in v) / len(R) / 3
print("\n=== les 8 causes les plus couteuses, trois filigranes confondus (points sur la moyenne des trois) ===")
for (m, cat), v in sorted(TOT.items(), key=lambda kv: -kv[1])[:8]: print(f"   {v:+.4f} | {m} | {cat}")
