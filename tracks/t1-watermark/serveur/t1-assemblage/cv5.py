"""Validation croisee complete de la procedure d'assemblage (demande du verificateur, 22/11 2 h).
Les 510 images du train, reserve comprise, sont reparties en 5 plis (index % 5). Pour chaque pli, TOUT est refait sur les 4 autres :
seuil de population, normalisation, choix de la source par population (validation croisee interne sur ces 4 plis), recette (lissage, seuil,
plancher de la classe rare, nettoyage, seuil de confiance des boites). Le pli tenu a l'ecart est binarise avec cette recette.
Sortie : candidat_<n>_cv_<filigrane>.npz (names, masks 170x128x128 bool, fold) + choix par pli dans cv5_<n>.json.
Usage : python cv5.py --n 30 --only REGEX [--skip REGEX] [--boxes REGEX] [--grille ancienne] [--plis 0,1,2,3,4]
Les sources "rang-*" et "pl-*" sont elles-memes hors-pli sur les 170 images (4 plis hors reserve, reserve predite par le modele appris sans elle)."""
import sys, json, argparse, time, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); import asm; from asm import *
ap = argparse.ArgumentParser(); ap.add_argument("--n", required=True); ap.add_argument("--only", default=""); ap.add_argument("--skip", default=""); ap.add_argument("--boxes", default="")
ap.add_argument("--grille", default="nouvelle"); ap.add_argument("--plis", default="0,1,2,3,4"); ap.add_argument("--methods", default=",".join(METHODS)); a = ap.parse_args()
OPT.update(rect=False, snap=False, only=a.only, skip=a.skip, boxes=a.boxes)
if a.grille == "ancienne": asm.grid = lambda Zs: np.quantile(Zs[:, ::2, ::2], np.linspace(0.01, 0.99, 50))     # grille de seuils d'avant la correction (celle du fichier 27)
PLIS = [int(k) for k in a.plis.split(",")]; t0 = time.time()
tr, te, S, notes = discover(); ztr = np.load(ME / "feat_train.npz"); ACC = dict(zip(ztr["names"].tolist(), ztr["acc"])); OUT = {}
for m in a.methods.split(","):
    rows = tr[m]; names = [r["image_name"] for r in rows]; G = np.stack([r["gt"] for r in rows]); fold = np.array([r["fold"] for r in rows]); full_true = np.array([r["full"] for r in rows]); acc = np.array([ACC[n] for n in names])
    C, _ = load_side(names, "train", m); src = {k: v for k, v in S[m].items() if v[3]}                 # sources hors-pli seulement
    if not src: sys.exit(f"aucune source hors-pli pour {m}")
    M = np.zeros_like(G); done = np.zeros(len(rows), bool); choix = {}
    for f in PLIS:
        fit = fold != f; ev = fold == f
        cands = np.unique(acc); th = max(cands, key=lambda t: (np.mean((acc[fit] > t) == full_true[fit]), -abs(t - 0.75))); pf = acc > th
        Z = {k: (v[0] - v[0][fit].mean()) / (v[0][fit].std() + 1e-6) for k, v in src.items()}
        for full in (True, False):
            I = pf == full; fi = I & fit; ei = I & ev; Cs = None if full else C; sub = lambda X, k: None if X is None else X[k]; best = None
            for k in Z:
                sc = cv(Z[k][fi], G[fi], fold[fi], full, sub(Cs, fi))
                if best is None or sc > best[0]: best = (sc, k)
            cst = float(miou(np.full_like(G[fi], full), G[fi]).mean()) if fi.any() else 0.0
            if cst > best[0]: M[ei] = full; choix[f"{f}/{POPN[full]}"] = dict(source="constante", cv=cst)
            else:
                p, sc = tune(Z[best[1]][fi], G[fi], full, sub(Cs, fi)); M[ei] = apply(Z[best[1]][ei], p, full, sub(Cs, ei)); choix[f"{f}/{POPN[full]}"] = dict(source=best[1], cv=round(best[0], 4), recette=p)
        done |= ev; print(m, "pli", f, "fait", round(time.time() - t0), "s", {k: v["source"] for k, v in choix.items() if k.startswith(f"{f}/")}, flush=True)
    sc = miou(M, G); suf = "" if len(PLIS) == 5 else "_plis" + "".join(str(k) for k in PLIS)
    np.savez_compressed(ME / f"candidat_{a.n}_cv_{m}{suf}.npz", names=np.array(names), masks=M, fold=fold, fait=done)
    OUT[m] = dict(choix=choix, cv510=round(float(sc[done].mean()), 4), pleines=round(float(sc[done & full_true].mean()), 4), partielles=round(float(sc[done & ~full_true].mean()), 4), par_pli={int(f): round(float(sc[fold == f].mean()), 4) for f in PLIS})
    print("CV5", a.n, m, "score sur les images faites :", OUT[m]["cv510"], "pleines", OUT[m]["pleines"], "partielles", OUT[m]["partielles"], "par pli", OUT[m]["par_pli"], flush=True)
(ME / f"cv5_{a.n}{'' if len(PLIS) == 5 else '_plis' + ''.join(str(k) for k in PLIS)}.json").write_text(json.dumps(OUT, indent=1, default=str))
if len(OUT) == 3: print("CV5", a.n, "moyenne des trois filigranes :", round(float(np.mean([OUT[m]["cv510"] for m in METHODS])), 4))
