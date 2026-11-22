"""Mesure d'assemblage pur sur PixelSeal, images partielles (demande de la session relais, 22/11 5 h 24).
Constat : le vote par region gagne sur les objets (0,748 -> 0,888) mais perd sur les rectangles (0,711 -> 0,691), et la carte reelp4 est devenue bonne
sur les petits rectangles. Test : par image, si le masque de la carte brute est rectangulaire (aire / aire de sa boite englobante >= r), on garde la
carte brute (mode "carte") ou sa boite englobante pleine (mode "boite") ; sinon le vote. r et le mode sont regles sur les 4 plis d'apprentissage,
le 5e est tenu a l'ecart (meme protocole que cv5.py). Sortie : ecart apparie contre le vote seul, plus le plafond (meilleur des deux par image).
Usage : python gate.py [base=reelp4]"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); import asm; from asm import *
base = sys.argv[1] if len(sys.argv) > 1 else "reelp4"; m = "pixelseal"
OPT.update(rect=False, snap=False, only=f"{base}", skip="rang2|rang3|residus", boxes="t1-pixelseal/box_")
tr, te, S, notes = discover(); ztr = np.load(ME / "feat_train.npz"); ACC = dict(zip(ztr["names"].tolist(), ztr["acc"]))
rows = tr[m]; names = [r["image_name"] for r in rows]; G = np.stack([r["gt"] for r in rows]); fold = np.array([r["fold"] for r in rows]); full_true = np.array([r["full"] for r in rows]); acc = np.array([ACC[n] for n in names])
C, _ = load_side(names, "train", m); kr = [k for k in S[m] if k.endswith(f"{base}/pred")][0]; kv = [k for k in S[m] if f"rang-{base}-pred" in k][0]; print("carte :", kr, "| vote :", kv, flush=True)
def rectness(M):
    out = np.zeros(len(M), np.float32); B = np.zeros_like(M)
    for i, x in enumerate(M):
        lab, n = ndi.label(x)
        if not n: continue
        k = 1 + int(np.argmax(ndi.sum(x, lab, range(1, n + 1)))); ys, xs = np.where(lab == k); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        out[i] = len(ys) / ((y1 - y0) * (x1 - x0)); B[i, y0:y1, x0:x1] = True
    return out, B
RS = [9.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5, 0.0]; sub = lambda X, k: None if X is None else X[k]
sv = np.full(len(rows), np.nan); sg = sv.copy(); so = sv.copy(); sr = sv.copy(); ch = {}
for f in range(5):
    fit = fold != f; ev = fold == f; cands = np.unique(acc); th = max(cands, key=lambda t: (np.mean((acc[fit] > t) == full_true[fit]), -abs(t - 0.75))); pf = acc > th
    I = ~pf; fi = I & fit; ei = I & ev; M = {}
    for k in (kr, kv):
        Z = (S[m][k][0] - S[m][k][0][fit].mean()) / (S[m][k][0][fit].std() + 1e-6); p, _ = tune(Z[fi], G[fi], False, sub(C, fi)); M[k] = (apply(Z[fi], p, False, sub(C, fi)), apply(Z[ei], p, False, sub(C, ei)))
    def mix(j, mode, r):
        rc, B = rectness(M[kr][j]); alt = M[kr][j] if mode == "carte" else B; return np.where((rc >= r)[:, None, None], alt, M[kv][j])
    best = max(((mode, r) for mode in ("carte", "boite") for r in RS), key=lambda q: miou(mix(0, *q), G[fi]).mean()); ch[f] = best
    a = miou(M[kv][1], G[ei]); b = miou(mix(1, *best), G[ei]); c = miou(M[kr][1], G[ei]); sv[ei] = a; sg[ei] = b; sr[ei] = c; so[ei] = np.maximum(a, c)
    print(f"pli {f} : {int(ei.sum())} images partielles | vote {a.mean():.4f} | carte {c.mean():.4f} | porte {best} {b.mean():.4f} | meilleur des deux par image {np.maximum(a, c).mean():.4f}", flush=True)
k = ~np.isnan(sv); d = sg[k] - sv[k]; rng = np.random.default_rng(0); n = len(rows); dd = np.zeros(n); dd[k] = d; boot = [rng.choice(dd, n).mean() for _ in range(4000)]
print(f"images partielles predites : {int(k.sum())} sur {n} | vote {sv[k].mean():.4f} | carte {sr[k].mean():.4f} | porte {sg[k].mean():.4f} | plafond {so[k].mean():.4f}")
print(f"ecart porte - vote sur les {n} images PixelSeal : {dd.mean():+.4f}, IC95 [{np.quantile(boot, .025):+.4f} ; {np.quantile(boot, .975):+.4f}] ({int((d > 1e-6).sum())} mieux, {int((d < -1e-6).sum())} pires) ; sur la moyenne des trois filigranes : {dd.mean() / 3:+.4f} | choix par pli {ch}")
print(f"plafond (choisir la meilleure des deux par image) sur les {n} images : {(so[k] - sv[k]).sum() / n:+.4f}, soit {(so[k] - sv[k]).sum() / n / 3:+.4f} sur la moyenne des trois")
