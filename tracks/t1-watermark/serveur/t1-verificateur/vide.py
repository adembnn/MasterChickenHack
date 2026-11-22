"""Partielles sans zone prédite : dans le candidat, combien d'images devinées partielles n'ont aucune (ou presque aucune) case filigranée, que valent-elles,
et que gagne-t-on à y rendre une tache autour du maximum de la carte lissée (cases où logit lissé >= maximum − delta) ? Les paramètres (aire minimale
qui déclenche, lissage, delta) sont choisis sur 4 plis (index % 5) et appliqués au cinquième.
Usage : vide.py CAND METHODE CARTE [CARTE_DE_LA_TACHE]"""
import sys
import numpy as np
from scipy import ndimage as ndi
from verif import HERE, miou_vec, load_truth
CAND, ME, TAG = sys.argv[1], sys.argv[2], sys.argv[3]; TAG2 = sys.argv[4] if len(sys.argv) > 4 else TAG
rng = np.random.default_rng(20261122); meta, g = load_truth(check_png=False); idx = {m["image_name"]: i for i, m in enumerate(meta)}; bym = {m["image_name"]: m for m in meta}
def L(tag):
    z = np.load(HERE / "t1-reel" / tag / f"pred_train_{ME}.npz"); return {str(a): l.astype(np.float32) for a, l in zip(z["names"], z["logits"])}
q = np.load(HERE / "t1-assemblage" / f"candidat_{CAND}_cv_{ME}.npz"); names = [str(a) for a in q["names"]]; C = q["masks"].astype(bool); n = len(names); G = np.stack([g[a] for a in names]); Z = np.stack([L(TAG2)[a] for a in names])
full = np.array([bym[a]["full_watermark"] for a in names]); fold = np.array([idx[a] % 5 for a in names]); f = np.load(HERE / "t1-assemblage" / "feat_train.npz"); acc = dict(zip([str(a) for a in f["names"]], f["acc"])); A = np.array([acc[a] for a in names])
sc = miou_vec(C, G); area = C.reshape(n, -1).mean(1); ov = (C & G).reshape(n, -1).sum(1)
print(f"=== {ME}, candidat {CAND} ({sc.mean():.4f}), tache tirée de {TAG2}")
p = ~full; print(f"  vraies partielles ({p.sum()}) : aucune case filigranée prédite {np.sum(p & (area == 0))} ; moins de 0,5 % de l'image {np.sum(p & (area < .005))} ; aucun recouvrement avec la vraie zone {np.sum(p & (ov == 0))} (score moyen de celles-ci {sc[p & (ov == 0)].mean():.3f}) ; score des autres partielles {sc[p & (ov > 0)].mean():.3f}")
SM = {s: np.stack([ndi.gaussian_filter(z, s) if s else z for z in Z]) for s in (0, 1, 2)}
def blob(i, s, delta):
    v = SM[s][i]; m = v >= v.max() - delta; lab, k = ndi.label(m); return lab == lab[np.unravel_index(v.argmax(), v.shape)]          # composante connexe qui contient le maximum
BL = {(s, dl): np.stack([blob(i, s, dl) for i in range(n)]) for s in SM for dl in (0.5, 1.0, 2.0, 3.0, 5.0)}
def apply(pf, amin, s, dl): use = ~pf & (area <= amin); return np.where(use[:, None, None], BL[(s, dl)], C), use
GRID = [(amin, s, dl) for amin in (-1, 0.0, 0.002, 0.005, 0.01) for s in SM for dl in (0.5, 1.0, 2.0, 3.0, 5.0)]; H = C.copy(); USE = np.zeros(n, bool); ch = []
for k in range(5):
    fit = fold != k; ev = fold == k; th = max(np.unique(A), key=lambda t: (np.mean((A[fit] > t) == full[fit]), -abs(t - 0.75))); pf = A > th
    best = max(GRID, key=lambda p_: (miou_vec(apply(pf, *p_)[0][fit], G[fit]).mean(), -p_[0])); h, u = apply(pf, *best); H[ev] = h[ev]; USE[ev] = u[ev]; ch.append(best)
sH = miou_vec(H, G); d = sH - sc; b = d[rng.integers(0, n, (20000, n))].mean(1)
print(f"  règle choisie par pli (aire max qui déclenche, lissage, delta) : {ch}")
print(f"  images modifiées {USE.sum()} (dont vraies partielles {np.sum(USE & ~full)}) ; candidat {sc.mean():.4f} → {sH.mean():.4f}, écart {d.mean():+.4f} IC95 [{np.percentile(b, 2.5):+.4f} ; {np.percentile(b, 97.5):+.4f}] ; réserve {d[fold == 0].mean():+.4f} ; hors réserve {d[fold != 0].mean():+.4f} ; mieux / pires {(d > 1e-9).sum()} / {(d < -1e-9).sum()}")
if USE.sum(): print(f"  sur les images modifiées : {sc[USE].mean():.3f} → {sH[USE].mean():.3f} ; tache touchant la vraie zone {np.mean((H & G).reshape(n, -1).sum(1)[USE] > 0):.2f}")
pf = A > 0.75
print("  pour mémoire, sans validation croisée (optimiste), lissage 1 : " + " ; ".join(f"aire <= {amin:g}, delta {dl:g} : {miou_vec(apply(pf, amin, 1, dl)[0], G).mean():.4f}" for amin in (0.0, 0.005) for dl in (1.0, 2.0, 3.0)))
