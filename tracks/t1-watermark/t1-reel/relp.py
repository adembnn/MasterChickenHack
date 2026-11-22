"""Pendant de rel.py pour les images prédites PLEINES : si presque aucune case n'est prédite propre, case propre = logit lissé < min(carte) + delta (un « trou jamais vide »).
Mesure CV / réserve sur les pleines, pour plusieurs delta et pour un seuil fixe relevé. usage : relp.py <filigrane> <dossier>"""
import sys, numpy as np
import ens
m_, d = sys.argv[1], sys.argv[2]; names, L = ens.load(d, m_, "train"); res = np.array([ens.pos[n] % 5 == 0 for n in names]); fu = np.array([ens.full[n] for n in names]); Y = np.stack([ens.Y[n] for n in names])
def smooth(a): p = np.pad(a, ((0, 0), (1, 1), (1, 1)), mode="edge"); return sum(p[:, i:i + 128, j:j + 128] for i in range(3) for j in range(3)) / 9
S = smooth(L); mn = S.reshape(len(S), -1).min(1); gate = (L > 0).reshape(len(L), -1).mean(1) > 0.5
print("pleines sans aucune case prédite propre :", int(((L > 0).reshape(len(L), -1).all(1))[fu].sum()), "sur", int(fu.sum()))
def show(nom, P):
    sc = np.array([ens.miou(p, y) for p, y in zip(P, Y)]); h = np.array([ens.iou0(p, y) for p, y in zip(P, Y)]); print(f"   {nom:24s} pleines CV {sc[~res & fu].mean():.4f} (trou {h[~res & fu].mean():.3f}) | réserve {sc[res & fu].mean():.4f} (trou {h[res & fu].mean():.3f})")
show("seuil 0", L > 0)
for t in (0.5, 1.0, 1.5, 2.0): show(f"seuil fixe {t}", np.where(gate[:, None, None], L > t, L > 0))
for de in (0.5, 1.0, 2.0, 3.0):
    t = np.maximum(0, mn + de); show(f"relatif, delta {de}", np.where(gate[:, None, None], (S > t[:, None, None]) & (L > 0), L > 0))
