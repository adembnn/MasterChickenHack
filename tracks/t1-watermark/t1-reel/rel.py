"""Seuil relatif pour les images prédites partielles où presque rien ne dépasse 0 : masque = logit > min(0, m - delta), m = maximum de la carte lissée (moyenne 3 x 3) de l'image.
N'agit que si m < delta (détection faible). Mesure CV / réserve sur les partielles et sur tout, pour plusieurs delta. usage : rel.py <filigrane> <dossier>"""
import sys, numpy as np
from pathlib import Path
import ens
m_, d = sys.argv[1], sys.argv[2]; names, L = ens.load(d, m_, "train"); res = np.array([ens.pos[n] % 5 == 0 for n in names]); fu = np.array([ens.full[n] for n in names]); Y = np.stack([ens.Y[n] for n in names])
print("images au masque vide :", int((Y.reshape(len(Y), -1).sum(1) == 0).sum()), "sur", len(Y), "| partielles sans aucune case prédite :", int(((L > 0).reshape(len(L), -1).sum(1) == 0)[~fu].sum()), "sur", int((~fu).sum()))
def smooth(a): p = np.pad(a, ((0, 0), (1, 1), (1, 1)), mode="edge"); return sum(p[:, i:i + 128, j:j + 128] for i in range(3) for j in range(3)) / 9
S = smooth(L); mx = S.reshape(len(S), -1).max(1); gate = (L > 0).reshape(len(L), -1).mean(1) <= 0.5
def show(nom, P):
    sc = np.array([ens.miou(p, y) for p, y in zip(P, Y)]); print(f"   {nom:22s} CV {sc[~res].mean():.4f} (partielles {sc[~res & ~fu].mean():.4f}) | réserve {sc[res].mean():.4f} (partielles {sc[res & ~fu].mean():.4f})")
show("seuil 0", L > 0)
for de in (1.0, 2.0, 3.0, 4.0, 6.0):
    t = np.where(gate, np.minimum(0, mx - de), 0.0); show(f"relatif, delta {de}", S > t[:, None, None] if False else np.where(gate[:, None, None], (S > t[:, None, None]) | (L > 0), L > 0))
