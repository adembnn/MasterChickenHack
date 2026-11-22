"""Écart apparié entre deux cartes (mIoU officiel, seuil 0), image par image, avec intervalle à 95 % par rééchantillonnage (10 000 tirages) : hors réserve, réserve, pleines, partielles.
usage : cmp.py <filigrane> <dossier nouveau> <dossier de base>"""
import sys, numpy as np
import ens
m, dn, db = sys.argv[1:4]; rng = np.random.default_rng(0)
nn, Ln = ens.load(dn, m, "train"); nb, Lb = ens.load(db, m, "train"); assert nn == nb
res = np.array([ens.pos[n] % 5 == 0 for n in nn]); fu = np.array([ens.full[n] for n in nn])
sn = np.array([ens.miou(a > 0, ens.Y[n]) for a, n in zip(Ln, nn)]); sb = np.array([ens.miou(a > 0, ens.Y[n]) for a, n in zip(Lb, nn)]); d = sn - sb
def ic(sel):
    v = d[sel]; b = v[rng.integers(0, len(v), (10000, len(v)))].mean(1); return f"{sn[sel].mean():.4f} contre {sb[sel].mean():.4f} : {v.mean():+.4f}  IC95 [{np.percentile(b, 2.5):+.4f} ; {np.percentile(b, 97.5):+.4f}]  ({int((v > 0.001).sum())} mieux, {int((v < -0.001).sum())} pires, {len(v)} images)"
print(f"{m} : {dn}  contre  {db}")
for nom, sel in (("hors réserve", ~res), ("réserve", res), ("toutes", res | ~res), ("pleines hors réserve", ~res & fu), ("pleines réserve", res & fu), ("partielles hors réserve", ~res & ~fu), ("partielles réserve", res & ~fu)): print(f"   {nom:24s} {ic(sel)}")
