"""Choix par type PRÉDIT (sans vérité) entre candidat 38 et 39 PixelSeal : 39 si la zone filigranée prédite par le 39 a la forme d'un rectangle
(remplissage de sa boîte englobante >= seuil) et si l'image est devinée partielle ; sinon 38. Seuil choisi sur 4 plis, appliqué au cinquième."""
import numpy as np
from verif import HERE, miou_vec, load_truth
rng = np.random.default_rng(1); meta, g = load_truth(check_png=False); idx = {m["image_name"]: i for i, m in enumerate(meta)}; A = HERE / "t1-assemblage"
def cand(c):
    q = np.load(A / f"candidat_{c}_cv_pixelseal.npz"); return {str(a): m.astype(bool) for a, m in zip(q["names"], q["masks"])}
c38, c39 = cand("38"), cand("39"); names = sorted(c38); n = len(names); C38 = np.stack([c38[a] for a in names]); C39 = np.stack([c39[a] for a in names]); G = np.stack([g[a] for a in names]); fold = np.array([idx[a] % 5 for a in names])
s38, s39 = miou_vec(C38, G), miou_vec(C39, G); part = C39.reshape(n, -1).mean(1) < .5
def fill(m):
    ys, xs = np.where(m); return m.sum() / ((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)) if len(ys) else 0
F = np.array([fill(C39[i]) for i in range(n)]); H = np.zeros(n); ch = []
for k in range(5):
    fit = fold != k; ev = fold == k; t = max([0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.01], key=lambda t: np.where(part[fit] & (F[fit] >= t), s39[fit], s38[fit]).mean()); ch.append(t); H[ev] = np.where(part[ev] & (F[ev] >= t), s39[ev], s38[ev])
d = H - s38; b = d[rng.integers(0, n, (20000, n))].mean(1); print(f"seuils par pli {ch} ; images prises au 39 : {int((part & (F >= np.median(ch))).sum())} ; PixelSeal {H.mean():.4f} contre 38 {s38.mean():.4f} et 39 {s39.mean():.4f} ; écart contre 38 {d.mean():+.4f} IC95 [{np.percentile(b, 2.5):+.4f} ; {np.percentile(b, 97.5):+.4f}] ; réserve {d[fold == 0].mean():+.4f} ; global {d.mean() / 3:+.4f}")
