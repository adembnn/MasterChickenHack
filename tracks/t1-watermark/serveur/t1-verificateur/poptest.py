"""Sans vérité : la population du test ressemble-t-elle à celle du train ? Part filigranée PRÉDITE par image (candidat 35), train (validation croisée)
contre test, par filigrane ; et train-vérité pour référence."""
import json, numpy as np, torch
from scipy.stats import ks_2samp
from verif import HERE, METHODS, load_truth
A = HERE / "t1-assemblage"; meta, g = load_truth(check_png=False); tm = json.load(open(HERE / "release" / "test" / "metadata.json")); s = torch.load(A / "submission_candidat_35.pt", weights_only=True); T = dict(zip(s["image_names"], s["masks"]))
def desc(f): r = np.minimum(f, 1 - f); return f"pleines prédites (> 50 % filigrané) {np.mean(f > .5):.3f} ; zone rare < 1 % : {np.mean(r < .01):.3f}, < 3 % : {np.mean(r < .03):.3f}, 3-15 % : {np.mean((r >= .03) & (r < .15)):.3f}, > 15 % : {np.mean(r >= .15):.3f} ; médiane de la zone rare {np.median(r):.3f}"
for me in METHODS:
    z = np.load(A / f"candidat_35_cv_{me}.npz"); names = [str(a) for a in z["names"]]; ftr = z["masks"].reshape(len(names), -1).mean(1); fte = np.array([T[m["image_name"]].float().mean().item() for m in tm if m["method"] == me]); ftruth = np.array([g[a].mean() for a in names])
    k = ks_2samp(np.minimum(ftr, 1 - ftr), np.minimum(fte, 1 - fte)); print(f"[{me}]\n   train vérité   (170) : {desc(ftruth)}\n   train prédit   (170) : {desc(ftr)}\n   test prédit    (340) : {desc(fte)}\n   Kolmogorov-Smirnov train prédit / test prédit sur la taille de la zone rare : D = {k.statistic:.3f}, p = {k.pvalue:.2f}")
