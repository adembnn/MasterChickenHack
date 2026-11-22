"""Les vérités du train se répètent-elles (même masque pour deux images, d'un filigrane à l'autre) ? Si oui, banque de masques réutilisable."""
import json
import numpy as np
from verif import HERE, load_truth
meta, g = load_truth(check_png=False); names = [m["image_name"] for m in meta]; me = np.array([m["method"] for m in meta]); full = np.array([m["full_watermark"] for m in meta])
print("clés train :", sorted(meta[0].keys()), "| exemple :", {k: meta[0][k] for k in meta[0]})
tm = json.load(open(HERE / "release" / "test" / "metadata.json")); print("clés test :", sorted(tm[0].keys()), "| exemple :", tm[0])
G = np.stack([g[a] for a in names]).reshape(len(names), -1).astype(np.float32); R = np.where(full[:, None], 1 - G, G)        # zone rare
inter = R @ R.T; area = R.sum(1); union = area[:, None] + area[None, :] - inter; iou = inter / np.maximum(union, 1); np.fill_diagonal(iou, 0)
best = iou.max(1); j = iou.argmax(1)
print(f"IoU de la zone rare avec la plus ressemblante des 509 autres images : médiane {np.median(best):.3f} ; >= 0,9 pour {np.mean(best >= .9):.3f} ; >= 0,7 pour {np.mean(best >= .7):.3f} ; >= 0,5 pour {np.mean(best >= .5):.3f}")
for m in ("pixelseal", "trustmark", "stegastamp"):
    k = me == m; print(f"   {m:10s} >= 0,9 : {np.mean(best[k] >= .9):.3f} ; >= 0,7 : {np.mean(best[k] >= .7):.3f} ; pleines >= 0,7 : {np.mean(best[k & full] >= .7):.3f} ; partielles >= 0,7 : {np.mean(best[k & ~full] >= .7):.3f}")
k = best >= .9; print("paires >= 0,9 :", [(names[i], me[i], names[j[i]], me[j[i]], round(float(best[i]), 2), round(float(area[i] / 16384), 3)) for i in np.where(k)[0][:12]])
# même forme à une translation près ? (corrélation croisée des masques, par FFT) sur un échantillon
F = np.fft.rfft2(R.reshape(-1, 128, 128)); 
def shift_iou(i):
    c = np.fft.irfft2(F[i][None] * np.conj(F), s=(128, 128)).reshape(len(R), -1).max(1); u = area[i] + area - c; v = c / np.maximum(u, 1); v[i] = 0; return v.max()
rng = np.random.default_rng(0); S = rng.choice(len(R), 150, replace=False); sv = np.array([shift_iou(i) for i in S])
print(f"même comparaison à une translation (circulaire) près, 150 images : médiane {np.median(sv):.3f} ; >= 0,9 pour {np.mean(sv >= .9):.3f} ; >= 0,7 pour {np.mean(sv >= .7):.3f}")
