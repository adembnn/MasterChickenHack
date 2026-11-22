"""Étiquettes de référence + sous-ensemble fixe de 20 images par classe (900).
Vraie tâche : étiquette = prédiction d'orig (pseudo-étiquette). Simulateur : forget_images.npz contient la vraie étiquette 'y', utilisée telle quelle
(écrite sous forme de logits fictifs pour que argmax = vraie classe ; les vrais logits vont dans forget_logits_reels.npy)."""
import nm, numpy as np
X = nm.forget(); m = nm.orig(); lg = nm.logits(m, X).numpy(); d = np.load(nm.T3 + '/forget_images.npz')
if 'y' in d.files:
    np.save(nm.MEAS + '/forget_logits_reels.npy', lg); yt = d['y']; print('précision du modèle sur ces images', float((lg.argmax(1) == yt).mean()))
    lg = np.zeros_like(lg); lg[np.arange(len(yt)), yt] = 20.0
np.save(nm.MEAS + '/forget_logits.npy', lg); y = lg.argmax(1)
rng = np.random.default_rng(0); sub = np.sort(np.concatenate([rng.permutation(np.where(y == c)[0])[:20] for c in np.unique(y)])); np.save(nm.MEAS + '/subset.npy', sub)
print('classes', len(np.unique(y)), 'subset', len(sub), 'effectifs', np.bincount(y, minlength=100).tolist(), flush=True)
