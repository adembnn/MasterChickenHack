"""Calibration dans le simulateur. Modèle M = un modèle du simulateur entraîné de zéro (gold_sK). Deux jeux de 900 images (20 par classe, 45 classes) :
  members    : images du jeu d'entraînement de M (M les a vues) ;
  nonmembers : images du jeu de contrôle (M ne les a jamais vues, même distribution).
On applique les mêmes générateurs aux deux. M sur generateur(members) comparé à M sur nonmembers dit quelle force efface la mémorisation ;
M sur generateur(nonmembers) comparé à M sur nonmembers dit ce que le générateur dégrade à lui seul.
usage : simcal_prep.py SIMDIR GOLD.pt DEST members|nonmembers"""
import sys, os, numpy as np
sim, gold, dest, mode = sys.argv[1:5]; os.makedirs(dest, exist_ok=True)
X = np.load(sim + '/data/x.npy', mmap_mode='r'); Y = np.load(sim + '/data/y.npy'); W = dict(np.load(sim + '/data/world_A.npz'))
fc = np.unique(Y[W['forget']]); rng = np.random.default_rng(1); pool = {'members': W['retain'], 'forget': W['forget'], 'nonmembers': W['ctrl']}[mode]
sel = np.sort(np.concatenate([rng.permutation(pool[Y[pool] == c])[:20] for c in fc]))
np.savez(dest + '/forget_images.npz', images=np.ascontiguousarray(X[sel]), y=Y[sel], idx=sel)
if gold != '-' and not os.path.exists(dest + '/model.pt'): os.symlink(gold, dest + '/model.pt')
print('simcal', mode, len(sel), 'images, classes', len(fc))
