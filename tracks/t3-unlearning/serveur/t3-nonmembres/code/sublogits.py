"""Logits de candidats déjà soumis sur les 3 500 images à oublier (pour confronter l'estimation du gold aux scores réels).
usage : sublogits.py nom=chemin.pt ...   -> meas/sub_<nom>.npz (propre + 4 familles de copies du simulateur)"""
import nm, sys, numpy as np, torch, torch.nn.functional as F
X = nm.forget(); y = torch.from_numpy(np.load(nm.MEAS + '/forget_logits.npy').argmax(1)).long()
for a in sys.argv[1:]:
    name, path = a.split('=', 1); m = nm.C.load_sd(path); d = {}
    for an in ['propre', 'alt0 recadrage+flip', 'alt1 couleur douce', 'alt2 flou doux', 'alt3 bruit+recadrage']:
        z = nm.logits(m, X, None if an == 'propre' else nm.AUG[an]); d[an.split(' ')[0]] = z.numpy().astype(np.float32)
        ce = F.cross_entropy(z.double(), y, reduction='none')
        print(f"{name:28s} {an:22s} CE {ce.mean():.4f} / {ce.std():.4f}  classes changées {(z.argmax(1) != y).float().mean():.4f}  maxp T2 {F.softmax(z / 2, 1).max(1).values.mean():.4f}", flush=True)
    np.savez_compressed(f'{nm.MEAS}/sub_{name}.npz', **d); del m
