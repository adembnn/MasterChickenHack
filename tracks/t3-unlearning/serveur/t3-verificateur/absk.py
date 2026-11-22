"""t3-verificateur : retention approchee sur les images generees de classes ABSENTES du forget set (t3-nonmembres img/abs_t2i.npz) :
precision relative et KL(orig||fichier) a T=2 ; ce proxy a reproduit le rapport reel de la sonde G10 (0,70 contre 0,64). usage : python absk.py nom=chemin ..."""
import os, sys, numpy as np, torch
import vlogits as V
ROOT = V.ROOT; im = np.load(ROOT + '/t3-nonmembres/img/abs_t2i.npz'); X = im['images']
if os.environ.get('ODD'): X = X[1::2]; print('moitie impaire seulement (non vue par l ancre) :', len(X), 'images')
ref = V.build(); ref.load_state_dict(torch.load(ROOT + '/model.pt', map_location='cpu', weights_only=True)); ref = ref.to(V.dev).eval(); zo = V.predict(ref, X, bs=50).astype(np.float64); del ref
def lsm(z, t=1.0): z = z / t; z = z - z.max(1, keepdims=True); return z - np.log(np.exp(z).sum(1, keepdims=True))
yo = zo.argmax(1); lo = lsm(zo, 2)
for a in sys.argv[1:]:
    nm, p = a.split('=', 1); m = V.build(); m.load_state_dict(torch.load(p, map_location='cpu', weights_only=True)); m = m.to(V.dev).eval(); z = V.predict(m, X, bs=50).astype(np.float64); del m; torch.cuda.empty_cache()
    kl = (np.exp(lo) * (lo - lsm(z, 2))).sum(1); acc = (z.argmax(1) == yo).mean()
    print(f'{nm:30s} accord avec orig {acc:.3f} KL(orig||fichier) T2 {kl.mean():.3f} (mediane {np.median(kl):.3f}, 90e centile {np.quantile(kl, 0.9):.3f}) retention approchee {min(acc, 1) / (1 + kl.mean()):.3f}')
