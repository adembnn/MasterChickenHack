"""Retention approchee sur classes absentes, moitie NON vue par l'ancre (abs_t2i indices impairs) et moitie vue (pairs). usage : absm.py chemin.pt ..."""
import sys, os, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.expanduser('~/eac/t3/t3-redistill/tmcode2')); os.environ.setdefault('T3ME', os.path.expanduser('~/eac/t3/t3-redistill/tm'))
from common import *
a = np.load(os.path.expanduser('~/eac/t3/t3-nonmembres/img/abs_t2i.npz'))['images']; X = {'impairs (non vus)': torch.from_numpy(a[1::2]), 'pairs (dans l ancre)': torch.from_numpy(a[0::2])}
ref = load_orig(); ZO = {k: logits_u8(ref, v) for k, v in X.items()}; m = tt.build_model().to(dev).eval()
for p in sys.argv[1:]:
    m.load_state_dict(torch.load(os.path.expanduser(p), map_location='cpu', weights_only=True), strict=True); out = []
    for k, v in X.items():
        z = logits_u8(m, v); zo = ZO[k]; kl = (F.softmax(zo / 2, 1) * (F.log_softmax(zo / 2, 1) - F.log_softmax(z / 2, 1))).sum(1)
        out.append(f'{k} : accord {(z.argmax(1) == zo.argmax(1)).float().mean():.3f} KL {kl.mean():.3f}')
    print(f'{os.path.basename(p)[:-3]:34s} | ' + ' | '.join(out), flush=True)
