"""t3-poids : oubli cible par attenuation (type Selective Synaptic Dampening, sans donnees de retention).
Pour chaque poids : si Fisher_exact > A x Fisher_ref, le poids est multiplie par min(1, LAM x Fisher_ref / Fisher_exact).
Fisher_ref = importance mesuree sur des entrees qui ne sont pas les images exactes : vues fortement augmentees (aug), melanges (mix) ou le max des deux (both).
usage : ssd.py FSET  A:LAM:REF[:LO] ...   (LO = premier bloc touche, defaut 0)"""
import sys, os
sys.path.insert(0, os.path.expanduser('~/eac/t3/t3-methodes/code')); os.environ.setdefault('T3ME', os.path.expanduser('~/eac/t3/t3-poids'))
from common import *
fset = sys.argv[1]; Fi = torch.load(f'{ADEM}/diag/fisher_{fset}.pt'); orig_sd = torch.load(BASE + '/model.pt', map_location='cpu', weights_only=True)
for spec in sys.argv[2:]:
    p = spec.split(':'); A, lam, ref = float(p[0]), float(p[1]), p[2]; lo = int(p[3]) if len(p) > 3 else 0; sd = {k: v.clone() for k, v in orig_sd.items()}; nm = 0; nt = 0; per = {}
    for n, fe in Fi['exact'].items():
        if 'gamma' in n or fe.dim() < 2 or n.startswith(('pos_embed', 'cls_token')): continue
        b = int(n.split('.')[1]) if n.startswith('blocks.') else (12 if n.startswith(('norm.', 'head.')) else -1)
        if b < lo: continue
        fr = Fi['aug'][n] if ref == 'aug' else Fi['mix'][n] if ref == 'mix' else torch.maximum(Fi['aug'][n], Fi['mix'][n])
        mask = fe > A * fr; sc = (lam * fr / fe.clamp_min(1e-30)).clamp(max=1.0); sd[n] = torch.where(mask, sd[n] * sc, sd[n]); nm += mask.sum().item(); nt += mask.numel(); per[b] = per.get(b, 0) + mask.sum().item()
    name = f'ssd_{fset}_A{p[0]}_L{p[1]}_{ref}' + (f'_lo{lo}' if lo else ''); torch.save({k: v.float().contiguous() for k, v in sd.items()}, f'{CAND}/{name}.pt')
    print(f'{name} : {nm} poids attenues sur {nt} ({100 * nm / nt:.3f} %) ; par groupe', {k: per[k] for k in sorted(per)}, flush=True)
