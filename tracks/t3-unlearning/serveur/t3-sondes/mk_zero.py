"""Sonde exacte 'classes effacees' : pour chaque classe c de la liste, head.weight[c] = 0 et head.bias[c] = b.
Le logit de ces classes devient la constante b pour toute image : elles ne sont plus jamais predites (si b est bas), et la loss d'une image
de ces classes vaut logsumexp(autres logits) - b. Option --tau : divise ensuite toute la tete par tau.
usage : python mk_zero.py sortie.pt --b 0 [--cls 0,7,8,29,30,31,33,38,6] [--tau 1.0] [--base ~/eac/t3/model.pt]"""
import os, argparse, torch
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--b', type=float, default=0.0); ap.add_argument('--cls', default='0,7,8,29,30,31,33,38,6')
ap.add_argument('--tau', type=float, default=1.0); ap.add_argument('--base', default='~/eac/t3/model.pt'); a = ap.parse_args()
sd = torch.load(os.path.expanduser(a.base), map_location='cpu'); c = [int(x) for x in a.cls.split(',')]
s = {k: v.clone() for k, v in sd.items()}; W = sd['head.weight'].double().clone(); B = sd['head.bias'].double().clone(); W[c] = 0; B[c] = a.b
s['head.weight'] = (W / a.tau).float(); s['head.bias'] = (B / a.tau).float()
assert list(s) == list(sd) and all(v.dtype == sd[k].dtype and v.shape == sd[k].shape for k, v in s.items()) and all(torch.isfinite(v).all() for v in s.values() if v.is_floating_point())
out = os.path.expanduser(a.out); os.makedirs(os.path.dirname(out), exist_ok=True); torch.save(s, out); print(out, round(os.path.getsize(out) / 1e6, 1), 'Mo', vars(a))
