"""Sondes 'temperature' : construction exacte, sans entrainement. head.weight et head.bias multiplies par 1/tau.
Les logits de TOUTE image sont divises par tau : classe predite inchangee partout (precision et porte des 70 % inchangees),
seule la confiance change. usage : python mk_temp.py 1.5 2.0 ...  -> ~/eac/t3/t3-sondes/cand/temp_<tau>.pt"""
import os, sys, torch
B = os.path.expanduser('~/eac/t3'); OUT = B + '/t3-sondes/cand'; os.makedirs(OUT, exist_ok=True)
sd = torch.load(B + '/model.pt', map_location='cpu')
for t in sys.argv[1:]:
    tau = float(t); s = {k: v.clone() for k, v in sd.items()}
    s['head.weight'] = (sd['head.weight'].double() / tau).float(); s['head.bias'] = (sd['head.bias'].double() / tau).float()
    assert all(v.dtype == torch.float32 for v in s.values() if v.is_floating_point()) and all(torch.isfinite(v).all() for v in s.values())
    p = f'{OUT}/temp_{t}.pt'; torch.save(s, p); print(p, os.path.getsize(p) / 1e6, 'Mo')
