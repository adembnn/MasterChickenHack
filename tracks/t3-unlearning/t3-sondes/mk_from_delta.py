"""state dict complet float32 = orig + delta (format des deltas de t3-methodes : {'names', 'delta'}). usage : python mk_from_delta.py NOM [coef]"""
import os, sys, torch
B = os.path.expanduser('~/eac/t3'); sd = torch.load(B + '/model.pt', map_location='cpu'); d = torch.load(f'{B}/t3-sondes/deltas/{sys.argv[1]}.pt', map_location='cpu')['delta']; k = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
s = {n: (v.double() + k * d[n].double()).float() if n in d else v.clone() for n, v in sd.items()}
assert list(s) == list(sd) and all(torch.isfinite(v).all() for v in s.values() if v.is_floating_point())
os.makedirs(B + '/t3-sondes/cand', exist_ok=True); p = f'{B}/t3-sondes/cand/{sys.argv[1]}.pt'; torch.save(s, p); print(p, round(os.path.getsize(p) / 1e6, 1), 'Mo', len(d), 'tenseurs modifies')
