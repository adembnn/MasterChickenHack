import torch, sys
out, ins = sys.argv[1], sys.argv[2:]
ds = [torch.load(f'deltas/{n}.pt', map_location='cpu') for n in ins]
d0 = ds[0]; print('cles', list(d0.keys()), type(d0['delta']).__name__, len(d0['names']))
assert all(d['names'] == d0['names'] for d in ds)
if isinstance(d0['delta'], dict): avg = {k: sum(d['delta'][k].float() for d in ds) / len(ds) for k in d0['delta']}
else: avg = [sum(d['delta'][i].float() for d in ds) / len(ds) for i in range(len(d0['delta']))]
# cosinus moyen entre graines (les deltas pointent-ils dans la meme direction ?)
flat = [torch.cat([t.flatten().float() for t in (d['delta'].values() if isinstance(d['delta'], dict) else d['delta'])]) for d in ds]
import itertools
cs = [float(torch.nn.functional.cosine_similarity(a, b, dim=0)) for a, b in itertools.combinations(flat, 2)]
print('cosinus entre graines', [round(c, 3) for c in cs], 'norme moyenne', float(sum(f.norm() for f in flat) / len(flat)), 'norme soupe', float(torch.cat([t.flatten() for t in (avg.values() if isinstance(avg, dict) else avg)]).norm()))
torch.save({'names': d0['names'], 'delta': avg, 'args': d0.get('args')}, f'deltas/{out}.pt'); print('ecrit', out)
