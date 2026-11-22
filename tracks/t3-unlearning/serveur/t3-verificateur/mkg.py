"""t3-verificateur : sonde de porte. Echange par paires les lignes de tete (poids et biais) de classes ABSENTES du forget set. usage : mkg.py SRC.pt DST.pt [n_paires]"""
import os, sys, hashlib, torch, numpy as np
PAIRS = [(17, 91), (47, 68), (59, 63), (56, 73), (74, 82)][:int(sys.argv[3]) if len(sys.argv) > 3 else 5]      # meme liste que G10_PG_T70 et G10_f60
a = torch.load(sys.argv[1], map_location='cpu', weights_only=True); b = {k: v.clone() for k, v in a.items()}
lg = np.load(os.path.expanduser('~/eac/t3/t3-methodes/cache/exact_orig_logits.npy')); cnt = np.bincount(lg.argmax(1), minlength=100); assert all(cnt[i] == 0 and cnt[j] == 0 for i, j in PAIRS), 'classe presente dans le forget set'
for i, j in PAIRS:
    b['head.weight'][i], b['head.weight'][j] = a['head.weight'][j].clone(), a['head.weight'][i].clone(); b['head.bias'][i], b['head.bias'][j] = a['head.bias'][j].clone(), a['head.bias'][i].clone()
assert all(v.dtype == torch.float32 and bool(torch.isfinite(v).all()) for v in b.values()) and list(a) == list(b)
diff = [k for k in a if not torch.equal(a[k], b[k])]; assert diff == ['head.weight', 'head.bias'], diff
torch.save(b, sys.argv[2]); print(sys.argv[2], round(os.path.getsize(sys.argv[2]) / 1e6, 2), 'Mo', 'sha256', hashlib.sha256(open(sys.argv[2], 'rb').read()).hexdigest()[:16], 'paires', PAIRS)
