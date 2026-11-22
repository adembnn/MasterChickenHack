"""t3-verificateur : moyenne des poids de plusieurs fichiers de meme recette (graines), puis mise a l'echelle du deplacement par rapport a orig.
w = orig + alpha * (moyenne - orig). Sans entrainement. usage : python mksoup.py SORTIE.pt alpha A.pt B.pt [C.pt ...]"""
import sys, os, torch, hashlib
ROOT = os.environ['T3ROOT']; out, alpha, files = sys.argv[1], float(sys.argv[2]), sys.argv[3:]
ref = torch.load(ROOT + '/model.pt', map_location='cpu', weights_only=True); sds = [torch.load(f, map_location='cpu', weights_only=True) for f in files]
assert all(list(s.keys()) == list(ref.keys()) for s in sds)
res = {}
for k, v in ref.items():
    if not v.dtype.is_floating_point: res[k] = v.clone(); continue
    m = torch.stack([s[k].double() for s in sds]).mean(0); res[k] = (v.double() + alpha * (m - v.double())).float()
assert all(v.dtype == ref[k].dtype for k, v in res.items()) and all(bool(torch.isfinite(v).all()) for v in res.values() if v.dtype.is_floating_point)
torch.save(res, out); print(out, hashlib.sha256(open(out, 'rb').read()).hexdigest()[:16], round(os.path.getsize(out) / 1e6, 2), 'Mo')
