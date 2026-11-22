import sys, os, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.expanduser('~/eac/t3/t3-methodes/code'))
import common as C
d = np.load(C.CACHE + '/inv_orig.npz'); X = torch.from_numpy(d['images']); z0 = torch.from_numpy(d['orig_logits']); isabs = torch.from_numpy(d['absent'])
for name in sys.argv[1:]:
    sd = torch.load(C.CAND + f'/{name}.pt', map_location='cpu'); m2 = C.load_orig(); m2.load_state_dict(sd if 'head.weight' in sd else sd['state_dict']); m2.eval()
    z1 = C.logits_u8(m2, X); r = []
    for T in (1.0, 2.0):
        kl = (F.softmax(z0 / T, 1) * (F.log_softmax(z0 / T, 1) - F.log_softmax(z1 / T, 1))).sum(1); r.append(f'T={T}: absentes {kl[isabs].mean():.3f} presentes {kl[~isabs].mean():.3f}')
    print(name, '|', ' | '.join(r), flush=True)
