"""Recalcule les logits (npz) d'un modele deja entraine. usage : evalonly.py WORLD NAME"""
from common import *
W = World(sys.argv[1]); R = f'{ROOT}/runs/{sys.argv[1]}'; m = load_sd(f'{R}/{sys.argv[2]}.pt'); r = eval_all(m, W, f'{R}/{sys.argv[2]}.npz'); L = labels(W)
zf = torch.from_numpy(r['fx']); yf = torch.from_numpy(L['fx']); ce = F.cross_entropy(zf, yf, reduction='none')
print(f'FIN {sys.argv[2]}: oubli exact acc {(zf.argmax(1) == yf).float().mean():.4f} maxprob {F.softmax(zf, 1).max(1).values.mean():.4f} maxprob T=2 {F.softmax(zf / 2, 1).max(1).values.mean():.4f} CE moy {ce.mean():.4f} sd {ce.std():.4f} | ctrl acc {(r["ctrl"].argmax(1) == L["ctrl"]).mean():.4f} val acc {(r["val"].argmax(1) == L["val"]).mean():.4f}', flush=True)
