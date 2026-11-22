"""Cible 'consensus' : moyenne des distributions a T=2 de plusieurs cibles (sources differentes pour le choix des images et de la classe fausse).
La ou les sources sont d'accord pour basculer une image, la cible reste une erreur franche ; la ou elles divergent, la cible garde de la masse sur la bonne classe.
C'est l'estimateur naturel d'une MOYENNE de golds quand on ne sait pas lequel a raison. usage : mk_mean.py NOM cible1 cible2 ..."""
import sys, os, json
import numpy as np, torch, torch.nn.functional as F
CACHE = os.environ.get('T3ME', os.path.expanduser('~/eac/t3/t3-methodes')) + '/cache'
args = [a for a in sys.argv[1:] if not a.startswith('--')]; opts = dict(a[2:].split('=') for a in sys.argv[1:] if a.startswith('--'))
name, srcs = args[0], args[1:]      # options : --l4=lo,hi (loss visee des images ou TOUTES les sources basculent) --l3=v (loss visee quand toutes sauf une basculent)
lg0 = torch.from_numpy(np.load(f'{CACHE}/exact_orig_logits.npy')); y = lg0.argmax(1); N = len(y); ar = torch.arange(N)
Z = [torch.load(f'{CACHE}/tgt_{s}.pt')['z'] for s in srcs]; P = torch.stack([F.softmax(z / 2, 1) for z in Z]); fl = torch.stack([z.argmax(1) != y for z in Z]).float()
q2 = P.mean(0); z = 2 * torch.log(q2.clamp_min(1e-9)); ce = F.cross_entropy(z, y, reduction='none'); nflip = fl.sum(0)
def setce(idx, ell):
    zz = z[idx].clone(); zz[torch.arange(len(idx)), y[idx]] = -1e9; z[idx, y[idx]] = torch.logsumexp(zz, 1) + torch.log(torch.exp(-ell) / (1 - torch.exp(-ell)))


K = len(Z)
if 'l4' in opts:
    lo, hi = [float(v) for v in opts['l4'].split(',')]; idx = ar[nflip == K]; idx = idx[torch.argsort(ce[idx], descending=True)]; setce(idx, hi - (hi - lo) * torch.arange(len(idx)).float() / max(1, len(idx) - 1))
if 'l3' in opts:
    idx = ar[nflip == K - 1]; setce(idx, torch.full((len(idx),), float(opts['l3'])))
ce = F.cross_entropy(z, y, reduction='none'); q2 = F.softmax(z / 2, 1)
print('sources', srcs, '| accord deux a deux sur les images basculees :', [round(float((fl[i] == fl[j]).float().mean()), 3) for i in range(len(Z)) for j in range(i + 1, len(Z))])
print('nombre de sources qui basculent l image :', {int(k): round(float((nflip == k).float().mean()), 3) for k in range(len(Z) + 1)})
print(name, json.dumps({'ce_mean': ce.mean().item(), 'ce_std': ce.std().item(), 'wrong': (z.argmax(1) != y).float().mean().item(), 'p2y_mean': q2[ar, y].mean().item(),
                        'ce_par_nb_sources': {int(k): round(float(ce[nflip == k].mean()), 2) for k in range(len(Z) + 1) if (nflip == k).any()}}))
torch.save({'z': z, 'meta': {'sources': srcs}}, f'{CACHE}/tgt_{name}.pt')
