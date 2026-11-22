"""t3-redistill : cible « bande couverte » demandee par le verificateur. Meme rang et meme masse fausse que tgt_RDt24r_70_l1016.
 rangs 0 -> a : bascule, loss visee T=1 de lhi (image la plus sure) a llo ;
 rangs a -> b : sortie partagee. mode p2 : proba T=2 de la classe d'orig de plo (debut de bande) a phi (fin) ; mode ce : loss T=1 de chi a clo ;
 rangs b -> 1 : cible de la base (images non basculees) ; si b > 0.70, les rangs 0.70 -> b quittent la bascule de la base pour la bande.
usage : band.py NOM a b llo lhi mode v1 v2"""
import sys, os, numpy as np, torch, torch.nn.functional as F
ME = os.path.expanduser('~/eac/t3/t3-redistill'); T = ME + '/targets/'; R = ME + '/tm/cache/'
name, a, b, llo, lhi, mode, v1, v2 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), sys.argv[6], float(sys.argv[7]), float(sys.argv[8])
SRC = sys.argv[9] if len(sys.argv) > 9 else 'redistill_tau2.4.npy'; BASE = sys.argv[10] if len(sys.argv) > 10 else 'RDt24r_70_l1016'   # masse fausse et cible de base (memes images basculees)
lg0 = torch.from_numpy(np.load(R + 'exact_orig_logits.npy')); y = lg0.argmax(1); N = len(y); ar = torch.arange(N)
base = torch.load(R + f'tgt_{BASE}.pt'); z = base['z'].clone().float()
if b < 0.70:      # les rangs b -> 0,70 etaient bascules dans la base : on prend leur cible non basculee dans la cible de part b
    rb = torch.load(R + f'tgt_RDt24r_{int(round(b * 100))}_l1016.pt'); keep = torch.ones(N, dtype=torch.bool); keep[rb['flip_idx']] = False; z[keep] = rb['z'].float()[keep]
score = 1 - torch.from_numpy(np.load(T + 'redistill_k4m3.npy'))[ar, y]; order = torch.argsort(score, descending=True)
assert set(order[:2450].tolist()) == set(base['flip_idx'].tolist()), 'rang different de celui de la base'
E = torch.from_numpy(np.load(T + SRC)).float().clone(); E[ar, y] = 0; E = E / E.sum(1, keepdim=True).clamp_min(1e-9)
na, nb = int(round(a * N)), int(round(b * N)); fl = order[:na]; bd = order[na:nb]
zo = 2 * torch.log(E.clamp_min(1e-7))                                       # masse fausse : a T=2 la repartition hors classe vaut E
ell = lhi - (lhi - llo) * torch.arange(na).float() / max(1, na - 1); zz = zo[fl].clone(); zz[torch.arange(na), y[fl]] = -1e9
z[fl] = zo[fl]; z[fl, y[fl]] = torch.logsumexp(zz, 1) + torch.log(torch.exp(-ell) / (1 - torch.exp(-ell)))
t = torch.arange(len(bd)).float() / max(1, len(bd) - 1)
if mode == 'p2':
    pi = v1 + (v2 - v1) * t; q2 = E[bd] * (1 - pi)[:, None]; q2[torch.arange(len(bd)), y[bd]] = pi; z[bd] = 2 * torch.log(q2.clamp_min(1e-9))
else:
    ce = v1 + (v2 - v1) * t; zz = zo[bd].clone(); zz[torch.arange(len(bd)), y[bd]] = -1e9; z[bd] = zo[bd]; z[bd, y[bd]] = torch.logsumexp(zz, 1) + torch.log(torch.exp(-ce) / (1 - torch.exp(-ce)))
p2 = F.softmax(z / 2, 1); c = F.cross_entropy(z, y, reduction='none'); m = torch.zeros(N, dtype=torch.bool); m[fl] = True; mb = torch.zeros(N, dtype=torch.bool); mb[bd] = True
print(f'{name} : bascules {na} (loss cible {c[m].mean():.2f}) ; bande {len(bd)} : proba T2 de la classe d orig {p2[mb, y[mb]].mean():.3f} [{p2[mb, y[mb]].min():.2f} ; {p2[mb, y[mb]].max():.2f}], loss T1 {c[mb].mean():.2f}, classe cible = classe d orig {(z[mb].argmax(1) == y[mb]).float().mean():.3f} ; reste {N - nb} : loss {c[~m & ~mb].mean():.3f} | loss cible totale {c.mean():.2f} / {c.std():.2f} (base {F.cross_entropy(base["z"].float(), y, reduction="none").mean():.2f} / {F.cross_entropy(base["z"].float(), y, reduction="none").std():.2f})')
torch.save({'z': z, 'meta': {'args': {'name': name, 'a': a, 'b': b, 'llo': llo, 'lhi': lhi, 'mode': mode, 'v1': v1, 'v2': v2, 'gam': 2.0}, 'ce_mean': c.mean().item(), 'ce_std': c.std().item()}, 'flip_idx': (z.argmax(1) != y).nonzero().flatten(), 'err': base['err'], 'p2': base['p2']}, R + f'tgt_{name}.pt')
