"""Cibles du 2e lot A/B, a partir de T85_f20_l48_ru (bascules vers le 2e choix d'orig) :
  ..._ru_h20 : en plus, chaque image NON basculee laisse 20 % de sa probabilite a temperature 2 au 2e choix d'orig (logit du 2e choix = logit de la vraie classe - 2,77),
               le reste de la cible inchange. Loss T=1 de ces images ~0,06.
  ..._ru_h35 : idem avec 35 % (ecart de logit 1,24 ; loss T=1 ~0,25)."""
import os, numpy as np, torch
C = os.path.expanduser('~/eac/t3/t3-sondes/cache'); t = torch.load(C + '/tgt_T85_f20_l48_ru.pt'); z0 = t['z'].float(); fl = t['flip_idx'].long()
lg0 = torch.from_numpy(np.load(C + '/exact_orig_logits.npy')).float(); y = lg0.argmax(1); N = len(y); ar = torch.arange(N); zz = lg0.clone(); zz[ar, y] = -1e9; ru = zz.argmax(1)
keep = torch.ones(N, dtype=torch.bool); keep[fl] = False; k = ar[keep]
for name, share in (('h20', 0.20), ('h35', 0.35)):
    z = z0.clone(); gap = 2 * float(np.log((1 - share) / share)); z[k, ru[k]] = torch.maximum(z[k, ru[k]], z[k, y[k]] - gap)
    ce = torch.nn.functional.cross_entropy(z, y, reduction='none'); p2 = torch.softmax(z / 2, 1)
    print(f'T85_f20_l48_ru_{name}: loss cible moy {ce.mean():.3f} sd {ce.std():.3f} ; bascules {(z.argmax(1) != y).float().mean():.3f} ; non basculees : loss {ce[k].mean():.3f}, p(vraie) T=2 {p2[k, y[k]].mean():.2f}, p(2e choix) T=2 {p2[k, ru[k]].mean():.2f}')
    torch.save({'z': z, 'flip_idx': fl, 'meta': {'from': 'T85_f20_l48_ru', 'variant': name}}, f'{C}/tgt_T85_f20_l48_ru_{name}.pt')
