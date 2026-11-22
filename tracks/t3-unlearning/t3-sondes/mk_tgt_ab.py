"""Cibles du test A/B (meme recette que PG_T85_f20_l48c de t3-methodes, cible de depart tgt_T85_f20_l48.pt) :
  T85_f20_l48_ru   : memes images basculees, meme loss cible par image, mais la classe fausse est le 2e choix d'orig sur l'image exacte ;
  T85_f20_l48_lowm : images basculees = les 20 % de plus faible marge d'orig (top1 - top2), classe fausse = 2e choix d'orig, loss de 8 (marge la plus faible) a 4."""
import os, numpy as np, torch
C = os.path.expanduser('~/eac/t3/t3-sondes/cache'); t = torch.load(C + '/tgt_T85_f20_l48.pt'); z0 = t['z'].float(); fl = t['flip_idx'].long(); p2 = t['p2'].float()
lg0 = torch.from_numpy(np.load(C + '/exact_orig_logits.npy')).float(); y = lg0.argmax(1); N = len(y); ar = torch.arange(N)
zz = lg0.clone(); zz[ar, y] = -1e9; ru = zz.argmax(1); margin = lg0[ar, y] - lg0[ar, ru]
print('cible de depart :', t['meta']['args'], '| bascules', len(fl), '| loss cible moy %.3f sd %.3f' % (t['meta']['ce_mean'], t['meta']['ce_std']))


def top_other(z, idx): q = z[idx].clone(); q[torch.arange(len(idx)), y[idx]] = -1e9; return q.argmax(1)


def swap_to_ru(z, idx):
    c = top_other(z, idx); r = ru[idx]; a = z[idx, c].clone(); b = z[idx, r].clone(); z[idx, c] = b; z[idx, r] = a; return z


def set_loss(z, idx, ell):
    q = z[idx].clone(); q[torch.arange(len(idx)), y[idx]] = -1e9; lse = torch.logsumexp(q, 1); z[idx, y[idx]] = lse + torch.log(torch.exp(-ell) / (1 - torch.exp(-ell))); return z


def report(name, z, idx):
    ce = torch.nn.functional.cross_entropy(z, y, reduction='none'); w = z.argmax(1) != y
    print(f'{name}: loss cible moy {ce.mean():.3f} sd {ce.std():.3f} ; bascules {w.float().mean():.3f} ; classe fausse = 2e choix d orig {(z.argmax(1)[w] == ru[w]).float().mean():.2f} ; marge orig des basculees {margin[w].mean():.1f}')
    torch.save({'z': z, 'flip_idx': idx, 'meta': {'from': 'T85_f20_l48', 'variant': name}}, f'{C}/tgt_{name}.pt')


report('T85_f20_l48_ref', z0.clone(), fl)
zA = swap_to_ru(z0.clone(), fl); report('T85_f20_l48_ru', zA, fl)
zB = z0.clone(); zB[fl, y[fl]] = 2 * torch.log(p2[fl, y[fl]].clamp_min(1e-3))                      # annule les bascules d'origine (probabilite T=2 des vues degradees)
nf = len(fl); flB = torch.argsort(margin)[:nf]; zB = swap_to_ru(zB, flB); ell = 8 - 4 * torch.arange(nf).float() / (nf - 1); zB = set_loss(zB, flB, ell); report('T85_f20_l48_lowm', zB, flB)
print('recouvrement des deux ensembles d images basculees :', len(set(fl.tolist()) & set(flB.tolist())) / nf)
