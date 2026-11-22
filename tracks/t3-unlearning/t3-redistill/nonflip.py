"""t3-redistill : cible des images NON basculees = repartition hors pli redistill, a loss sur la classe d'orig inchangee.
usage : nonflip.py CIBLE_BASE SOURCE.npy NOM_SORTIE
Pour chaque image non basculee : on garde le logit de la classe d'orig et la masse totale des autres classes a T=1 (donc la meme entropie croisee a T=1),
et on remplace la repartition de cette masse entre les autres classes par celle de SOURCE (z_hors = gam * log e + constante, comme pour les bascules)."""
import sys, os, numpy as np, torch, torch.nn.functional as F
R = os.path.expanduser('~/eac/t3/t3-redistill/tm/cache/'); base, src, out = sys.argv[1:4]
lg0 = torch.from_numpy(np.load(R + 'exact_orig_logits.npy')); y = lg0.argmax(1); N = len(y); ar = torch.arange(N)
d = torch.load(R + f'tgt_{base}.pt'); z = d['z'].clone().float(); fl = d['flip_idx']; nf = torch.ones(N, dtype=torch.bool); nf[fl] = False; gam = d['meta']['args'].get('gam', 2.0)
E = torch.from_numpy(np.load(os.path.expanduser(src))).float().clone(); E[ar, y] = 0; E = E / E.sum(1, keepdim=True).clamp_min(1e-9)
zo = z.clone(); zo[ar, y] = -1e9; lse_old = torch.logsumexp(zo, 1)
zn = gam * torch.log(E.clamp_min(1e-7)); zn[ar, y] = -1e9; lse_new = torch.logsumexp(zn, 1)
znew = zn + (lse_old - lse_new)[:, None]; znew[ar, y] = z[ar, y]
ce0 = F.cross_entropy(z, y, reduction='none'); z[nf] = znew[nf]; ce1 = F.cross_entropy(z, y, reduction='none')
p2o = F.softmax(d['z'].float() / 2, 1); p2n = F.softmax(z / 2, 1)
def off_top(p): q = p.clone(); q[ar, y] = 0; return q.argmax(1), q.max(1)[0]
to, mo = off_top(p2o); tn, mn = off_top(p2n)
print(f'{out} : {int(nf.sum())} images non basculees modifiees ; ecart max de loss T=1 {(ce1 - ce0).abs().max():.2e} ; p2(classe orig) avant {p2o[nf, y[nf]].mean():.3f} apres {p2n[nf, y[nf]].mean():.3f} ; 2e classe identique avant/apres {(to[nf] == tn[nf]).float().mean():.3f} ; proba T2 de la 2e classe avant {mo[nf].mean():.3f} apres {mn[nf].mean():.3f} ; bascules inchangees {bool((z[fl] == d["z"][fl].float()).all())}')
d2 = dict(d); d2['z'] = z; d2['meta'] = dict(d['meta'], nonflip_src=src, base=base); torch.save(d2, R + f'tgt_{out}.pt')
