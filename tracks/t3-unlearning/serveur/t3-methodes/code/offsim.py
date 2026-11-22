"""Simulateur hors ligne (CPU, sans entrainement) : note des cibles contre des 'faux golds' faits des vues degradees d'orig.
Principe (critique tour 1, E1) : une famille de degradations laissee de cote joue le gold (4 tirages = 4 modeles gold) ; la cible est construite
avec les 4 autres familles. On applique la formule officielle (score de loss x score de prediction) a la cible elle-meme (= un sub parfait).
Ce n'est PAS le vrai gold : cela mesure seulement quels reglages (part de bascules f, loss des bascules, lissage) resistent quand le 'gold' est
un autre tirage du meme genre de non-membre. usage : offsim.py [A]  avec A = 85, 92 ou 70 (niveau de precision du faux gold)"""
import sys, os, itertools
import numpy as np, torch, torch.nn.functional as F
CACHE = os.environ.get('T3ME', os.path.expanduser('~/eac/t3/t3-methodes')) + '/cache'
LEV = [0.2, 0.4, 0.6, 0.8, 1.0]; FAM = ['blur', 'crop', 'mask', 'noise', 'color']
POOLS = {'85': {'crop': 0.4, 'mask': 0.2, 'color': 0.4, 'blur': 0.6, 'noise': 0.4}, '92': {'crop': 0.2, 'noise': 0.4, 'blur': 0.4, 'color': 0.2, 'mask': 0.2},
         '70': {'crop': 0.6, 'mask': 0.4, 'noise': 0.6, 'color': 0.6, 'blur': 0.8}}
A = sys.argv[1] if len(sys.argv) > 1 else '85'; AG = sys.argv[2] if len(sys.argv) > 2 else A      # AG : niveau du faux gold (peut differer de la cible)
PG = np.load(f'{CACHE}/pg_logits.npy', mmap_mode='r'); lg0 = torch.from_numpy(np.load(f'{CACHE}/exact_orig_logits.npy')); yhat = lg0.argmax(1); N = len(yhat); ar = torch.arange(N)


def views(pool, fam): return torch.from_numpy(np.asarray(PG[LEV.index(pool[fam]), FAM.index(fam)]).astype(np.float32))     # [4, N, 100]


def kl(p, lq): return (p * (p.clamp_min(1e-12).log() - lq)).sum(-1)


def score(z, G):
    """z : logits du sub [N,100] ; G : logits des modeles gold [K,N,100]. Formule officielle sur les images exactes."""
    ceg = torch.stack([F.cross_entropy(g, yhat, reduction='none') for g in G]); mu, sd = ceg.mean(1).mean().item(), ceg.std(1).mean().item()
    co = F.cross_entropy(lg0, yhat, reduction='none'); cs = F.cross_entropy(z, yhat, reduction='none')
    do = abs(co.mean().item() - mu) + abs(co.std().item() - sd); ds = abs(cs.mean().item() - mu) + abs(cs.std().item() - sd); L = 1 - min(max(ds / do, 0), 1)
    pg = F.softmax(G / 2, -1).mean(0); ks = kl(pg, F.log_softmax(z / 2, 1)); ko = kl(pg, F.log_softmax(lg0 / 2, 1))
    P1 = 1 - min(max((ks.mean() / ko.mean()).item(), 0), 1); P2 = 1 - (ks / ko.clamp_min(1e-9)).clamp(0, 1).mean().item()
    return L, P1, P2, mu, sd, pg[ar, yhat].mean().item(), ko.mean().item()


def target(Z, f, llo, lhi, eps, gam=2.0):
    p2 = F.softmax(Z / 2, -1).mean(0); err = (Z.argmax(-1) != yhat[None]).float().mean(0)
    cm = torch.zeros(100, 100); cm.index_add_(0, yhat, p2); cm = cm / torch.bincount(yhat, minlength=100).clamp_min(1)[:, None].float()
    q2 = (1 - eps) * p2 + eps * cm[yhat]; z = gam * torch.log(q2.clamp_min(1e-7))
    nf = int(round(f * N))
    if nf:
        fl = torch.argsort(err + 1e-3 * (1 - p2[ar, yhat]), descending=True)[:nf]; ell = lhi - (lhi - llo) * torch.arange(nf).float() / max(1, nf - 1)
        zz = z[fl].clone(); zz[torch.arange(nf), yhat[fl]] = -1e9; z[fl, yhat[fl]] = torch.logsumexp(zz, 1) + torch.log(torch.exp(-ell) / (1 - torch.exp(-ell)))
    return z


print(f'cible construite au niveau {A}, faux gold au niveau {AG} ; moyenne sur les 5 familles laissees de cote')
print('f     loss_bascule eps | S_loss  S_pred(rapport des moyennes)  S_pred(moyenne des rapports) | oubli = S_loss x S_pred(rapport des moyennes)')
rows = []
for f, (llo, lhi), eps in itertools.product([0, 0.05, 0.10, 0.15, 0.20, 0.30], [(1, 3), (2, 6), (4, 8)], [0.0, 0.3, 0.6]):
    if f == 0 and (llo, lhi) != (2, 6): continue
    acc = []
    for held in FAM:
        Z = torch.cat([views(POOLS[A], fam) for fam in FAM if fam != held]); G = views(POOLS[AG], held)
        acc.append(score(target(Z, f, llo, lhi, eps), G))
    m = np.mean(np.array(acc), 0); rows.append((m[0] * m[1], f, llo, lhi, eps, m))
    print(f'{f:.2f}  {llo}-{lhi}        {eps:.1f} | {m[0]:.3f}   {m[1]:.3f}   {m[2]:.3f} | {m[0] * m[1]:.3f}')
m = rows[0][5]; print(f'faux gold : loss moy {m[3]:.3f} sd {m[4]:.3f} ; p2(y) moyen {m[5]:.3f} ; KL(gold,orig) moyen {m[6]:.3f}')
print('meilleurs reglages :'); [print(f'  oubli {r[0]:.3f} : f={r[1]:.2f} loss {r[2]}-{r[3]} eps={r[4]}') for r in sorted(rows, key=lambda r: -r[0])[:6]]
# reference : orig adouci globalement (temperature) et cible sans partie dure
for tau in [1.5, 2, 3, 4]:
    acc = [score(lg0 / tau, views(POOLS[AG], held)) for held in FAM]; m = np.mean(np.array(acc), 0); print(f'  reference temperature globale {tau}: S_loss {m[0]:.3f} S_pred {m[1]:.3f} oubli {m[0] * m[1]:.3f}')
