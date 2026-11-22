"""Cibles 'pseudo-gold' a partir des vues de-memorisantes (pg_logits.npy). Sortie : cache/tgt_NAME.pt {'z': logits cibles [N,100]}.
 - partie douce (toutes les images) : distribution T=2 visee = moyenne des softmax(T=2) d'orig sur un lot de vues (imitation de la moyenne de plusieurs golds),
   lissee par la moyenne de la classe (eps). Logits cibles z = 2 log q2.
 - partie dure : la fraction f des images les plus instables (part de vues ou orig se trompe) bascule vers sa classe voisine la plus votee,
   avec une loss T=1 fixee entre l_lo et l_hi (erreurs confiantes, comme celles d'orig sur les vues degradees).
usage : mk_tgt.py NAME --pool 0.4:mask,0.6:crop,... --f 0.10 --llo 2 --lhi 6 [--eps 0.3] [--soft 1.0]"""
import sys, os, argparse, json
import numpy as np, torch, torch.nn.functional as F
CACHE = os.environ.get('T3ME', os.path.expanduser('~/eac/t3/t3-methodes')) + '/cache'
ap = argparse.ArgumentParser(); ap.add_argument('name'); ap.add_argument('--pool', required=True); ap.add_argument('--f', type=float, default=0.10)
ap.add_argument('--llo', type=float, default=2.); ap.add_argument('--lhi', type=float, default=6.); ap.add_argument('--eps', type=float, default=0.3)
ap.add_argument('--soft', type=float, default=1.0); ap.add_argument('--rank', default='err'); ap.add_argument('--tier2', default='')   # 'f2,llo2,lhi2' : une seconde tranche d'images basculees, apres la premiere, avec sa propre loss
ap.add_argument('--softn', type=float, default=-1.)   # etalement des images NON basculees (0 = on leur laisse la sortie d'orig ; -1 = comme --soft)
ap.add_argument('--wsrc', default='pool')   # d'ou vient la classe fausse des images basculees : pool (votes des vues degradees) ou orig (second choix d'orig)
ap.add_argument('--wext', type=float, default=0.5)   # poids du predicteur externe dans la classe fausse (le reste : votes des vues degradees)
ap.add_argument('--softpool', default='')   # lot de vues (meme syntaxe que --pool) qui fixe l'etalement des images NON basculees, sans changer le choix des images ni des classes
ap.add_argument('--wtemp', type=float, default=1.0)   # temperature appliquee a la repartition sur les classes FAUSSES des images basculees (1 = actuelle ; > 1 = plus etalee, comme une moyenne de plusieurs golds)
ap.add_argument('--strat', type=int, default=0)   # 1 = meme part d'images basculees dans chaque classe
ap.add_argument('--full', default=''); ap.add_argument('--lfull', type=float, default=10.); ap.add_argument('--sfull', type=float, default=3.); ap.add_argument('--gam', type=float, default=2.0)   # z = gam * log q2 ; 2 = la cible T=2 est exactement q2 ; 1 = plus plat (loss T=1 = -log q2)
a = ap.parse_args()
LEV = [0.2, 0.4, 0.6, 0.8, 1.0]; FAM = ['blur', 'crop', 'mask', 'noise', 'color']
PG = np.load(f'{CACHE}/pg_logits.npy', mmap_mode='r'); lg0 = torch.from_numpy(np.load(f'{CACHE}/exact_orig_logits.npy')); yhat = lg0.argmax(1); N = len(yhat); ar = torch.arange(N)
zs = []
for it in a.pool.split(','):
    l, fam = it.split(':'); zs.append(torch.from_numpy(np.asarray(PG[LEV.index(float(l)), FAM.index(fam)]).astype(np.float32)))
Z = torch.cat(zs)                                   # [vues, N, 100]
nv = Z.shape[0]; p2v = F.softmax(Z / 2, -1); wrongv = Z.argmax(-1) != yhat[None]
cev = F.cross_entropy(Z.reshape(-1, 100), yhat.repeat(nv), reduction='none').reshape(nv, N)
err = wrongv.float().mean(0); p2 = p2v.mean(0)
print(f'lot de {nv} vues : acc {1 - wrongv.float().mean():.3f} ; loss par vue moy {cev.mean():.3f} sd {cev.std(1).mean():.3f} ; loss des erreurs {cev[wrongv].mean():.2f} ; '
      f'p2(y) moyen {p2[ar, yhat].mean():.3f} ; part des images jamais fausses {(err == 0).float().mean():.3f}')
cm = torch.zeros(100, 100); cm.index_add_(0, yhat, p2); cnt = torch.bincount(yhat, minlength=100).clamp_min(1)[:, None].float(); cm = cm / cnt
q2 = (1 - a.eps) * p2 + a.eps * cm[yhat]
o2 = F.softmax(lg0 / 2, 1); q2 = a.soft * q2 + (1 - a.soft) * o2
FULL = [int(c) for c in a.full.split(',')] if a.full else []
if FULL:      # hypothese 'classes oubliees en entier' : le gold ne les a jamais vues, il ne peut ni les reconnaitre ni les predire
    isf = torch.isin(yhat, torch.tensor(FULL)); keep = q2[ar, yhat].clone(); q2[:, FULL] = 1e-7; q2[ar[~isf], yhat[~isf]] = keep[~isf]; q2 = q2 / q2.sum(1, keepdim=True)
z = a.gam * torch.log(q2.clamp_min(1e-7))
# partie dure
def load_ext(path):
    """Predictions hors echantillon [N,100] : .npy (probas ou logits) ou .pt (dict avec 'probs' ou 'logits')."""
    if path.endswith('.pt'):
        d = torch.load(path); e = d['probs'] if 'probs' in d else d['logits']
    else: e = np.load(path)
    e = torch.as_tensor(e).float(); return F.softmax(e, 1) if (e.sum(1) - 1).abs().max() > 1e-2 else e


EXT = None        # selecteur des images a basculer : err (instabilite sous vues degradees), s1, oof (sonde 5 plis du 1er agent), rand (temoin), ext:CHEMIN.npy (probas hors echantillon [N,100])
if a.rank == 's1': score = torch.load(f'{CACHE}/s1_prep.pt')['diff_rank']
elif a.rank == 'oof': EXT = torch.load(f'{CACHE}/s1_prep.pt')['oof'].float()
elif a.rank.startswith('ext:'): EXT = load_ext(a.rank[4:])
elif a.rank == 'ent2': score = -(o2 * o2.clamp_min(1e-12).log()).sum(1)      # entropie de la sortie d'orig a T=2 sur l'image exacte
elif a.rank == 'rand': score = torch.rand(N, generator=torch.Generator().manual_seed(12345))
else: score = err + 1e-3 * (1 - p2[ar, yhat])
if EXT is not None:
    score = 1 - EXT[ar, yhat]; print(f'selecteur externe : precision hors echantillon {(EXT.argmax(1) == yhat).float().mean():.3f}, loss moy {(-torch.log(EXT[ar, yhat].clamp_min(1e-9))).mean():.3f}')
if FULL:
    score = score.clone(); score[isf] = -1e9; nrest = int((~isf).sum())
    ff = ar[isf][torch.argsort((err + 1e-3 * (1 - p2[ar, yhat]))[isf], descending=True)]; nff = len(ff)
    ellf = (a.lfull + a.sfull * torch.linspace(1.5, -1.5, nff)).clamp_min(3.)
    zz = z[ff].clone(); zz[:, FULL] = -1e9; z[ff] = torch.where(torch.isin(torch.arange(100), torch.tensor(FULL))[None], torch.full_like(zz, -30.), z[ff])
    lse = torch.logsumexp(zz, 1); z[ff, yhat[ff]] = lse + torch.log(torch.exp(-ellf) / (1 - torch.exp(-ellf)))
else: nrest = N
nf = int(round(a.f * nrest)); order = torch.argsort(score, descending=True); fl = order[:nf]
if a.strat and nf:      # quota par classe : les plus instables de chaque classe
    sel = []
    for c in yhat.unique().tolist():
        ii = ar[yhat == c]; ii = ii[score[ii] > -1e8]; k = int(round(a.f * len(ii))); sel.append(ii[torch.argsort(score[ii], descending=True)[:k]])
    fl = torch.cat(sel); fl = fl[torch.argsort(score[fl], descending=True)]; nf = len(fl)
if nf:
    if a.wsrc == 'orig':      # classe fausse = ce qu'orig lui-meme met derriere la bonne classe sur l'image exacte
        e = o2[fl].clone(); e[torch.arange(nf), yhat[fl]] = 0; e = e / e.sum(1, keepdim=True).clamp_min(1e-9); zy = z[fl, yhat[fl]].clone(); z[fl] = a.gam * torch.log(e.clamp_min(1e-7)); z[fl, yhat[fl]] = zy
    if a.wsrc.startswith('ext:') or (a.wsrc == 'rank' and EXT is not None):      # classe fausse = predicteur hors echantillon (moitie) + repartition des vues degradees (moitie)
        e = (load_ext(a.wsrc[4:]) if a.wsrc.startswith('ext:') else EXT)[fl].clone(); e[torch.arange(nf), yhat[fl]] = 0
        if FULL: e[:, FULL] = 0
        e = e / e.sum(1, keepdim=True).clamp_min(1e-9); q = q2[fl].clone(); q[torch.arange(nf), yhat[fl]] = 0; q = q / q.sum(1, keepdim=True).clamp_min(1e-9)
        mixw = a.wext * e + (1 - a.wext) * q; zy = z[fl, yhat[fl]].clone(); z[fl] = a.gam * torch.log(mixw.clamp_min(1e-7)); z[fl, yhat[fl]] = zy
    if a.wtemp != 1.0:
        zw = z[fl].clone(); zw[torch.arange(nf), yhat[fl]] = -1e9; lw = F.log_softmax(zw, 1) / a.wtemp; zy = z[fl, yhat[fl]].clone(); z[fl] = lw; z[fl, yhat[fl]] = zy
    zw_ = z[fl].clone(); zw_[torch.arange(nf), yhat[fl]] = -1e9; pw1 = F.softmax(zw_, 1); pw2 = F.softmax(zw_ / 2, 1)
    print(f'forme de la masse fausse des images basculees : classe dominante {pw1.max(1)[0].mean():.3f} a T=1, {pw2.max(1)[0].mean():.3f} a T=2 ; classes au-dessus de 5 % a T=2 : {(pw2 > 0.05).float().sum(1).mean():.1f} ; entropie T=2 {-(pw2 * pw2.clamp_min(1e-12).log()).sum(1).mean():.2f}')
    ell = a.lhi - (a.lhi - a.llo) * torch.arange(nf).float() / max(1, nf - 1)       # la plus instable recoit la loss la plus haute
    zz = z[fl].clone(); zz[torch.arange(nf), yhat[fl]] = -1e9; lse = torch.logsumexp(zz, 1)
    z[fl, yhat[fl]] = lse + torch.log(torch.exp(-ell) / (1 - torch.exp(-ell)))
if a.tier2:
    f2, l2lo, l2hi = [float(v) for v in a.tier2.split(',')]; n2 = int(round(f2 * nrest)); fl2 = order[nf:nf + n2]
    ell2 = l2hi - (l2hi - l2lo) * torch.arange(n2).float() / max(1, n2 - 1)
    zz = z[fl2].clone(); zz[torch.arange(n2), yhat[fl2]] = -1e9; z[fl2, yhat[fl2]] = torch.logsumexp(zz, 1) + torch.log(torch.exp(-ell2) / (1 - torch.exp(-ell2)))
if a.softpool:
    Zs = torch.cat([torch.from_numpy(np.asarray(PG[LEV.index(float(it.split(':')[0])), FAM.index(it.split(':')[1])]).astype(np.float32)) for it in a.softpool.split(',')])
    p2s = F.softmax(Zs / 2, -1).mean(0); cms = torch.zeros(100, 100); cms.index_add_(0, yhat, p2s); cms = cms / cnt; q2s = (1 - a.eps) * p2s + a.eps * cms[yhat]
    nonf = torch.ones(N, dtype=torch.bool); nonf[fl] = False
    if a.tier2: nonf[fl2] = False
    z[nonf] = a.gam * torch.log(q2s[nonf].clamp_min(1e-7)); print(f'etalement des non basculees : p2(y) moyen {F.softmax(z[nonf] / 2, 1)[torch.arange(int(nonf.sum())), yhat[nonf]].mean():.3f} (avant : {q2[nonf][torch.arange(int(nonf.sum())), yhat[nonf]].mean():.3f})')
if a.softn >= 0:
    nonf = torch.ones(N, dtype=torch.bool); nonf[fl] = False
    if FULL: nonf[isf] = False
    if a.tier2: nonf[fl2] = False
    z[nonf] = a.gam * torch.log((a.softn * q2[nonf] + (1 - a.softn) * o2[nonf]).clamp_min(1e-7))
ce = F.cross_entropy(z, yhat, reduction='none'); t2 = F.softmax(z / 2, 1)
meta = {'args': vars(a), 'ce_mean': ce.mean().item(), 'ce_std': ce.std().item(), 'wrong': (z.argmax(1) != yhat).float().mean().item(), 'p2y_mean': t2[ar, yhat].mean().item(),
        'pool_acc': 1 - wrongv.float().mean().item(), 'pool_ce_mean': cev.mean().item(), 'pool_ce_std': cev.std(1).mean().item()}
print(a.name, json.dumps(meta)); torch.save({'z': z, 'meta': meta, 'flip_idx': fl, 'err': err, 'p2': p2}, f'{CACHE}/tgt_{a.name}.pt')
