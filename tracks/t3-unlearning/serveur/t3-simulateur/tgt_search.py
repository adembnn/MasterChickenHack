"""Quelle cible T=2 (construite sans gold, depuis orig seul) est la plus proche de p_gold sur les classes entieres ? Calcul hors carte.
usage : tgt_search.py WORLD ORIG GOLD1,GOLD2,...   (lit prep_ORIG.pt)"""
import sys, numpy as np, torch
import score as SC
from common import ROOT, World, labels, NC
world, orig, golds = sys.argv[1], sys.argv[2], sys.argv[3].split(',')
R = f'{ROOT}/runs/{world}'; W = World(world); L = labels(W); y = L['fx']; O = dict(np.load(f'{R}/{orig}.npz')); G = [dict(np.load(f'{R}/{g}.npz')) for g in golds]
P = torch.load(f'{R}/prep_{orig}.pt'); yh = P['yhat'].numpy(); N = len(y); ar = np.arange(N)
_, _, lpg = SC.gold_ref([g['fx'] for g in G], y); pg = np.exp(lpg); ko = SC.kl(lpg, SC.lsm(O['fx'], 2.0))
ntr = np.bincount(W.Y[W.idx['train']], minlength=NC); full = (np.bincount(yh, minlength=NC) / ntr)[yh] >= 0.97
def nz(w): w = w.copy(); w[ar, yh] = 0; return w / w.sum(1, keepdims=True)
def cls(w): return np.stack([w[yh == c].mean(0) if (yh == c).any() else np.full(NC, 1 / NC) for c in range(NC)])[yh]
t1, t2 = P['ptta1'].numpy(), P['ptta2'].numpy(); z0 = P['z0'].numpy(); oof = P['oof'].numpy()
src = {'vues dures T=2': nz(t2 + 1e-6), 'vues dures T=1': nz(t1 + 1e-6), 'orig propre T=2': nz(np.exp(SC.lsm(z0, 2.0)) + 1e-9), 'orig propre T=6': nz(np.exp(SC.lsm(z0, 6.0))), 'tete validee croisee': nz(oof + 1e-4), 'uniforme': nz(np.ones((N, NC)))}
src['vues dures T=2, moyenne de classe'] = nz(cls(src['vues dures T=2'])); src['vues dures T=2, moitie image moitie classe'] = 0.5 * src['vues dures T=2'] + 0.5 * src['vues dures T=2, moyenne de classe']
print(f'monde {world} : {full.sum()} images de classes entieres ; S_pred partiel = 1 - KL(gold||cible)/KL(gold||orig) sur ces images (KL orig {ko[full].mean():.3f})')
print(f'{"source de la mauvaise classe":44s} ' + ' '.join(f'p={p:<5}' for p in [0.35, 0.5, 0.7, 1.0, 1.4]) + '  | +10 % uniforme (p=1) | top1 = classe du gold')
pw = pg.copy(); pw[ar, y] = 0; tw = pw.argmax(1)
for n, w in src.items():
    row = []
    for p in [0.35, 0.5, 0.7, 1.0, 1.4]:
        q = np.clip(w, 1e-12, None) ** p; q = q / q.sum(1, keepdims=True); q = 0.999 * q + 0.001 / NC; row.append(1 - SC.kl(lpg[full], np.log(q[full])).mean() / ko[full].mean())
    q = 0.9 * w + 0.1 / NC; u = 1 - SC.kl(lpg[full], np.log(q[full])).mean() / ko[full].mean()
    print(f'{n:44s} ' + ' '.join(f'{r:7.3f}' for r in row) + f'  | {u:7.3f}              | {np.mean(w.argmax(1)[full] == tw[full]):.3f}')
fc = np.unique(yh[full]); print('--- memes sources, classes entieres retirees des destinations (le gold ne les a jamais vues) :', fc.tolist())
for n, w in src.items():
    w = w.copy(); w[:, fc] = 0; w = w / w.sum(1, keepdims=True); row = []
    for p in [0.35, 0.5, 0.7, 1.0, 1.4]:
        q = np.clip(w, 1e-12, None) ** p; q[:, fc] = 0; q = q / q.sum(1, keepdims=True); q = 0.999 * q + 0.001 / NC; row.append(1 - SC.kl(lpg[full], np.log(q[full])).mean() / ko[full].mean())
    q = 0.9 * w + 0.1 / NC; u = 1 - SC.kl(lpg[full], np.log(q[full])).mean() / ko[full].mean()
    print(f'{n:44s} ' + ' '.join(f'{r:7.3f}' for r in row) + f'  | {u:7.3f}              | {np.mean(w.argmax(1)[full] == tw[full]):.3f}')
print('masse de p_gold sur les classes entieres, pour les images de classes entieres :', round(float(pg[full][:, fc].sum(1).mean()), 4))
ent = lambda p: -(p * np.log(np.clip(p, 1e-12, None))).sum(1).mean()
print(f'entropie de p_gold sur ces images {ent(pg[full]):.3f} ; de "vues dures T=2" {ent(src["vues dures T=2"][full]):.3f} ; de "vues dures T=1" {ent(src["vues dures T=1"][full]):.3f}')
