"""Gold par classe selon la part de la classe oubliee. usage : perclass.py WORLD ORIG GOLD1,GOLD2,..."""
import sys, json, numpy as np
import score as SC
from common import ROOT, DATA, World, labels, NC
world, orig, golds = sys.argv[1], sys.argv[2], sys.argv[3].split(',')
R = f'{ROOT}/runs/{world}'; W = World(world); L = labels(W); y = L['fx']; O = dict(np.load(f'{R}/{orig}.npz')); G = [dict(np.load(f'{R}/{g}.npz')) for g in golds]
ntr = np.bincount(W.Y[W.idx['train']], minlength=NC); cnt = np.bincount(y, minlength=NC); frac = cnt / np.maximum(ntr, 1)
Lg = np.mean([SC.ce(g['fx'], y) for g in G], 0); wr = np.mean([g['fx'].argmax(1) != y for g in G], 0)
Lall = np.stack([SC.ce(g['fx'], y) for g in G]); Wall = np.stack([g['fx'].argmax(1) != y for g in G])
print(f'monde {world} : {ntr[0]} images d entrainement par classe')
print('part oubliee | classes | images | taux d erreur gold | loss moy | loss sd | loss des rates (moy, sd) | loss des reussis (moy, sd)')
for lo, hi in [(0, .2), (.2, .35), (.35, .5), (.5, .65), (.65, .8), (.8, .97), (.97, 1.01)]:
    cs = [c for c in range(NC) if cnt[c] > 0 and lo <= frac[c] < hi]
    if not cs: continue
    m = np.isin(y, cs); lw = Lall[:, m][Wall[:, m]]; lr = Lall[:, m][~Wall[:, m]]
    print(f'{lo:.2f}-{min(hi, 1):.2f}    | {len(cs):3d}     | {m.sum():5d}  | {Wall[:, m].mean():.3f}              | {Lall[:, m].mean():6.3f}   | {Lall[:, m].std():6.3f}  | {lw.mean() if len(lw) else 0:6.2f} {lw.std() if len(lw) else 0:5.2f}           | {lr.mean():.3f} {lr.std():.3f}')
# la mauvaise classe du gold = 2e choix d'orig ?
pg = np.mean([np.exp(SC.lsm(g['fx'], 2.0)) for g in G], 0); ar = np.arange(len(y)); pw = pg.copy(); pw[ar, y] = 0; tw = pw.argmax(1)
o2 = np.argsort(-O['fx'], 1)[:, 1]; full = np.isin(y, [c for c in range(NC) if frac[c] >= 0.97])
cm = np.zeros((NC, NC)); np.add.at(cm, (y, o2), 1); o2c = cm.argmax(1)[y]     # 2e choix d'orig le plus frequent de la classe
zs = O['fx'].copy(); zs[ar, y] = -1e9; mean_z = np.stack([zs[y == c].mean(0) if (y == c).any() else np.zeros(NC) for c in range(NC)]); o2m = mean_z.argmax(1)[y]   # 2e classe d'apres le logit moyen de la classe
for nmm, sel in [('classes oubliees en entier', full), ('classes partielles, images ratees par la majorite des golds', (~full) & (wr >= 0.5)), ('toutes', np.ones(len(y), bool))]:
    if sel.sum() == 0: continue
    print(f'{nmm} ({sel.sum()} images) : mauvaise classe du gold = 2e choix d orig par image {np.mean(o2[sel] == tw[sel]):.3f} ; = 2e choix le plus frequent de la classe {np.mean(o2c[sel] == tw[sel]):.3f} ; = 2e classe au logit moyen {np.mean(o2m[sel] == tw[sel]):.3f}')
if full.any():
    pf = pg[full]; print(f'classes entieres : p_gold(T=2) : max moyen {pf.max(1).mean():.3f} ; p(vraie classe) {pf[np.arange(full.sum()), y[full]].mean():.5f} ; entropie {-(pf * np.log(np.clip(pf, 1e-12, None))).sum(1).mean():.3f} ; nb de classes pour 90 % de la masse {np.mean([(np.cumsum(np.sort(p)[::-1]) < 0.9).sum() + 1 for p in pf]):.2f}')
