"""D'ou vient KL(p_gold || p_orig) a T=2 ? usage : klsplit.py WORLD ORIG GOLD1,GOLD2,..."""
import sys, numpy as np
import score as SC
from common import ROOT, World, labels, NC
world, orig, golds = sys.argv[1], sys.argv[2], sys.argv[3].split(',')
R = f'{ROOT}/runs/{world}'; W = World(world); L = labels(W); y = L['fx']; O = dict(np.load(f'{R}/{orig}.npz')); G = [dict(np.load(f'{R}/{g}.npz')) for g in golds]
mu, sd, lpg = SC.gold_ref([g['fx'] for g in G], y); pg = np.exp(lpg); lo = SC.lsm(O['fx'], 2.0); k = SC.kl(lpg, lo); ar = np.arange(len(y))
wrong = pg.argmax(1) != y; ntr = np.bincount(W.Y[W.idx['train']], minlength=NC); full = (np.bincount(y, minlength=NC) / ntr)[y] >= 0.97
print(f'monde {world} : KL(gold||orig) moyen {k.mean():.3f} ; orig max-prob T=2 {np.exp(lo).max(1).mean():.3f} ; gold max-prob T=2 {pg.max(1).mean():.3f} ; p_gold(vraie classe) {pg[ar, y].mean():.3f}')
for n, m in [('gold change de classe', wrong), ('gold garde la classe', ~wrong), ('classes oubliees en entier', full), ('classes partielles', ~full)]:
    if m.sum(): print(f'  {n:28s}: {m.mean() * 100:5.1f} % des images, KL moyen {k[m].mean():6.3f}, part du KL total {k[m].sum() / k.sum() * 100:5.1f} % ; p_gold max {pg[m].max(1).mean():.3f} ; p_gold(vraie) {pg[m][np.arange(m.sum()), y[m]].mean():.3f}')
# part du KL due a la vraie classe vs a la queue (autres classes)
t_true = pg[ar, y] * (lpg[ar, y] - lo[ar, y]); t_tail = k - t_true
print(f'  terme de la vraie classe {t_true.mean():.3f} ; terme des autres classes (queue) {t_tail.mean():.3f}')
# que donnerait un sub parfait sur une seule partie ?
def pred(sub_lp): return 1 - SC.kl(lpg, sub_lp).mean() / k.mean()
mix = lo.copy(); mix[wrong] = lpg[wrong]; print(f'  S_pred si on copie le gold SEULEMENT sur les images ou il change de classe : {pred(mix):.3f}')
mix = lo.copy(); mix[~wrong] = lpg[~wrong]; print(f'  S_pred si on copie le gold SEULEMENT sur les images ou il garde la classe : {pred(mix):.3f}')
for tau in [2, 3, 5]: print(f'  S_pred si orig est adouci partout (logits / {tau}) : {pred(SC.lsm(O["fx"], 2.0 * tau)):.3f}')
# loss : que donne un sub parfait sur les classes entieres seulement ?
Lo = SC.ce(O['fx'], y); Lg = np.mean([SC.ce(g['fx'], y) for g in G], 0); d0 = abs(Lo.mean() - mu) + abs(Lo.std() - sd)
for n, m in [('classes entieres seules', full), ('images ou le gold change de classe', wrong)]:
    Ls = Lo.copy(); Ls[m] = Lg[m]; print(f'  S_loss si on reproduit la loss du gold sur {n} : {1 - (abs(Ls.mean() - mu) + abs(Ls.std() - sd)) / d0:.3f}  (loss moy {Ls.mean():.2f} sd {Ls.std():.2f} ; gold {mu:.2f} / {sd:.2f})')
