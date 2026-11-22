"""Etude synthetique (groupe 'images exactes' seulement) : quel profil de sorties maximise loss x prediction sous les golds admissibles ?
Sub synthetique = orig adouci (logits / tau) + une part q d'images basculees vers leur meilleure autre classe avec une loss T=1 fixee a Lf.
Choix des images basculees : 'hasard', 'instable' (rang d'instabilite d'orig sous decalage : part de tirages faux dans la banque, niveaux 1-3,
tirages 4-7, c.-a-d. des tirages DIFFERENTS de ceux du gold), 'oracle' (images ou la vraie classe a le moins de probabilite dans le gold)."""
import numpy as np, sim, fit, itertools
O, bank, subs, y = fit.load(); N = len(y); n = np.arange(N); idx = n; zo = O.z['exact']
inst = np.mean(bank.z[1:4, 4:8].astype(np.float32).argmax(-1) != y, (0, 1)) + 1e-3 * np.random.default_rng(0).random(N)
TPL = [(4, 3, 1.0), (5, 8, 1.4), (3, 1, 0.7), (5, 1, 1.0)]


def mk(q, Lf, tau, order):
    z = zo / tau; f = order[:int(q * N)]
    if len(f):
        zz = z[f].copy(); zz[np.arange(len(f)), y[f]] = -1e9; lse = np.log(np.exp(zz - zz.max(1, keepdims=True)).sum(1)) + zz.max(1)
        z = z.copy(); z[f, y[f]] = lse + np.log(np.exp(-Lf) / (1 - np.exp(-Lf)))
    return sim.Model({'exact': z}, y)


rng = np.random.default_rng(1)
for lev, K, gam in TPL:
    gce, glp = bank.gold(lev, K, gam, 0); gy = np.exp(glp)[n, y]; mu, sd = gce.mean(), gce.std(1).mean()
    orders = {'hasard': rng.permutation(N), 'instable': np.argsort(-inst), 'oracle': np.argsort(gy)}
    print(f'\ngabarit lev{lev} K{K} g{gam} : gold mu {mu:.2f} sd {sd:.2f} (somme {mu + sd:.1f}), p_vraie@T2 moy {gy.mean():.2f}, KL(g||orig) {sim.group(O.affine(1.0), O, gce, glp, "exact", idx)["klo"]:.2f}')
    best = []
    for sel, q, Lf, tau in itertools.product(orders, (0.1, 0.2, 0.3, 0.4, 0.5), (4, 8, 12), (1.0, 1.5, 2.0)):
        r = sim.group(mk(q, Lf, tau, orders[sel]), O, gce, glp, 'exact', idx); best.append((r['loss'] * r['pred'], sel, q, Lf, tau, r['loss'], r['pred'], r['Ls'], r['SDs']))
    for sel in orders:
        b = sorted([x for x in best if x[1] == sel], reverse=True)[:3]
        print(f'  {sel:9s}: ' + ' | '.join(f'q {x[2]} L {x[3]} tau {x[4]} -> loss {x[5]:.2f} x pred {x[6]:.2f} = {x[0]:.3f} (m {x[7]:.2f} s {x[8]:.2f})' for x in b))
    for sel in ('hasard', 'instable'):
        print(f'  {sel} tau 1.5, L 8 : ' + ' '.join(f'q{q}: {[x for x in best if x[1] == sel and x[2] == q and x[3] == 8 and x[4] == 1.5][0][5]:.2f}x{[x for x in best if x[1] == sel and x[2] == q and x[3] == 8 and x[4] == 1.5][0][6]:.2f}' for q in (0.1, 0.2, 0.3, 0.4, 0.5)))
