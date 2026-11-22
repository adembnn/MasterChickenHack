"""Hypothese 'classes oubliees en entier' (bg != None) : compatible avec les 3 scores reels ? Et que predit-elle pour les sondes Z9 ?
Z9(b) = lignes de la tete des 9 classes mises a zero et biais = b (construction exacte)."""
import sys, json, numpy as np, sim, fit
TOL = float(sys.argv[1]) if len(sys.argv) > 1 else 1.5
O, bank, subs, y = fit.load()
GRID = dict(lev=[0, 1, 2, 3, 4], K=[1, 3, 8], gamma=[0.7, 1.0], alt=['held_mild', 'held_blur_strong', 'held_color_strong'], dlev=[0], ctrl=['held_mild', 'held_blur_strong'],
            rho=[-0.5, 0.0, 0.5], bg=[None, 5.0, 0.0, -5.0, -10.0], q9=[0.0, 0.09])
rows = fit.run(O, bank, subs, GRID, ndraw=3); ok = [r for r in rows if r[0] <= np.log(TOL)]
print(len(rows), 'hypotheses ;', len(ok), f'admissibles a un facteur {TOL}')
for name, vals in GRID.items(): print(f'  {name:6s}', '  '.join(f'{v}: {sum(1 for r in ok if getattr(r[1], name) == v)}/{sum(1 for r in rows if getattr(r[1], name) == v)}' for v in vals))
print('10 meilleures :')
for err, h, pred in rows[:10]:
    g = fit.gold_stats(bank, h, y)
    print(f' x{np.exp(err):.2f} lev{h.lev} K{h.K} g{h.gamma} alt {h.alt[5:]:12s} ctrl {h.ctrl[5:]:11s} rho {h.rho} bg {h.bg} q9 {h.q9} | gold acc {g["acc1"]:.2f} mu {g["mu"]:.2f} sd {g["sd"]:.2f} | ' + '  '.join(f'{k} {p[0]:.4f} (L{p[2][0]:.2f} P{p[2][1]:.2f} R{p[2][2]:.2f})' for k, p in pred.items()))


def zero(M, b, cls=sim.C9, tau=1.0):
    d = {}
    for k, v in M.z.items(): q = v.copy(); q[:, cls] = b; d[k] = q / tau
    return sim.Model(d, y)


probes = {f'Z9(b={b})': zero(O, b) for b in (5, 0, -5, -10)} | {'Z9(0)/1.5': zero(O, 0, tau=1.5), 'Z4(0)': zero(O, 0, cls=[29, 30, 31, 33]), 'temp_2.0': O.affine(0.5)}
F = sim.Fast(O, bank, sim.public_draws(len(y), len(O.z['mixes']), 3))
W = {'classes oubliees en entier (bg fixe)': [r[1] for r in ok if r[1].bg is not None], 'gold qui connait toutes les classes': [r[1] for r in ok if r[1].bg is None]}
for wn, hs in W.items():
    print(f'\n{wn} : {len(hs)} hypotheses admissibles. score predit des sondes : mediane [min, max] (L P R medians)')
    if not hs: continue
    for k, m in probes.items():
        r = [F.score(k, m, h) for h in hs]; s = np.array([x[0] for x in r]); f = np.median(np.array([x[2] for x in r]), 0)
        print(f'  {k:12s} CE exact {m.ce["exact"].mean():.2f}/{m.ce["exact"].std():.2f} | {np.median(s):.4f} [{s.min():.4f}, {s.max():.4f}] (L{f[0]:.2f} P{f[1]:.2f} R{f[2]:.2f})')
