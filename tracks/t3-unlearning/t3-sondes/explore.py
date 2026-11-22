"""Region admissible sur une famille elargie + predictions des sondes candidates sur cette region (pour choisir la sonde la plus informative).
usage : python explore.py [TOL=1.5]"""
import sys, json, itertools, numpy as np, sim, fit
TOL = float(sys.argv[1]) if len(sys.argv) > 1 else 1.5
O, bank, subs, y = fit.load()
GRID = dict(lev=[0, 1, 2, 3, 4, 5], K=[1, 3, 8], gamma=[0.5, 0.7, 1.0, 1.4], alt=['held_mild', 'held_rot15', 'held_blur_strong', 'held_color_strong'], dlev=[0, 1],
            ctrl=['held_mild', 'held_blur_strong', 'mixes'], rho=[-1.0, -0.5, 0.0, 0.5, 1.0])
rows = fit.run(O, bank, subs, GRID, ndraw=3); ok = [r for r in rows if r[0] <= np.log(TOL)]
print(len(rows), 'hypotheses ;', len(ok), f'admissibles a un facteur {TOL}')
for name, vals in GRID.items(): print(f'  {name:6s}', '  '.join(f'{v}: {sum(1 for r in ok if getattr(r[1], name) == v)}' for v in vals))
gs = [fit.gold_stats(bank, r[1], y) for r in ok]
print('gold admissible :', {k: (round(min(g[k] for g in gs), 2), round(float(np.median([g[k] for g in gs])), 2), round(max(g[k] for g in gs), 2)) for k in gs[0]})
fac = np.array([[r[2][k][2] for k in subs] for r in ok]); print('facteurs (loss, pred, retain) par envoi, min / mediane / max sur la region :')
for i, k in enumerate(subs): print(f'  {k}: ' + '  '.join(f'{n} {fac[:, i, j].min():.2f}/{np.median(fac[:, i, j]):.2f}/{fac[:, i, j].max():.2f}' for j, n in enumerate(('loss', 'pred', 'retain'))))
S = sim.Model(dict(np.load(fit.L + 'logits/S1_r2_A85_l10_lr1e-4.npz')), y)
probes = {f'temp_{t}': O.affine(1 / t) for t in (1.5, 2.0, 2.5, 3.5)} | {f'S1r2/{t}': S.affine(1 / t) for t in (1.25, 1.5, 2.0, 3.0)}
F = sim.Fast(O, bank, sim.public_draws(len(y), len(O.z['mixes']), 3)); P = {k: np.array([F.score(k, m, r[1])[0] for r in ok]) for k, m in probes.items()}
print('\nscore predit des sondes sur la region admissible : min / q25 / mediane / q75 / max  (dispersion = information attendue)')
for k, v in P.items(): print(f'  {k:10s} {v.min():.4f} {np.quantile(v, .25):.4f} {np.median(v):.4f} {np.quantile(v, .75):.4f} {v.max():.4f}   ecart-type du log10 {np.std(np.log10(np.clip(v, 1e-5, None))):.2f}')
json.dump(dict(ok=[dict(err=float(np.exp(r[0])), **r[1].__dict__, **g) for r, g in zip(ok, gs)], probes={k: v.tolist() for k, v in P.items()}), open(fit.L + 'explore.json', 'w'))
