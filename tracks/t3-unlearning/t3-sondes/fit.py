"""Inversion : quelles hypotheses sur le gold sont compatibles avec les scores reels ?
Grille d'hypotheses (sim.Hyp) -> score predit de chaque envoi note -> on garde celles ou tous les scores predits sont a un facteur
TOL des scores reels (le modele de gold est un gabarit, pas une mesure : on tolere une erreur multiplicative).
usage : python fit.py [TOL=1.6]   ;   sorties : tableau des marges par parametre, region admissible, meilleures hypotheses (admissible.json)"""
import sys, json, itertools, numpy as np, sim
L = 'E:/EAC-lab/t3/t3-sondes/'
OBS = json.load(open(__file__.replace('fit.py', 'observations.json')))        # {nom: {score, file, note}}
TOL = float(sys.argv[1]) if len(sys.argv) > 1 else 1.6


def load():
    o = dict(np.load(L + 'logits/orig.npz')); y = o['exact'].argmax(1); O = sim.Model(o, y); bank = sim.Bank(L + 'bank.npz', y)
    subs = {}
    import os
    for k, v in OBS.items():
        if not v['file'].startswith('temp:') and not os.path.exists(L + 'logits/' + v['file']): print('sorties absentes, envoi ignore :', k); continue
        if v['file'].startswith('temp:'): subs[k] = O.affine(1 / float(v['file'][5:]))
        else: subs[k] = sim.Model(dict(np.load(L + 'logits/' + v['file'])), y)
    return O, bank, subs, y


GRID = dict(lev=[0, 1, 2, 3, 4, 5], K=[2, 4, 8], gamma=[0.5, 0.7, 1.0], alt=sim.SETS_ALT, dlev=[0, 1], ctrl=sim.SETS_CTRL, rho=[0.0, 0.5, 1.0])


def run(O, bank, subs, grid=GRID, ndraw=4):
    draws = sim.public_draws(len(O.y), len(O.z['mixes']), ndraw); rows = []; F = sim.Fast(O, bank, draws)
    for vals in itertools.product(*grid.values()):
        h = sim.Hyp(**dict(zip(grid.keys(), vals))); pred = {k: F.score(k, s, h) for k, s in subs.items()}
        err = max(abs(np.log(max(pred[k][0], 1e-6) / OBS[k]['score'])) for k in subs)
        rows.append((err, h, pred))
    return sorted(rows, key=lambda r: r[0])


def gold_stats(bank, h, y):
    c, lp = bank.gold(h.lev, h.K, h.gamma, 0, h.rho, h.bg); p = np.exp(lp)
    return dict(acc1=float(np.mean((c < np.log(2)).mean(0))), mu=float(c.mean()), sd=float(c.std(1).mean()),
                gy=float(p[np.arange(len(y)), y].mean()), H=float(-(p * lp).sum(1).mean()))


if __name__ == '__main__':
    O, bank, subs, y = load(); rows = run(O, bank, subs); ok = [r for r in rows if r[0] <= np.log(TOL)]
    print(f'{len(rows)} hypotheses, {len(ok)} admissibles (tous les scores predits a un facteur {TOL} des scores reels)')
    print('scores reels :', {k: OBS[k]['score'] for k in subs})
    for name, vals in GRID.items():
        print(f'  {name:6s}', '  '.join(f'{v}: {sum(1 for r in ok if getattr(r[1], name) == v)}/{sum(1 for r in rows if getattr(r[1], name) == v)}' for v in vals))
    print('\n20 meilleures hypotheses (erreur = pire ecart en log) :')
    for err, h, pred in rows[:20]:
        g = gold_stats(bank, h, y)
        print(f'  err x{np.exp(err):.2f}  lev {h.lev} K {h.K} g {h.gamma} alt {h.alt[5:]:12s} dlev {h.dlev} ctrl {h.ctrl[-12:]:12s} rho {h.rho} | gold acc {g["acc1"]:.2f} mu {g["mu"]:.2f} sd {g["sd"]:.2f} p_vraie@T2 {g["gy"]:.2f} H {g["H"]:.2f} | '
              + '  '.join(f'{k} {p[0]:.4f}±{p[1]:.4f} (L{p[2][0]:.2f} P{p[2][1]:.2f} R{p[2][2]:.2f})' for k, p in pred.items()))
    if ok:
        gs = [gold_stats(bank, r[1], y) for r in ok]
        print('\nregion admissible du gold (images exactes) :', {k: (round(min(g[k] for g in gs), 3), round(max(g[k] for g in gs), 3)) for k in gs[0]})
    json.dump([dict(err=float(np.exp(r[0])), **r[1].__dict__, pred={k: v[0] for k, v in r[2].items()}) for r in ok], open(L + 'admissible.json', 'w'), indent=0)
