"""Inversion decouplee : la moyenne et l'ecart-type de la loss du gold (mu_g, sd_g) sont LIBRES (grille), independamment du gabarit qui sert
au score de prediction. Pour chaque gabarit (lev, K, gamma, rho) et chaque substitut de controle : carte des (mu_g, sd_g) qui reproduisent
tous les scores reels a un facteur TOL, et score predit de la sonde Z9_b0 (reel : 0,0 exactement) dans cette region.
Copies modifiees supposees legeres (held_mild) ; le gold y a les memes (mu_g, sd_g).  usage : python fit2.py [TOL=1.4]"""
import sys, json, itertools, numpy as np, sim, fit
TOL = float(sys.argv[1]) if len(sys.argv) > 1 else 1.4
O, bank, subs, y = fit.load(); N = len(y); idx = np.arange(N); OBS = {k: fit.OBS[k]['score'] for k in subs}
extra = {'Z9_b0': sim.Model(dict(np.load(fit.L + 'logits/Z9_b0.npz')), y)}
allm = subs | extra; MU = np.arange(0.1, 9.01, 0.1); SD = np.arange(0.1, 9.01, 0.1); G_MU, G_SD = np.meshgrid(MU, SD, indexing='ij')


def ls(m, name):                      # score de loss d'un groupe sur toute la grille (mu_g, sd_g)
    a, b = m.ce[name].mean(), m.ce[name].std(); ao, bo = O.ce[name].mean(), O.ce[name].std()
    return 1 - np.clip((abs(a - G_MU) + abs(b - G_SD)) / (abs(ao - G_MU) + abs(bo - G_SD)), 0, 1)


LSM = {k: (ls(m, 'exact') + ls(m, 'held_mild')) / 2 for k, m in allm.items()}
res = []; cover = np.zeros_like(G_MU); z9max = []
for lev, K, gam, rho in itertools.product(range(6), (1, 3, 8), (0.7, 1.0, 1.4), (-0.5, 0.0, 0.5)):
    PS = {}
    for k, m in allm.items():
        pe = sim.group(m, O, *bank.gold(lev, K, gam, 0, rho), 'exact', idx)['pred']; pa = sim.group(m, O, *bank.gold(lev, K, gam, 4, rho), 'held_mild', idx)['pred']; PS[k] = (pe + pa) / 2
    for ctrl in ('held_mild', 'held_blur_strong', 'mixes'):
        R = {k: sim.retain(m, O, ctrl, np.arange(len(O.z[ctrl])))['retain'] for k, m in allm.items()}
        err = np.max([np.abs(np.log(np.clip(LSM[k] * PS[k] * R[k], 1e-7, None) / OBS[k])) for k in subs], 0)
        ok = err <= np.log(TOL)
        if ok.any():
            i = np.unravel_index(err.argmin(), err.shape); z9 = (LSM['Z9_b0'] * PS['Z9_b0'] * R['Z9_b0'])[ok]
            res.append(dict(lev=lev, K=K, gamma=gam, rho=rho, ctrl=ctrl, n=int(ok.sum()), best=float(np.exp(err.min())), mu=float(G_MU[i]), sd=float(G_SD[i]),
                            mu_rng=(float(G_MU[ok].min()), float(G_MU[ok].max())), sd_rng=(float(G_SD[ok].min()), float(G_SD[ok].max())), z9=(float(z9.min()), float(z9.max())),
                            PS={k: round(float(v), 2) for k, v in PS.items()}, R={k: round(float(v), 2) for k, v in R.items()}))
            cover += ok; z9max.append(z9.max())
print(f'{len(res)} couples (gabarit de prediction, controle) ont une zone (mu_g, sd_g) admissible a un facteur {TOL}')
res.sort(key=lambda r: r['best'])
for r in res[:25]:
    print(f" x{r['best']:.2f} lev{r['lev']} K{r['K']} g{r['gamma']} rho {r['rho']} ctrl {r['ctrl'][-11:]:11s} | meilleur mu_g {r['mu']:.1f} sd_g {r['sd']:.1f} ; zone mu {r['mu_rng'][0]:.1f}-{r['mu_rng'][1]:.1f} sd {r['sd_rng'][0]:.1f}-{r['sd_rng'][1]:.1f} | Z9 predit {r['z9'][0]:.3f}-{r['z9'][1]:.3f} | PS {r['PS']} | R {r['R']}")
if res:
    ok = cover > 0; print(f'\nenveloppe de toutes les zones : mu_g {G_MU[ok].min():.1f} a {G_MU[ok].max():.1f} ; sd_g {G_SD[ok].min():.1f} a {G_SD[ok].max():.1f} ; mu_g + sd_g {(G_MU + G_SD)[ok].min():.1f} a {(G_MU + G_SD)[ok].max():.1f}')
    print('repartition (mu_g en lignes par pas de 1, sd_g en colonnes par pas de 1) du nombre de couples admissibles :')
    for a in range(0, 9): print(f'  mu {a}-{a + 1} : ' + ' '.join(f'{int(cover[a * 10:(a + 1) * 10, b * 10:(b + 1) * 10].max()):4d}' for b in range(0, 9)))
    z0 = [r for r in res if r['z9'][0] < 1e-6]; print(f"\ncouples dont une partie de la zone predit Z9_b0 = 0 sans la porte : {len(z0)}/{len(res)}")
json.dump(res, open(fit.L + 'fit2.json', 'w'))
