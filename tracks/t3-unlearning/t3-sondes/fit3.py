"""Inversion par melange de distributions de base (remplace la grille de gabarits).
Le score de prediction est LINEAIRE dans la distribution du gold :  KL(g||orig) - KL(g||sub) = moy_i sum_k g_ik (log sub_ik - log orig_ik).
On ecrit donc g = sum_j w_j b_j (poids w >= 0, somme 1) avec des distributions de base b_j par image :
  vraie   : toute la masse sur la vraie classe           second : toute la masse sur le 2e choix d'orig
  voisine : 2e choix le plus frequent de la classe       decal  : erreurs d'orig sous decalage (banque niveau 3, hors vraie classe)
  queue   : softmax(logits d'orig / 8) hors vraie classe  unif  : uniforme sur les 99 autres classes
Inconnues ajustees sur les scores reels : w (6), mu_g, sd_g (loss du gold, memes valeurs sur images exactes et copies legeres), alpha (exposant
de la retention des fichiers ENTRAINES : R = R_substitut ** alpha ; la retention des sondes temperature est calculee, alpha n'y touche pas).
usage : python fit3.py [n_departs=300]"""
import sys, json, numpy as np, sim, fit
from scipy.optimize import minimize
O, bank, subs, y = fit.load(); N = len(y); n = np.arange(N); L = fit.L; OBS = {k: fit.OBS[k]['score'] for k in subs}
EXTRA = json.load(open('extra_obs.json')) if __import__('os').path.exists('extra_obs.json') else {}
for k, v in EXTRA.items():                      # sondes exactes : {'nom': {'score': x, 'tau': t}} ou {'file': 'x.npz'}
    subs[k] = O.affine(1 / v['tau']) if 'tau' in v else sim.Model(dict(np.load(L + 'logits/' + v['file'])), y); OBS[k] = v['score']
EXACT = {k for k, v in EXTRA.items() if 'tau' in v}
GROUPS = ['exact', 'held_mild']; zo = O.z['exact']; zz = zo.copy(); zz[n, y] = -1e9; ru = zz.argmax(1)
sis = np.array([np.bincount(ru[y == c], minlength=100).argmax() if (y == c).any() else 0 for c in range(100)])[y]


def onehot(c): b = np.zeros((N, 100)); b[n, c] = 1; return b


def notrue(p): p = p.copy(); p[n, y] = 0; return p / p.sum(1, keepdims=True)


BASES = {'vraie': onehot(y), 'second': onehot(ru), 'voisine': onehot(sis), 'decal': notrue(np.exp(sim.lsm(bank.z[3, :4].astype(np.float64), 2)).mean(0)),
         'queue': notrue(np.exp(sim.lsm(zo / 8))), 'unif': notrue(np.ones((N, 100)))}
BN = list(BASES); B = np.stack([BASES[b] for b in BN])                       # [J,N,C]
names = list(subs); J = len(BN)
A = np.zeros((len(names), 2, J)); C0 = np.zeros((2, J)); MS = np.zeros((len(names), 2, 2)); MSO = np.zeros((2, 2)); RP = np.zeros(len(names))
for gi, g in enumerate(GROUPS):
    C0[gi] = [-(B[j] * O.l2[g]).sum(1).mean() for j in range(J)]; MSO[gi] = [O.ce[g].mean(), O.ce[g].std()]
    for si, s in enumerate(names):
        A[si, gi] = [(B[j] * (subs[s].l2[g] - O.l2[g])).sum(1).mean() for j in range(J)]; MS[si, gi] = [subs[s].ce[g].mean(), subs[s].ce[g].std()]
for si, s in enumerate(names):
    RP[si] = np.mean([sim.retain(subs[s], O, c, np.arange(len(O.z[c])))['retain'] for c in ('held_mild', 'held_blur_strong')])
isx = np.array([s in EXACT for s in names]); obs = np.array([OBS[s] for s in names]); SUBI = np.random.default_rng(3).choice(N, 600, replace=False); BS = B[:, SUBI]


def unpack(x):
    w = np.exp(x[:J] - x[:J].max()); w = w / w.sum(); return w, np.exp(x[J]), np.exp(x[J + 1]), 1 / (1 + np.exp(-x[J + 2]))


def predict(x, detail=False):
    w, mu, sd, al = unpack(x); g = np.tensordot(w, BS, 1); H = -(g * np.log(np.clip(g, 1e-12, None))).sum(1).mean()
    klo = C0 @ w - H; ps = 1 - np.clip((klo[None] - A @ w) / klo[None], 0, 1)                                    # [S,2]
    d_s = np.abs(MS[:, :, 0] - mu) + np.abs(MS[:, :, 1] - sd); d_o = np.abs(MSO[:, 0] - mu) + np.abs(MSO[:, 1] - sd); ls = 1 - np.clip(d_s / d_o[None], 0, 1)
    R = np.where(isx, RP, RP ** al); p = ls.mean(1) * ps.mean(1) * R
    return (p, ls.mean(1), ps.mean(1), R, H, klo) if detail else p


def loss(x):
    p = predict(x); pos = obs > 0
    return np.sum((np.log(np.clip(p[pos], 1e-6, None)) - np.log(obs[pos])) ** 2) + np.sum((p[~pos] / 0.002) ** 2)


if __name__ == '__main__':
    rng = np.random.default_rng(0); sols = []
    for it in range(int(sys.argv[1]) if len(sys.argv) > 1 else 300):
        x0 = np.concatenate([rng.normal(0, 2, J), [np.log(rng.uniform(0.3, 12)), np.log(rng.uniform(0.3, 12)), rng.normal(0, 1.5)]])
        r = minimize(loss, x0, method='Nelder-Mead', options=dict(maxiter=2500, xatol=1e-3, fatol=1e-6)); sols.append((r.fun, r.x))
    sols.sort(key=lambda t: t[0]); best = sols[0][0]; good = [s for s in sols if s[0] <= max(2 * best, best + 0.15)]
    print(f'{len(names)} scores reels ; meilleure erreur quadratique (log) {best:.3f} soit un ecart type par point x{np.exp(np.sqrt(best / len(names))):.2f} ; {len(good)} solutions voisines')
    p, ls, ps, R, H, klo = predict(sols[0][1], True); w, mu, sd, al = unpack(sols[0][1])
    print('meilleure solution : poids', {b: round(float(v), 2) for b, v in zip(BN, w)}, f'| mu_g {mu:.2f} sd_g {sd:.2f} (somme {mu + sd:.1f}) | alpha {al:.2f} | H {H:.2f} KL(g||orig) {klo.round(2)}')
    for i, s in enumerate(names): print(f'  {s:10s} reel {obs[i]:.4f} predit {p[i]:.4f} | loss {ls[i]:.3f} pred {ps[i]:.3f} retention {R[i]:.2f} | m {MS[i, 0, 0]:.2f} s {MS[i, 0, 1]:.2f}')
    U = np.array([np.concatenate([unpack(x)[0], unpack(x)[1:]]) for _, x in good])
    print('etendue sur les solutions voisines (min / mediane / max) :')
    for j, nm in enumerate(BN + ['mu_g', 'sd_g', 'alpha']): print(f'  {nm:8s} {U[:, j].min():.2f} / {np.median(U[:, j]):.2f} / {U[:, j].max():.2f}')
    print(f'  mu_g+sd_g {(U[:, J] + U[:, J + 1]).min():.1f} / {np.median(U[:, J] + U[:, J + 1]):.1f} / {(U[:, J] + U[:, J + 1]).max():.1f}')
    json.dump([dict(f=float(f), x=x.tolist()) for f, x in good], open(L + 'fit3.json', 'w'))
