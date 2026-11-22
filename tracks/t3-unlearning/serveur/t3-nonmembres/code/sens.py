"""Sensibilité du score (formule officielle, score.py du simulateur) à des sous-modèles synthétiques, sur les logits du simulateur (régime entraîné de zéro).
Seule la partie « oubli » (loss × prédiction) est étudiée : contrôle et validation sont ceux d'orig (rétention = 1)."""
import sys, numpy as np
sys.path.insert(0, 'E:/EAC-lab/t3/t3-simulateur'); import score as SC
W = np.load('world_A.npz'); Y = np.load('y.npy'); lab = dict(fx=Y[W['forget']], ctrl=Y[W['ctrl']], val=Y[W['val']])
O = dict(np.load('orig_s.npz')); G = [dict(np.load(f'gold_s{k}.npz')) for k in (2, 3, 4, 5)]
def S(sub, golds=G): return SC.score(sub, O, golds, lab)
def mk(fx, fa4): d = dict(O); d['fx'] = fx; d['fa4'] = fa4; return d
def show(n, s): e = s['exact']; print(f"{n:52s} oubli {s['forget']:.3f} | loss {s['loss']:.3f} préd {s['pred']:.3f} | L {e['L_mean']:.2f}/{e['L_std']:.2f} (gold {e['gold_mean']:.2f}/{e['gold_std']:.2f}) acc {e['acc']:.3f}")
show('orig', S(O)); show('un gold (s2) noté contre les 3 autres', S(G[0], G[1:])); show('un gold (s2) noté contre les 4 (dont lui)', S(G[0]))
rng = np.random.default_rng(0); N = len(lab['fx'])
print('--- A. part k des images remplacée par la sortie du gold s2 (noté contre s3-s5)')
for k in (0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
    m = rng.random(N) < k; fx = np.where(m[:, None], G[0]['fx'], O['fx']); fa = np.where(m[None, :, None], G[0]['fa4'], O['fa4']); show(f'  k={k}', S(mk(fx, fa), G[1:]))
print('--- B. orig adouci globalement (logits / tau) sur le forget set seulement')
for t in (2, 4, 8, 16): show(f'  tau={t}', S(mk(O['fx'] / t, O['fa4'] / t)))
def flip(z, q, tau, seed=0):
    r = np.random.default_rng(seed); z = z.copy() / tau; n = z.shape[-2]; m = r.random(n) < q; o = np.argsort(-z, -1); i = np.arange(n)
    if z.ndim == 2:
        a, b = o[:, 0], o[:, 1]; za, zb = z[i, a].copy(), z[i, b].copy(); z[i[m], a[m]] = zb[m]; z[i[m], b[m]] = za[m]
    else:
        for t in range(z.shape[0]): z[t] = flip(z[t] * tau, q, tau, seed)
    return z
print('--- C. part q des images basculée vers le 2e choix d orig (erreur sûre), le tout adouci par tau')
for tau in (1, 2, 3):
    for q in (0.1, 0.2, 0.3, 0.4): show(f'  tau={tau} q={q}', S(mk(flip(O['fx'], q, tau), flip(O['fa4'], q, tau))))
print('--- D. cible par classe : chaque image reçoit la distribution moyenne des golds s3-s5 de sa classe (oracle de classe), noté contre s3-s5')
for T1 in (1.0,):
    P = np.mean([np.exp(SC.lsm(g['fx'], 1.0)) for g in G[1:]], 0); y = lab['fx']; fx = np.zeros_like(O['fx'], dtype=np.float64)
    for c in np.unique(y): fx[y == c] = np.log(P[y == c].mean(0) + 1e-9)
    show('  oracle de classe (déterministe)', S(mk(fx, np.stack([fx] * 4)), G[1:]))
    # même cible de classe, mais échantillonnée : chaque image tire une classe selon la distribution de sa classe et y met une sortie tranchée
    for conf in (4, 8):
        fx2 = np.zeros_like(fx)
        for c in np.unique(y):
            idx = np.where(y == c)[0]; pc = P[idx].mean(0); pc = pc / pc.sum(); k = rng.choice(100, len(idx), p=pc); fx2[idx] = np.log(pc + 1e-9)[None]; fx2[idx, k] += conf
        show(f'  cible de classe + tirage tranché (+{conf})', S(mk(fx2, np.stack([fx2] * 4)), G[1:]))
print('--- E. sensibilité à une cible de loss fausse : famille C recalée, score de loss si le vrai gold avait (mu, sd)')
Ls = {(tau, q): SC.ce(flip(O['fx'], q, tau), lab['fx']) for tau in (1, 2, 3) for q in (0.05, 0.1, 0.2, 0.3, 0.4)}; Lo = SC.ce(O['fx'], lab['fx'])
for mu, sd in ((0.55, 1.17), (0.9, 2.1), (1.35, 2.5), (2.1, 3.5)):
    best = max(Ls, key=lambda k: -(abs(Ls[k].mean() - mu) + abs(Ls[k].std() - sd))); d0 = abs(Lo.mean() - mu) + abs(Lo.std() - sd)
    line = ' '.join(f"{(mu2, sd2)}:{1 - min(1, (abs(Ls[best].mean() - mu2) + abs(Ls[best].std() - sd2)) / (abs(Lo.mean() - mu2) + abs(Lo.std() - sd2))):.2f}" for mu2, sd2 in ((0.55, 1.17), (0.9, 2.1), (1.35, 2.5), (2.1, 3.5)))
    print(f'  réglé pour ({mu},{sd}) -> tau,q={best}, L={Ls[best].mean():.2f}/{Ls[best].std():.2f} ; score de loss si le vrai gold est {line}')
