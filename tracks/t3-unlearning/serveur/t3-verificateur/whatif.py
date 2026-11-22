"""t3-verificateur : que predit le modele (golds H_0 ajustes sur les envois notes) pour des candidats SYNTHETIQUES ?
Candidat synthetique = sorties d'un envoi note (base), ou l'on fait basculer une part f d'images vers une classe fausse avec une loss c.
But : ou est l'optimum en (f, c), et combien coute un mauvais choix d'images / de classes fausses. Le gold du modele tire ses erreurs au hasard
(il ne sait pas quelles images le vrai gold rate) : le 'choix aligne' ci-dessous est une borne haute, le 'choix au hasard' une borne basse.
usage : python whatif.py BASE   (BASE = nom d'un npz de out/, ex. PG_T85)"""
import os, sys, json, itertools, numpy as np
from statistics import NormalDist
ROOT = os.environ.get('T3ROOT', 'E:/EAC-lab/t3'); OUT = ROOT + '/t3-verificateur/out'; BASE = sys.argv[1] if len(sys.argv) > 1 else 'PG_T85'
REAL = {'y1180': 0.003861573, 's1197': 0.0114, 's1265': 0.0273, 'NEGce_all_a2': 0.0219, 'PG_T85': 0.0241, 'PG_T70': 0.0323, 'TS_tau4': 0.0057, 'PG_T85_f20_l48c': 0.0735}
AF = ['a_shift', 'a_crop', 'a_rot', 'a_color', 'a_blur', 'a_noise', 'a_jpeg']; CF = ['c_mix', 'c_blur_strong', 'c_color_strong', 'c_crop_strong']


def lsm(z, t=1.0):
    z = np.asarray(z, np.float64) / t; z = z - z.max(-1, keepdims=True); return z - np.log(np.exp(z).sum(-1, keepdims=True))


def load(nm):
    d = dict(np.load(f'{OUT}/{nm}.npz')); N = len(d['exact']); i = np.arange(N); d['alt'] = np.stack([d[a] for a in AF])[i % len(AF), i]; return d


O = load('orig'); y = O['exact'].argmax(1); n = len(y); ar = np.arange(n)
FIT = [k for k in REAL if os.path.exists(f'{OUT}/{k}.npz')]; Z = {k: load(k) for k in set(FIT + [BASE])}
LO2 = {s: lsm(O[s], 2) for s in ['exact', 'alt'] + CF}; CEO = {s: -lsm(O[s])[ar, y] for s in ['exact', 'alt']}


def ret(S):
    r = []
    for c in CF:
        ls = lsm(S[c], 2); kl = float((np.exp(LO2[c]) * (LO2[c] - ls)).sum(1).mean()); yy = O[c].argmax(1) if c == 'c_mix' else y
        r.append(min((S[c].argmax(1) == yy).mean() / (O[c].argmax(1) == yy).mean(), 1) / (1 + kl))
    return r


def gold(A, kappa, s_t, s_n, rho=0.8, K=3, seed=0):
    rng = np.random.default_rng(seed); q = NormalDist().inv_cdf(A); u = rng.standard_normal(n); out = {}
    dk = [np.sqrt(rho) * u + np.sqrt(1 - rho) * rng.standard_normal(n) for _ in range(K)]; nz = [rng.standard_normal((n, 100)) for _ in range(K)]
    zo = O['exact'].astype(np.float64); tail = zo.copy(); tail[ar, y] = -np.inf; tail -= tail.max(1, keepdims=True); ces = []; P = np.zeros((n, 100)); wrong = np.zeros(n); wcls = []
    for k in range(K):
        z = s_t * tail + s_n * nz[k]; z[ar, y] = -np.inf; mx = z.max(1); z[ar, y] = mx + kappa * (q - dk[k]); ces.append(-lsm(z)[ar, y]); P += np.exp(lsm(z, 2)); wrong += (z.argmax(1) != y); wcls.append(z.argmax(1))
    P /= K; ces = np.array(ces); lp = np.log(np.clip(P, 1e-12, None)); Pw = P.copy(); Pw[ar, y] = 0
    return dict(mu=ces.mean(1).mean(), sd=ces.std(1).mean(), p=P, lp=lp, ko={s: float((P * (lp - LO2[s])).sum(1).mean()) for s in ['exact', 'alt']}, nwrong=wrong, wtop=Pw.argmax(1), acc=1 - wrong.mean() / K)


def LP(S, g):
    r = []
    for s in ['exact', 'alt']:
        Ls = -lsm(S[s])[ar, y]; d_s = abs(Ls.mean() - g['mu']) + abs(Ls.std() - g['sd']); d_o = abs(CEO[s].mean() - g['mu']) + abs(CEO[s].std() - g['sd'])
        ks = float((g['p'] * (g['lp'] - lsm(S[s], 2))).sum(1).mean()); r.append((1 - float(np.clip(d_s / d_o, 0, 1)), 1 - float(np.clip(ks / g['ko'][s], 0, 1))))
    return (r[0][0] + r[1][0]) / 2, (r[0][1] + r[1][1]) / 2


rows = []
for A, kappa, s_t, s_n in itertools.product([0.5, 0.6, 0.7, 0.8], [5, 9, 13], [0.3, 1.0], [0.5, 2.0, 4.0]):
    g = gold(A, kappa, s_t, s_n); err = 0
    for k in FIT:
        L, P = LP(Z[k], g); R = ret(Z[k]); lo_, hi_ = L * P * min(R), L * P * max(R)
        e = 0 if lo_ <= REAL[k] <= hi_ else min(abs(np.log(max(lo_, 1e-9) / REAL[k])), abs(np.log(max(hi_, 1e-9) / REAL[k]))); err = max(err, e)
    rows.append((err, dict(A=A, kappa=kappa, s_t=s_t, s_n=s_n), g))
rows.sort(key=lambda t: t[0]); top = rows[:6]
print(f'ajustement H_0 sur {len(FIT)} envois notes : {FIT}')
for err, h, g in top: print(f"  x{np.exp(err):.2f} {h} | gold precision {g['acc']:.2f} m {g['mu']:.2f} s {g['sd']:.2f}")


def flip(S, idx, cls, c):
    """Fait basculer les images idx vers les classes cls avec une loss c : logit de la vraie classe abaisse, logit de la classe fausse releve au maximum."""
    out = {}
    for s in ['exact', 'alt']:
        z = S[s].astype(np.float64).copy(); zi = z[idx]; top_ = zi.max(1); zi[np.arange(len(idx)), cls] = top_ + 2.0; zi[np.arange(len(idx)), y[idx]] = top_ + 2.0 - c; z[idx] = zi; out[s] = z
    for c_ in CF: out[c_] = S[c_]
    return out


B = Z[BASE]; RB = ret(B); run2 = O['exact'].astype(np.float64).copy(); run2[ar, y] = -np.inf; runner = run2.argmax(1); rng = np.random.default_rng(1)
done = B['exact'].argmax(1) != y
print(f'\nbase {BASE} : deja {done.mean():.3f} de classes changees ; retention des substituts {min(RB):.2f} .. {max(RB):.2f} (supposee INCHANGEE par les bascules ajoutees : hypothese optimiste)')
print(f"{'part basculee f':>16s} {'loss c':>7s} | {'m':>5s} {'s':>5s} | choix des images au hasard : S_loss S_pred score(min..max) | choix aligne sur les erreurs du gold du modele : S_loss S_pred score(min..max)")
for f in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    for c in [6.0, 10.0, 14.0]:
        if f == 0.0 and c > 6.0: continue
        res = {}
        for mode in ['hasard', 'aligne']:
            sc = []; Ls_ = []; Ps_ = []
            for err, h, g in top:
                k = int(round(f * n)); free = np.where(~done)[0]
                if mode == 'hasard': idx = rng.choice(free, min(k, len(free)), replace=False); cls = runner[idx]
                else: order = free[np.argsort(-g['nwrong'][free] + 1e-3 * rng.random(len(free)))]; idx = order[:k]; cls = g['wtop'][idx]
                S = flip(B, idx, cls, c) if k else B; L, P = LP(S, g); Ls_.append(L); Ps_.append(P); sc += [L * P * min(RB), L * P * max(RB)]
                if mode == 'hasard' and err == top[0][0]: e_ = -lsm(S['exact'])[ar, y]; ms = (e_.mean(), e_.std())
            res[mode] = (np.median(Ls_), np.median(Ps_), min(sc), max(sc))
        print(f"{f:16.2f} {c:7.1f} | {ms[0]:5.2f} {ms[1]:5.2f} | {res['hasard'][0]:26.3f} {res['hasard'][1]:6.3f} {res['hasard'][2]:.3f}..{res['hasard'][3]:.3f} | {res['aligne'][0]:38.3f} {res['aligne'][1]:6.3f} {res['aligne'][2]:.3f}..{res['aligne'][3]:.3f}")
