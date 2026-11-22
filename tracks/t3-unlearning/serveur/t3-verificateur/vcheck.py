"""t3-verificateur, passage 2 : (1) diagnostics recalcules sur MES sorties (vlogits.py) ; (2) modele de gold ajuste sur 3 envois (1180, 1197, 1265),
puis prediction HORS AJUSTEMENT des autres envois notes et des candidats. usage : python vcheck.py  (numpy seul)"""
import os, sys, json, itertools, numpy as np
from statistics import NormalDist
ROOT = os.environ.get('T3ROOT', 'E:/EAC-lab/t3'); OUT = ROOT + '/t3-verificateur/out'
REAL = {'y1180': 0.003861573, 's1197': 0.0114, 's1265': 0.0273, 'NEGce_all_a2': 0.0219, 'PG_T85': 0.0241, 'PG_T70': 0.0323, 'Z9_b0': 0.0, 'TS_tau4': 0.0057, 'PG_T85_f20_l48c': 0.0735}
REAL.update(json.load(open(OUT + '/real_extra.json')) if os.path.exists(OUT + '/real_extra.json') else {})
FITON = sys.argv[1].split(',') if len(sys.argv) > 1 else ['y1180', 's1197', 's1265']
names = sorted(f[:-4] for f in os.listdir(OUT) if f.endswith('.npz')); assert 'orig' in names
AF = ['a_shift', 'a_crop', 'a_rot', 'a_color', 'a_blur', 'a_noise', 'a_jpeg']; CF = ['c_mix', 'c_blur_strong', 'c_color_strong', 'c_crop_strong']


def lsm(z, t=1.0):
    z = np.asarray(z, np.float64) / t; z = z - z.max(-1, keepdims=True); return z - np.log(np.exp(z).sum(-1, keepdims=True))


def load(n):
    d = dict(np.load(f'{OUT}/{n}.npz')); N = len(d['exact']); i = np.arange(N)
    d['alt'] = np.stack([d[a] for a in AF])[i % len(AF), i]          # groupe 'copies' : l'image i prend la famille i mod 7
    return d


Z = {n: load(n) for n in names}; O = Z['orig']; y = O['exact'].argmax(1); n = len(y); ar = np.arange(n)
cnt = np.bincount(y, minlength=100); FULL = np.where(cnt >= 129)[0]; isf = np.isin(y, FULL)
EMU = []                                                    # sondes construites exactement sur la tete : reconstruites depuis les logits d'orig si le fichier n'a pas ete passe
if 'TS_tau4' not in Z: Z['TS_tau4'] = {k: v / 4.0 for k, v in O.items()}; EMU.append('TS_tau4')
if 'Z9_b0' not in Z:
    Z['Z9_b0'] = {k: v.copy() for k, v in O.items()}
    for v in Z['Z9_b0'].values(): v[:, FULL] = 0.0
    EMU.append('Z9_b0')
for fac in (1.25, 1.5, 2.0):                                 # tete du meilleur fichier multipliee : logits x fac, calcul exact
    if 'PG_T85_f20_l48c' in Z and f'f20_x{fac}' not in Z: Z[f'f20_x{fac}'] = {k: v * fac for k, v in Z['PG_T85_f20_l48c'].items()}; EMU.append(f'f20_x{fac}')
names = sorted(Z)
CE = {k: {s: -lsm(v[s])[ar, y] for s in ['exact', 'alt'] + AF + CF if len(v[s]) == n} for k, v in Z.items()}
L2 = {k: {s: lsm(v[s], 2) for s in ['exact', 'alt'] + CF} for k, v in Z.items()}


def ret(k, c):
    lo, ls = L2['orig'][c], L2[k][c]; kl = float((np.exp(lo) * (lo - ls)).sum(1).mean())
    yy = O[c].argmax(1) if c == 'c_mix' else y; a = (Z[k][c].argmax(1) == yy).mean() / (O[c].argmax(1) == yy).mean(); return float(min(a, 1) / (1 + kl)), kl, float(a)


print('== 1. controle de fichier (chargement strict, float32, fini, taille, tenseurs modifies) ==')
for k in names:
    if k in EMU: print(f'{k:18s} (reconstruit depuis les logits d orig, fichier non repasse)'); continue
    c = json.load(open(f'{OUT}/{k}.check.json'))
    print(f"{k:18s} {c['size_MB']:7.2f} Mo strict={'OK' if 'All keys matched' in c['strict_load'] else 'ECHEC'} f32={c['all_float32']} fini={c['finite']} cles={c['keys_equal']} modifies={c['n_changed']:3d} "
          f"gamma_modifies={len(c['changed_gamma'])} hors_blocs={sorted(c['changed_other'])} max_rel={c['rel_change_max']:.4f} fp16-fp32: max|dlogit|={c['fp16_vs_fp32_max_abs_logit']:.3f} argmax_diff={c['fp16_vs_fp32_argmax_diff']}/500 groupes={c['changed_groups'][0] if c['changed_groups'] else '-'}..{c['changed_groups'][-1] if c['changed_groups'] else '-'}")

print('\n== 2. diagnostics recalcules (fp16, etiquette = classe predite par orig) ==')
print(f"{'fichier':18s} | exactes : m     s     bascule p_yT2 | copies legeres : m     s     bascule | precision : rot   recadr flou_fort coul_forte recadr_fort | KL(o||s)T2 : melange flou_f coul_f recadr_f | reel")
for k in names:
    e = CE[k]['exact']; a = CE[k]['alt']; p2 = np.exp(L2[k]['exact'])[ar, y].mean()
    acc = [(Z[k][s].argmax(1) == y).mean() for s in ['a_rot', 'a_crop', 'c_blur_strong', 'c_color_strong', 'c_crop_strong']]; kl = [ret(k, c)[1] for c in CF]
    print(f"{k:18s} | {e.mean():6.3f} {e.std():6.3f} {(Z[k]['exact'].argmax(1) != y).mean():6.3f} {p2:6.3f} | {a.mean():6.3f} {a.std():6.3f} {(Z[k]['alt'].argmax(1) != y).mean():6.3f} | "
          + ' '.join(f'{v:6.3f}' for v in acc) + ' | ' + ' '.join(f'{v:6.3f}' for v in kl) + ' | ' + (f"{REAL[k]:.4f}" if k in REAL else ''))


def gold(hz, A, kappa, kz, s_t, s_n, rho=0.8, K=3, seed=0):
    rng = np.random.default_rng(seed); out = {}
    u = rng.standard_normal(n); q = NormalDist().inv_cdf(A); dk = [np.sqrt(rho) * u + np.sqrt(1 - rho) * rng.standard_normal(n) for _ in range(K)]; kzn = [(1 + 0.3 * rng.standard_normal(n)).clip(0.3) for _ in range(K)]
    for s in ['exact', 'alt']:
        zo = O['exact'].astype(np.float64); tail = zo.copy(); tail[ar, y] = -np.inf; tail -= tail.max(1, keepdims=True); ces = []; P = np.zeros((n, 100))
        for k in range(K):
            z = s_t * tail + s_n * rng.standard_normal((n, 100)); z[ar, y] = -np.inf
            if hz: z[:, FULL] = -np.inf
            mx = np.where(np.isfinite(z), z, -1e9).max(1); zy = mx + kappa * (q - dk[k])
            if hz: zy = np.where(isf, mx - kz * kzn[k], zy)
            z[ar, y] = zy; z = np.where(np.isfinite(z), z, mx[:, None] - 25.0); ces.append(-lsm(z)[ar, y]); P += np.exp(lsm(z, 2))
        P /= K; ces = np.array(ces); out[s] = dict(mu=ces.mean(1).mean(), sd=ces.std(1).mean(), p=P, lp=np.log(np.clip(P, 1e-12, None)))
        out[s]['ko'] = float((P * (out[s]['lp'] - L2['orig'][s])).sum(1).mean())
    return out


def LP(k, g):
    r = []
    for s in ['exact', 'alt']:
        G = g[s]; Ls, Lo = CE[k][s], CE['orig'][s]; d_s = abs(Ls.mean() - G['mu']) + abs(Ls.std() - G['sd']); d_o = abs(Lo.mean() - G['mu']) + abs(Lo.std() - G['sd'])
        ks = float((G['p'] * (G['lp'] - L2[k][s])).sum(1).mean()); r.append((1 - float(np.clip(d_s / d_o, 0, 1)), 1 - float(np.clip(ks / G['ko'], 0, 1))))
    return (r[0][0] + r[1][0]) / 2, (r[0][1] + r[1][1]) / 2


RET = {k: [ret(k, c)[0] for c in CF] for k in names}
print('\n== 3. modele ajuste sur', FITON, 'puis predictions hors ajustement ==')
for hz, grid in [(False, dict(A=[0.5, 0.6, 0.7, 0.85], kappa=[5, 9, 13], kz=[0], s_t=[0.3, 1.0], s_n=[0.5, 2.0, 4.0])), (True, dict(A=[0.8, 0.9, 0.97], kappa=[5, 9], kz=[6, 10, 14], s_t=[0.3, 1.0], s_n=[0.5, 2.0, 4.0]))]:
    rows = []
    for vals in itertools.product(*grid.values()):
        h = dict(zip(grid.keys(), vals)); g = gold(hz, **h); err = 0
        for k in FITON:
            L, P = LP(k, g); lo_, hi_ = L * P * min(RET[k]), L * P * max(RET[k])
            e = 0 if lo_ <= REAL[k] <= hi_ else min(abs(np.log(max(lo_, 1e-9) / REAL[k])), abs(np.log(max(hi_, 1e-9) / REAL[k]))); err = max(err, e)
        rows.append((err, h, g))
    rows.sort(key=lambda t: t[0]); top = rows[:10]
    print(f"\n--- {'H_Z : 9 classes jamais vues par le gold' if hz else 'H_0 : gold ordinaire'} : {sum(1 for t in rows if t[0] <= np.log(1.5))}/{len(rows)} golds a un facteur 1,5 ; meilleur x{np.exp(rows[0][0]):.2f} ; "
          f"gold des 10 meilleurs : m {min(t[2]['exact']['mu'] for t in top):.1f}-{max(t[2]['exact']['mu'] for t in top):.1f}, s {min(t[2]['exact']['sd'] for t in top):.1f}-{max(t[2]['exact']['sd'] for t in top):.1f} ---")
    print(f"{'fichier':18s} {'S_loss':>7s} {'S_pred':>7s} {'retention (substituts)':>24s} {'score predit (min .. max sur 10 golds x substituts)':>52s}   reel")
    for k in names:
        if k == 'orig': continue
        res = [LP(k, t[2]) for t in top]; lo_ = min(L * P for L, P in res) * min(RET[k]); hi_ = max(L * P for L, P in res) * max(RET[k]); md = float(np.median([L * P for L, P in res])) * float(np.median(RET[k]))
        print(f"{k:18s} {np.median([r[0] for r in res]):7.3f} {np.median([r[1] for r in res]):7.3f} {min(RET[k]):11.2f} .. {max(RET[k]):4.2f} {lo_:20.4f} .. {md:.4f} .. {hi_:.4f}   " + (f"{REAL[k]:.4f}" + ('  (ajustement)' if k in FITON else '') if k in REAL else ''))
