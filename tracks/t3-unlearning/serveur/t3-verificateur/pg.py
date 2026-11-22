"""t3-verificateur : gold de SUBSTITUTION fait des redistillations hors pli de t3-redistill (oof/redistill_k{K}m{n}_logits_s{i}.npy), groupe des images exactes seulement.
1) cale un facteur gamma sur les logits des modeles de substitution pour que leur loss moyenne vaille M_CIBLE (gold ajuste sur les envois : 5,1), et lit leur ecart-type ;
2) note chaque fichier deja passe dans vlogits (out/*.npz) : score de loss, score de prediction, produit ; compare au score reel ;
3) compare des sorties SYNTHETIQUES construites sur un jeu de modeles A et notees sur un jeu B disjoint : bascules dures par rang (recette actuelle) contre trois paliers.
usage : python pg.py [M_CIBLE]"""
import os, sys, numpy as np
ROOT = os.environ.get('T3ROOT', 'E:/EAC-lab/t3'); ME = ROOT + '/t3-verificateur'; OUT = ME + '/out'; OOF = ME + '/oof'; MC = float(sys.argv[1]) if len(sys.argv) > 1 else 5.1
REAL = {'PG_T85_f20_l814c': .1487, 'PG_T85_f20_l1218c': .1707, 'PG_T85_f30_l814c': .1962, 'PG_T85_f30_l1218c': .1998, 'PG_T85_f35_l814c': .2179, 'PG_T85_f40_l814c': .2400, 'PG_T85_f50_l814c': .2779,
        'PG_T85_f60_l814c': .3009, 'PGho_f60_l814c': .2701, 'PG_T85_f70_l814c': .3159, 'PG_RDt24_60_l915c': .3273, 'PG_RDt24_70_l915c': .3502, 'PG_RDt24_70_l1016c': .3653, 'PG_RDt24_70_l1218c': .3682,
        'PG_RDt24_80_l915c': .3638, 'PG_RDm3t24_70_l1016c': .3676, 'PG_RDm3t24_80_l1016c': .3713, 'PG_RDt24n_70_l1016c': .3663, 'PG_RD6m4t24_70_l1117c': .3654, 'PG_RDt24r_60_l1016c': .3648,
        'PG_RDt24r_65_l1016c': .3743, 'PG_RDt24r_70_l1016c': .3888, 'SOUPs1': .3942, 'PG_RDt24r_75_l1016c': .3785, 'PG_RDt24r_80_l1016c': .3657, 'SOUP4_RDt24r_70': .3838, 'PD_f20_l814': .0916, 'PG_T85_f20_l48c': .0735}
CAND = ['PG_RDt24rRV2_70_l8515c', 'PGB6_RDt24r_70', 'PG_RDp_70_l1016c', 'PG_RDt24r_72_l1016c', 'PG_RDt24r_68_l1016c', 'PG_RDt24r_70_l1117c', 'PG_RDt24rc_70_l1117c', 'PG_RDt24r_70_l1016c_s2']


def lsm(z, t=1.0):
    z = np.asarray(z, np.float64) / t; z = z - z.max(-1, keepdims=True); return z - np.log(np.exp(z).sum(-1, keepdims=True))


O = np.load(OUT + '/orig.npz')['exact'].astype(np.float64); y = O.argmax(1); n = len(y); ar = np.arange(n); LO2 = lsm(O, 2); CEO = -lsm(O)[ar, y]
def ld(k, m): return [np.load(f'{OOF}/redistill_k{k}m{m}_logits_s{i}.npy').astype(np.float64) for i in range(m)]
M4, M6, M8 = ld(4, 8), ld(6, 4), ld(8, 2)


def gam(Z, mc):   # facteur sur les logits tel que la loss moyenne du modele vaille mc
    lo, hi = 0.2, 30.0
    for _ in range(50):
        g = (lo + hi) / 2
        if (-lsm(Z * g)[ar, y]).mean() < mc: lo = g
        else: hi = g
    return g


def mkgold(models, mc):
    ces, P, gs = [], np.zeros((n, 100)), []
    for Z in models:
        g = gam(Z, mc) if mc else 1.0; gs.append(g); ces.append(-lsm(Z * g)[ar, y]); P += np.exp(lsm(Z * g, 2))
    P /= len(models); ces = np.array(ces); lp = np.log(np.clip(P, 1e-12, None))
    return dict(mu=ces.mean(1).mean(), sd=ces.std(1).mean(), p=P, lp=lp, ko=float((P * (lp - LO2)).sum(1).mean()), g=float(np.mean(gs)), acc=float(np.mean([(Z.argmax(1) == y).mean() for Z in models])))


def score(z, G, per=False):
    ce = -lsm(z)[ar, y]; d = abs(ce.mean() - G['mu']) + abs(ce.std() - G['sd']); do = abs(CEO.mean() - G['mu']) + abs(CEO.std() - G['sd']); sl = 1 - min(1, d / do)
    kl = (G['p'] * (G['lp'] - lsm(z, 2))).sum(1); sp = 1 - min(1, kl.mean() / G['ko'])
    return (sl, sp, sl * sp, kl) if per else (sl, sp, sl * sp)


GOLDS = {'6 blocs x4': M6, '8 blocs x2': M8, '6+8 blocs x6': M6 + M8, '4 blocs x8': M4}
print(f'== 1. golds de substitution, logits multiplies par gamma pour une loss moyenne de {MC} ==')
G = {}
for k, ms in GOLDS.items():
    G[k] = mkgold(ms, MC); g0 = mkgold(ms, 0)
    print(f"{k:14s} precision {G[k]['acc']:.3f} | brut : m {g0['mu']:.2f} s {g0['sd']:.2f} | cale : gamma {G[k]['g']:.2f} m {G[k]['mu']:.2f} s {G[k]['sd']:.2f} | KL(gold||orig) T2 {G[k]['ko']:.3f} | p_y T2 moyen du gold {G[k]['p'][ar, y].mean():.3f}")

print('\n== 2. fichiers notes par le serveur : score local sur le gold de substitution (images exactes) contre score reel ==')
have = [k for k in REAL if os.path.exists(f'{OUT}/{k}.npz')]; Zf = {k: np.load(f'{OUT}/{k}.npz')['exact'].astype(np.float64) for k in have + [c for c in CAND if os.path.exists(f'{OUT}/{c}.npz')]}
from itertools import combinations
def spear(a, b): ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])
for gk in GOLDS:
    r = np.array([REAL[k] for k in have]); S = np.array([score(Zf[k], G[gk]) for k in have]); top = [i for i, k in enumerate(have) if REAL[k] >= 0.30]
    print(f"gold {gk:14s} : Spearman(reel, local) sur {len(have)} fichiers {spear(r, S[:, 2]):.3f} ; sur les {len(top)} fichiers >= 0,30 : {spear(r[top], S[top, 2]):.3f} ; avec S_pred seul {spear(r[top], S[top, 1]):.3f}")
gk = '6+8 blocs x6'; print(f'\ndetail, gold {gk} :'); print(f"{'fichier':26s} {'S_loss':>7s} {'S_pred':>7s} {'local':>7s} {'reel':>7s} {'reel/local':>10s}")
for k in sorted(have, key=lambda k: REAL[k]):
    a, b, c = score(Zf[k], G[gk]); print(f'{k:26s} {a:7.3f} {b:7.3f} {c:7.3f} {REAL[k]:7.4f} {REAL[k] / max(c, 1e-9):10.3f}')
print('candidats non notes :')
for k in CAND:
    if k in Zf: a, b, c = score(Zf[k], G[gk]); print(f'{k:26s} {a:7.3f} {b:7.3f} {c:7.3f}')

print('\n== 3. sorties synthetiques : construites sur A (8 modeles a 4 blocs + 2 a 6 blocs), notees sur B (2 a 6 blocs + 2 a 8 blocs) ==')
A = M4 + M6[:2]; GB = mkgold(M6[2:] + M8, MC); gA = [gam(Z, MC) for Z in A]
PA = np.mean([np.exp(lsm(Z * g, 2)) for Z, g in zip(A, gA)], 0)                    # moyenne des distributions T=2 des modeles A cales
piA = np.mean([(Z.argmax(1) != y) for Z in A], 0)                                   # part des modeles A qui se trompent, par image
W = PA.copy(); W[ar, y] = 0; W /= W.sum(1, keepdims=True); rank = np.argsort(-(1 - PA[ar, y]))   # repartition de la masse fausse ; rang = probabilite de la vraie classe la plus basse d'abord
rng = np.random.default_rng(0); u = rng.random(n)


def build(tiers, w_sharp=1.0):
    """tiers : liste (part, mode, val) du haut du rang vers le bas ; mode 'ce' : loss T=1 visee val +- 3 ; mode 'py' : probabilite T=2 de la vraie classe = val ; le reste : p_y T=2 = 0,8."""
    Ws = W ** w_sharp; Ws /= Ws.sum(1, keepdims=True); py = np.full(n, 0.8); pos = 0
    for part, mode, val in tiers:
        idx = rank[pos:pos + int(round(part * n))]; pos += len(idx)
        if mode == 'py': py[idx] = val
        else:
            c = val + 6 * (u[idx] - 0.5); s2 = (Ws[idx] ** 2).sum(1); r = np.sqrt(np.exp(-c) * s2 / (1 - np.exp(-c))); py[idx] = r / (1 + r)
    P = (1 - py)[:, None] * Ws; P[ar, y] = py; return 2 * np.log(np.clip(P, 1e-30, None))


def show(name, z):
    ce = -lsm(z)[ar, y]; a, b, c = score(z, GB); print(f'{name:58s} m {ce.mean():5.2f} s {ce.std():5.2f} | S_loss {a:.3f} S_pred {b:.3f} produit {c:.3f}'); return c


print(f"gold B : m {GB['mu']:.2f} s {GB['sd']:.2f} ; part des modeles A en erreur par image : 0 -> {np.mean(piA == 0):.2f}, entre 0 et 1 -> {np.mean((piA > 0) & (piA < 1)):.2f}, tous -> {np.mean(piA == 1):.2f}")
for k in ['SOUPs1', 'PG_RDt24r_70_l1016c', 'PG_RDt24_70_l1016c', 'PG_T85_f70_l814c']:
    if k in Zf: a, b, c = score(Zf[k], GB); ce = -lsm(Zf[k])[ar, y]; print(f"{'fichier reel ' + k:58s} m {ce.mean():5.2f} s {ce.std():5.2f} | S_loss {a:.3f} S_pred {b:.3f} produit {c:.3f}")
show('moyenne des distributions T=2 de A (aucune bascule forcee)', 2 * np.log(np.clip(PA, 1e-30, None)))
best = (0, None)
for sh in (0.5, 0.6, 0.7, 0.8):
    for c in (6.0, 7.5, 9.0):
        v = show(f'bascules dures : {int(sh * 100)} % a loss {c}', build([(sh, 'ce', c)])); best = max(best, (v, f'{sh} {c}'))
print('  meilleur des bascules dures :', best)
best3 = (0, None)
for hi in (0.3, 0.4, 0.5):
    for mid in (0.2, 0.3, 0.4):
        for c in (9.0, 11.0, 13.0):
            for q in (0.15, 0.3, 0.45):
                z = build([(hi, 'ce', c), (mid, 'py', q)]); v = score(z, GB)[2]
                if v > best3[0]: best3 = (v, (hi, mid, c, q))
hi, mid, c, q = best3[1]; show(f'TROIS PALIERS (meilleur) : {int(hi * 100)} % a loss {c}, {int(mid * 100)} % a p_y T2 {q}', build([(hi, 'ce', c), (mid, 'py', q)]))
for ws in (1.5, 2.0):
    show(f'  idem, masse fausse resserree (puissance {ws})', build([(hi, 'ce', c), (mid, 'py', q)], ws)); show(f'  bascules dures 70 % a 7.5, masse fausse resserree ({ws})', build([(0.7, 'ce', 7.5)], ws))
# continu : p_y T2 de la moyenne A, puis exposant pour remonter la loss
for e in (1.0, 1.5, 2.0, 3.0):
    P = PA ** e; P /= P.sum(1, keepdims=True); show(f'moyenne A elevee a la puissance {e}', 2 * np.log(np.clip(P, 1e-30, None)))
