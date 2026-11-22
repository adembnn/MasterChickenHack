"""t3-verificateur : quel gold de substitution range correctement les fichiers notes par le serveur, surtout a l'interieur de la famille 'rang redistill' ?
Pour chaque gold candidat (jeu de redistillations hors pli, calage des logits) : correlation de rang entre le score reel et (score de loss ajuste x score de prediction local),
sur tous les fichiers notes et sur la famille r seule ; signe de l'ecart rc - SOUPs1 (reel : -0,019) ; part de la KL de SOUPs1 portee par la case 'basculee alors que le gold a juste'.
usage : python pgv.py"""
import os, numpy as np
ROOT = os.environ['T3ROOT']; ME = ROOT + '/t3-verificateur'; OUT = ME + '/out'; OOF = ME + '/oof'; MG, SG = 5.1, 4.7
REAL = {'PG_T85_f20_l814c': .1487, 'PG_T85_f30_l814c': .1962, 'PG_T85_f40_l814c': .2400, 'PG_T85_f50_l814c': .2779, 'PG_T85_f60_l814c': .3009, 'PGho_f60_l814c': .2701, 'PG_T85_f70_l814c': .3159,
        'PG_RDt24_60_l915c': .3273, 'PG_RDt24_70_l915c': .3502, 'PG_RDt24_70_l1016c': .3653, 'PG_RDt24_70_l1218c': .3682, 'PG_RDt24_80_l915c': .3638, 'PG_RDm3t24_70_l1016c': .3676, 'PG_RDm3t24_80_l1016c': .3713,
        'PG_RDt24n_70_l1016c': .3663, 'PG_RD6m4t24_70_l1117c': .3654, 'PG_RDt24r_60_l1016c': .3648, 'PG_RDt24r_65_l1016c': .3743, 'PG_RDt24r_70_l1016c': .3888, 'SOUPs1': .3942, 'PG_RDt24r_72_l1016c': .3849,
        'PG_RDt24r_75_l1016c': .3785, 'PG_RDt24r_80_l1016c': .3657, 'SOUP4_RDt24r_70': .3838, 'PG_RDt24rRV2_70_l8515c': .3847, 'PG_RDt24rc_70_l1117c': .3753}
RFAM = ['PG_RDt24r_60_l1016c', 'PG_RDt24r_65_l1016c', 'PG_RDt24r_70_l1016c', 'SOUPs1', 'PG_RDt24r_72_l1016c', 'PG_RDt24r_75_l1016c', 'PG_RDt24r_80_l1016c', 'SOUP4_RDt24r_70', 'PG_RDt24rRV2_70_l8515c', 'PG_RDt24rc_70_l1117c']


def lsm(z, t=1.0):
    z = np.asarray(z, np.float64) / t; z = z - z.max(-1, keepdims=True); return z - np.log(np.exp(z).sum(-1, keepdims=True))


O = np.load(OUT + '/orig.npz')['exact'].astype(np.float64); y = O.argmax(1); n = len(y); ar = np.arange(n); LO2 = lsm(O, 2); CEO = -lsm(O)[ar, y]
def ld(k, m): return [np.load(f'{OOF}/redistill_k{k}m{m}_logits_s{i}.npy').astype(np.float64) for i in range(m)]
M4, M6, M8 = ld(4, 8), ld(6, 4), ld(8, 2)
have = [k for k in REAL if os.path.exists(f'{OUT}/{k}.npz')]; Z = {k: np.load(f'{OUT}/{k}.npz')['exact'].astype(np.float64) for k in have}
SL = {}
for k in have: ce = -lsm(Z[k])[ar, y]; SL[k] = 1 - min(1, (abs(ce.mean() - MG) + abs(ce.std() - SG)) / (abs(CEO.mean() - MG) + abs(CEO.std() - SG)))


def gam(Zm, mc):
    lo, hi = 0.2, 30.0
    for _ in range(50):
        g = (lo + hi) / 2
        if (-lsm(Zm * g)[ar, y]).mean() < mc: lo = g
        else: hi = g
    return g


def spear(a, b): ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])
rf = [k for k in RFAM if k in have]; r_all = np.array([REAL[k] for k in have]); r_rf = np.array([REAL[k] for k in rf])
print(f'fichiers notes disponibles : {len(have)} ; famille rang redistill : {len(rf)}')
print(f"{'gold':28s} {'calage':>10s} | rang tous  rang famille r | S_pred SOUPs1  rc  (rc - SOUPs1, reel -0,019) | case basculee/gold juste : part des images, part de la KL | gold faux")
for gname, ms in [('6+8 blocs (6 modeles)', M6 + M8), ('6 blocs (4)', M6), ('8 blocs (2)', M8), ('4 blocs, partitions 3 a 7 (5)', M4[3:]), ('4 blocs, toutes (8)', M4), ('4 blocs, partitions 0 a 2 (3)', M4[:3])]:
    for cal in ('m=5.1', 'm=3.0', 'x1', 'x2'):
        gs = [gam(m_, float(cal[2:])) if cal.startswith('m=') else float(cal[1:]) for m_ in ms]
        P = np.mean([np.exp(lsm(m_ * g, 2)) for m_, g in zip(ms, gs)], 0); LP = np.log(np.clip(P, 1e-12, None)); KO = float((P * (LP - LO2)).sum(1).mean())
        sp = {k: 1 - float((P * (LP - lsm(Z[k], 2))).sum(1).mean()) / KO for k in have}; loc = np.array([SL[k] * sp[k] for k in have]); locr = np.array([SL[k] * sp[k] for k in rf])
        z = Z['SOUPs1']; kl = (P * (LP - lsm(z, 2))).sum(1); fl = z.argmax(1) != y; gr = P.argmax(1) == y; c3 = fl & gr
        print(f"{gname:28s} {cal:>10s} | {spear(r_all, loc):9.3f} {spear(r_rf, locr):14.3f} | {sp['SOUPs1']:13.3f} {sp['PG_RDt24rc_70_l1117c']:.3f} {sp['PG_RDt24rc_70_l1117c'] - sp['SOUPs1']:+.3f} | {c3.mean():24.3f} {kl[c3].sum() / kl.sum():8.3f} | {1 - gr.mean():.3f}")
print('\nfamille r, detail (gold 4 blocs partitions 3 a 7, calage m=5.1) : reel, S_loss, S_pred')
ms = M4[3:]; P = np.mean([np.exp(lsm(m_ * gam(m_, 5.1), 2)) for m_ in ms], 0); LP = np.log(np.clip(P, 1e-12, None)); KO = float((P * (LP - LO2)).sum(1).mean())
for k in sorted(rf, key=lambda k: -REAL[k]): print(f'  {k:28s} {REAL[k]:.4f} {SL[k]:.3f} {1 - float((P * (LP - lsm(Z[k], 2))).sum(1).mean()) / KO:.3f}')
