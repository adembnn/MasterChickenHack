"""t3-verificateur : ou est le gold (m_g, s_g) ? Ajustement sur des PAIRES de fichiers notes qui ne different que par la loss visee des bascules
(meme part, meme source de classe) : le rapport des scores reels y est suppose egal au rapport des scores de loss. Puis prediction des nouveaux fichiers.
usage : python fitk.py"""
import os, numpy as np
ROOT = os.environ.get('T3ROOT', 'E:/EAC-lab/t3'); OUT = ROOT + '/t3-verificateur/out'
AF = ['a_shift', 'a_crop', 'a_rot', 'a_color', 'a_blur', 'a_noise', 'a_jpeg']
def lsm(z): z = np.asarray(z, np.float64); z = z - z.max(-1, keepdims=True); return z - np.log(np.exp(z).sum(-1, keepdims=True))
O = dict(np.load(OUT + '/orig.npz')); y = O['exact'].argmax(1); n = len(y); ar = np.arange(n)
def st(nm):
    d = dict(np.load(f'{OUT}/{nm}.npz')); alt = np.stack([d[a] for a in AF])[ar % 7, ar]; r = []
    for z in (d['exact'], alt): ce = -lsm(z)[ar, y]; r.append((ce.mean(), ce.std()))
    return r
SO = st('orig')
def sl(S, mg, sg, dc=0.0):   # dc : decalage du gold sur le groupe copies (m et s)
    v = []
    for gi, ((m, s), (mo, so)) in enumerate(zip(S, SO)):
        a, b = mg - dc * gi, sg - dc * gi; v.append(1 - min(1, (abs(m - a) + abs(s - b)) / (abs(mo - a) + abs(so - b))))
    return (v[0] + v[1]) / 2
P = 'PG_T85_'; PAIRS = [(P + 'f20_l814c', P + 'f20_l1218c', 0.1487, 0.1707), (P + 'f30_l814c', P + 'f30_l1218c', 0.1962, 0.1998), ('PG_RDt24_70_l915c', 'PG_RDt24_70_l1016c', 0.3502, 0.3653), ('PG_RDt24_80_l915c', 'PG_RDm3t24_80_l1016c', 0.3638 * 0.3676 / 0.3653, 0.3713), ('PG_RDt24_70_l1016c', 'PG_RDt24_70_l1218c', 0.3653, 0.3682)][:int(os.environ.get('NP', 5))]
ST = {}
for a, b, _, _ in PAIRS: ST[a] = st(a); ST[b] = st(b)
print('orig', SO); [print(k, [(round(m, 2), round(s, 2)) for m, s in v]) for k, v in ST.items()]
R7 = 'PG_RDt24r_70_l1016c'; NEW = [('PG_RDt24rn_70_l1016c', R7, 0.3888), ('PG_RDt24r_80_l1016c', R7, 0.3888), ('PG_RDt24r_60_l1016c', R7, 0.3888), ('PG_RDt24r_70_l1117c', R7, 0.3888), ('PG_RDt24rc_70_l1016c', R7, 0.3888), ('PG_RDt24r6_70_l1016c', R7, 0.3888),
       ('PG_RDt24_70_l1117c', 'PG_RDt24_70_l1016c', 0.3653)]
NEW = [t for t in NEW if os.path.exists(f'{OUT}/{t[0]}.npz')]
for a, b, _ in NEW: ST[a] = st(a); ST[b] = st(b)
rows = []
for mg in np.arange(3.6, 8.01, 0.2):
    for sg in np.arange(4.0, 7.01, 0.1):
        for dc in (0.0, 0.3, 0.6):
            e = max(abs(np.log(sl(ST[b], mg, sg, dc) / sl(ST[a], mg, sg, dc) / (rb / ra))) for a, b, ra, rb in PAIRS); rows.append((e, mg, sg, dc))
rows.sort(); TOL = float(os.environ.get('TOL', 0.02)); ok = [r for r in rows if r[0] <= TOL]
print(f'golds compatibles ({len(PAIRS)} rapports, tolerance {TOL}) : {len(ok)}/{len(rows)} ; meilleur ecart {rows[0][0]:.4f}')
for dc in (0.0, 0.3, 0.6):
    o = [r for r in ok if r[3] == dc]
    if o: print(f'  decalage copies {dc} : m_g {min(r[1] for r in o):.1f}-{max(r[1] for r in o):.1f}  s_g {min(r[2] for r in o):.1f}-{max(r[2] for r in o):.1f}  n={len(o)}')
for r in rows[:8]: print('  ', f'ecart {r[0]:.4f} m_g {r[1]:.1f} s_g {r[2]:.1f} dc {r[3]}')
use = ok if len(ok) >= 5 else rows[:30]
print('\npredictions (score de la base x rapport des scores de loss ; min / mediane / max sur les golds compatibles)')
for a, b, rb in NEW:
    v = np.array([rb * sl(ST[a], mg, sg, dc) / sl(ST[b], mg, sg, dc) for _, mg, sg, dc in use]); print(f'{a:26s} base {b:24s} {rb:.4f} -> {v.min():.4f} / {np.median(v):.4f} / {v.max():.4f}   exact {ST[a][0][0]:.2f}/{ST[a][0][1]:.2f} copies {ST[a][1][0]:.2f}/{ST[a][1][1]:.2f}')
