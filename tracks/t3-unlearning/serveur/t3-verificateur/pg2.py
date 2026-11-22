"""t3-verificateur : suite de pg.py. (a) note les CIBLES d'entrainement (cache/tgt_*.pt de t3-methodes) et les fichiers obtenus, sur le gold de substitution 6+8 blocs :
ecart cible -> fichier = ce que l'entrainement perd. (b) decompose la KL(gold||fichier) par groupe d'images. (c) sorties synthetiques avec le score de loss calcule
sur le gold AJUSTE sur les envois reels (m 5,1 ; s 4,7) et le score de prediction sur le gold de substitution B.
usage : python pg2.py [gamma_fixe]   (sans argument : gamma cale pour une loss moyenne de 5,1)"""
import os, sys, numpy as np, torch
ROOT = os.environ.get('T3ROOT'); ME = ROOT + '/t3-verificateur'; OUT = ME + '/out'; OOF = ME + '/oof'; CACHE = ROOT + '/t3-methodes/cache'; GF = float(sys.argv[1]) if len(sys.argv) > 1 else 0; MG, SG = 5.1, 4.7


def lsm(z, t=1.0):
    z = np.asarray(z, np.float64) / t; z = z - z.max(-1, keepdims=True); return z - np.log(np.exp(z).sum(-1, keepdims=True))


O = np.load(OUT + '/orig.npz')['exact'].astype(np.float64); y = O.argmax(1); n = len(y); ar = np.arange(n); LO2 = lsm(O, 2); CEO = -lsm(O)[ar, y]
def ld(k, m): return [np.load(f'{OOF}/redistill_k{k}m{m}_logits_s{i}.npy').astype(np.float64) for i in range(m)]
M4, M6, M8 = ld(4, 8), ld(6, 4), ld(8, 2)


def gam(Z, mc=5.1):
    if GF: return GF
    lo, hi = 0.2, 30.0
    for _ in range(50):
        g = (lo + hi) / 2
        if (-lsm(Z * g)[ar, y]).mean() < mc: lo = g
        else: hi = g
    return g


def mkgold(models):
    P = np.mean([np.exp(lsm(Z * gam(Z), 2)) for Z in models], 0); lp = np.log(np.clip(P, 1e-12, None)); return dict(p=P, lp=lp, ko=float((P * (lp - LO2)).sum(1).mean()), ko_i=(P * (lp - LO2)).sum(1))


def sloss(z): ce = -lsm(z)[ar, y]; return 1 - min(1, (abs(ce.mean() - MG) + abs(ce.std() - SG)) / (abs(CEO.mean() - MG) + abs(CEO.std() - SG)))
def spred(z, G): kl = (G['p'] * (G['lp'] - lsm(z, 2))).sum(1); return 1 - min(1, kl.mean() / G['ko']), kl


G = mkgold(M6 + M8); GB = mkgold(M6[2:] + M8)
print(f"gamma {'fixe ' + str(GF) if GF else 'cale (m = 5,1)'} ; KL(gold||orig) {G['ko']:.3f} ; score de loss sur le gold ajuste aux envois (m {MG}, s {SG})")
print('\n== a. cible d entrainement contre fichier obtenu (gold 6+8 blocs) =='); print(f"{'nom':26s} | cible : m     s    S_loss S_pred | fichier : m     s    S_loss S_pred | reel")
REAL = {'PG_RDt24r_70_l1016c': .3888, 'PG_RDt24_70_l1016c': .3653, 'PG_T85_f70_l814c': .3159, 'PG_RDt24r_60_l1016c': .3648, 'PG_RDt24r_80_l1016c': .3657, 'PG_RDt24_70_l1218c': .3682}
ZF = {}
for f, t in [('PG_T85_f70_l814c', 'PG_T85_f70_l814'), ('PG_RDt24_70_l1016c', 'RDt24_70_l1016'), ('PG_RDt24_70_l1218c', 'RDt24_70_l1218'), ('PG_RDt24r_60_l1016c', 'RDt24r_60_l1016'), ('PG_RDt24r_70_l1016c', 'RDt24r_70_l1016'),
             ('PG_RDt24r_80_l1016c', 'RDt24r_80_l1016'), ('PG_RDp_70_l1016c', 'RDp_70_l1016'), ('PG_RDt24rc_70_l1117c', 'RDt24rc_70_l1117'), ('PGB6_RDt24r_70', 'RDt24r_70_l1016'), ('PG_RDt24rRV2_70_l8515c', 'RDt24rRV2_70_l8515')]:
    row = f'{f:26s} |'
    cands = [t, t.replace('PG_T85_', ''), t.replace('RV2', '')]; zt = None
    for c in cands:
        if os.path.exists(f'{CACHE}/tgt_{c}.pt'): zt = torch.load(f'{CACHE}/tgt_{c}.pt', map_location='cpu')['z'].double().numpy(); break
    if zt is not None: ce = -lsm(zt)[ar, y]; row += f'       {ce.mean():5.2f} {ce.std():5.2f} {sloss(zt):6.3f} {spred(zt, G)[0]:6.3f} |'; ZF['tgt:' + f] = zt
    else: row += '       cible introuvable          |'
    if os.path.exists(f'{OUT}/{f}.npz'): z = np.load(f'{OUT}/{f}.npz')['exact'].astype(np.float64); ZF[f] = z; ce = -lsm(z)[ar, y]; row += f'         {ce.mean():5.2f} {ce.std():5.2f} {sloss(z):6.3f} {spred(z, G)[0]:6.3f} | ' + (f'{REAL[f]:.4f}' if f in REAL else '')
    print(row)

print('\n== b. ou part la KL(gold||fichier) ? base PG_RDt24r_70_l1016c, par groupe d images (gold 6+8 blocs) ==')
f = 'PG_RDt24r_70_l1016c'; z = ZF[f]; zt = ZF.get('tgt:' + f); kl = spred(z, G)[1]; pg_y = G['p'][ar, y]; Pw = G['p'].copy(); Pw[ar, y] = 0; gw = Pw.argmax(1); fl = z.argmax(1) != y; pred = z.argmax(1)
want = (zt.argmax(1) != y) if zt is not None else fl
grp = {'voulue, atteinte, classe = 1re classe fausse du gold': want & fl & (pred == gw), 'voulue, atteinte, autre classe': want & fl & (pred != gw), 'voulue, non atteinte': want & ~fl, 'non voulue (laissee juste)': ~want & ~fl, 'non voulue, basculee': ~want & fl}
print(f"{'groupe':52s} {'part':>6s} {'p_y gold T2':>11s} {'KL fichier':>10s} {'KL orig':>8s} {'part de la KL totale':>20s}" + ('  KL cible' if zt is not None else ''))
klt = spred(zt, G)[1] if zt is not None else None
for k, m in grp.items():
    if m.sum(): print(f"{k:52s} {m.mean():6.3f} {pg_y[m].mean():11.3f} {kl[m].mean():10.3f} {G['ko_i'][m].mean():8.3f} {kl[m].sum() / kl.sum():20.3f}" + (f'  {klt[m].mean():8.3f}' if zt is not None else ''))
print(f"total : KL fichier {kl.mean():.3f}, KL orig {G['ko']:.3f}" + (f", KL cible {klt.mean():.3f}" if zt is not None else ''))
for lo, hi in [(0, .05), (.05, .3), (.3, .7), (.7, .95), (.95, 1.01)]:
    m = (pg_y >= lo) & (pg_y < hi); print(f"  p_y du gold T2 dans [{lo}, {hi}) : part {m.mean():.3f} ; basculees par le fichier {fl[m].mean():.3f} ; KL fichier {kl[m].mean():.3f} ; KL orig {G['ko_i'][m].mean():.3f} ; part de la KL totale {kl[m].sum() / kl.sum():.3f}")

print('\n== c. sorties synthetiques : construites sur A (8 x 4 blocs + 2 x 6 blocs), S_pred sur B (2 x 6 blocs + 2 x 8 blocs), S_loss sur le gold ajuste ==')
A = M4 + M6[:2]; PA = np.mean([np.exp(lsm(Z * gam(Z), 2)) for Z in A], 0); W = PA.copy(); W[ar, y] = 0; W /= W.sum(1, keepdims=True); rank = np.argsort(PA[ar, y]); u = np.random.default_rng(0).random(n)


def build(tiers, ws=1.0, rest=0.8, width=6.0, rk=rank, Wb=W):
    Ws = Wb ** ws; Ws /= Ws.sum(1, keepdims=True); py = np.full(n, rest); pos = 0
    for part, mode, val in tiers:
        idx = rk[pos:pos + int(round(part * n))]; pos += len(idx)
        if mode == 'py': py[idx] = val
        else: c = val + width * (u[idx] - 0.5); s2 = (Ws[idx] ** 2).sum(1); r = np.sqrt(np.exp(-c) * s2 / (1 - np.exp(-c))); py[idx] = r / (1 + r)
    P = (1 - py)[:, None] * Ws; P[ar, y] = py; return 2 * np.log(np.clip(P, 1e-30, None))


def show(name, z): ce = -lsm(z)[ar, y]; a = sloss(z); b = spred(z, GB)[0]; print(f'{name:66s} m {ce.mean():5.2f} s {ce.std():5.2f} | S_loss {a:.3f} S_pred {b:.3f} produit {a * b:.3f}'); return a * b


for k in ['PG_RDt24r_70_l1016c', 'tgt:PG_RDt24r_70_l1016c', 'PG_RDt24_70_l1016c', 'PG_T85_f70_l814c']:
    if k in ZF: show(('fichier reel ' if not k.startswith('tgt') else 'cible reelle ') + k.replace('tgt:', ''), ZF[k])
print(f'masse fausse W (moyenne A) : probabilite max moyenne {W.max(1).mean():.3f}')
for sh, c in [(0.5, 9.5), (0.55, 9.0), (0.6, 8.5), (0.65, 8.0), (0.7, 7.3), (0.75, 6.8), (0.8, 6.4)]: show(f'bascules dures par rang A : {int(sh * 100)} % a loss {c} +- 3', build([(sh, 'ce', c)]))
for ws in (0.5, 0.75, 1.5): show(f'  65 % a 8.0, masse fausse a la puissance {ws}', build([(0.65, 'ce', 8.0)], ws))
for rest in (0.6, 0.9, 0.97): show(f'  65 % a 8.0, images non basculees a p_y T2 = {rest}', build([(0.65, 'ce', 8.0)], rest=rest))
for wd in (0.0, 12.0): show(f'  65 % a 8.0, largeur de la plage de loss {wd}', build([(0.65, 'ce', 8.0)], width=wd))
best = (0, None)
for hi in (0.3, 0.4, 0.5, 0.55):
    for mid in (0.1, 0.2, 0.3):
        for c in (9.0, 10.0, 11.0, 12.0):
            for q in (0.1, 0.2, 0.35, 0.5):
                zz = build([(hi, 'ce', c), (mid, 'py', q)]); v = sloss(zz) * spred(zz, GB)[0]
                if v > best[0]: best = (v, (hi, mid, c, q))
hi, mid, c, q = best[1]; show(f'TROIS PALIERS (meilleur de 192) : {int(hi * 100)} % a loss {c}, {int(mid * 100)} % a p_y T2 {q}', build([(hi, 'ce', c), (mid, 'py', q)]))
P1 = np.exp(lsm(M4[0] * gam(M4[0]), 2)); W1 = P1.copy(); W1[ar, y] = 0; W1 /= W1.sum(1, keepdims=True); show('  65 % a 8.0, rang A mais masse fausse d UN modele a 4 blocs', build([(0.65, 'ce', 8.0)], Wb=W1))
show('  65 % a 8.0, rang et masse fausse d UN modele a 4 blocs', build([(0.65, 'ce', 8.0)], rk=np.argsort(P1[ar, y]), Wb=W1))
P48 = np.mean([np.exp(lsm(Z * gam(Z), 2)) for Z in M4], 0); W48 = P48.copy(); W48[ar, y] = 0; W48 /= W48.sum(1, keepdims=True); show('  65 % a 8.0, rang et masse fausse de 8 modeles a 4 blocs', build([(0.65, 'ce', 8.0)], rk=np.argsort(P48[ar, y]), Wb=W48))
P62 = np.mean([np.exp(lsm(Z * gam(Z), 2)) for Z in M6[:2]], 0); W62 = P62.copy(); W62[ar, y] = 0; W62 /= W62.sum(1, keepdims=True); show('  65 % a 8.0, rang et masse fausse de 2 modeles a 6 blocs', build([(0.65, 'ce', 8.0)], rk=np.argsort(P62[ar, y]), Wb=W62))

print('\n== d. bascules dures contre trois paliers (bande intermediaire couverte), selon la qualite de la source du rang et de la masse fausse ; notes sur B ==')
def src(models): Pm = np.mean([np.exp(lsm(Z * gam(Z), 2)) for Z in models], 0); Wm = Pm.copy(); Wm[ar, y] = 0; Wm /= Wm.sum(1, keepdims=True); return np.argsort(Pm[ar, y]), Wm
errB = np.mean([(Z.argmax(1) != y) for Z in M6[2:] + M8], 0)
for nm, ms in [('UN modele a 4 blocs', M4[:1]), ('3 modeles a 4 blocs', M4[:3]), ('8 modeles a 4 blocs', M4), ('8 x 4 blocs + 2 x 6 blocs', M4 + M6[:2])]:
    rk, Wm = src(ms); bh = (0, None); bt = (0, None)
    prec = [errB[rk[int(a * n):int(b * n)]].mean() for a, b in [(0, .5), (.5, .6), (.6, .7), (.7, .8), (.8, 1)]]
    for sh in (.5, .55, .6, .65, .7, .75, .8):
        for c in (6.5, 7.5, 8.5, 9.5, 10.5):
            zz = build([(sh, 'ce', c)], rk=rk, Wb=Wm); a, b = sloss(zz), spred(zz, GB)[0]
            if a * b > bh[0]: bh = (a * b, (sh, c, a, b))
    for hi in (.35, .4, .45, .5, .55, .6):
        for mid in (.1, .2, .3):
            for c in (8.0, 9.0, 10.0, 11.0, 12.0):
                for q in (.2, .35, .5, .65):
                    zz = build([(hi, 'ce', c), (mid, 'py', q)], rk=rk, Wb=Wm); a, b = sloss(zz), spred(zz, GB)[0]
                    if a * b > bt[0]: bt = (a * b, (hi, mid, c, q, a, b))
    print(f"source {nm:26s} : part d erreurs du gold B par tranche de rang 0-50/50-60/60-70/70-80/80-100 % : " + ' / '.join(f'{p:.2f}' for p in prec))
    print(f"   bascules dures, meilleur : {int(bh[1][0] * 100)} % a loss {bh[1][1]} -> S_loss {bh[1][2]:.3f} S_pred {bh[1][3]:.3f} produit {bh[0]:.3f}")
    print(f"   trois paliers, meilleur : {int(bt[1][0] * 100)} % a loss {bt[1][2]}, puis {int(bt[1][1] * 100)} % a p_y T2 {bt[1][3]} -> S_loss {bt[1][4]:.3f} S_pred {bt[1][5]:.3f} produit {bt[0]:.3f}  (gain {100 * (bt[0] / bh[0] - 1):+.1f} %)")
