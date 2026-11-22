"""Simulateur : les variations générées désignent-elles la classe vers laquelle le gold se trompe, mieux que le 2e choix d'orig ?
Et quel score d'oubli donne « sur les images sélectionnées (signal gratuit), prendre la distribution d'orig sur la variation ; ailleurs, orig » ?"""
import sys, os, glob, numpy as np
sys.path.insert(0, 'E:/EAC-lab/t3/t3-simulateur'); import score as SC
W = np.load('world_A.npz'); Y = np.load('y.npy'); fi = np.load('forget/forget_images.npz'); idx = fi['idx']; y = fi['y']; pos = np.searchsorted(W['forget'], idx)
O = dict(np.load('orig_s.npz')); G = [dict(np.load(f'gold_s{k}.npz')) for k in (2, 3, 4, 5)]
V = {}
for p in sorted(glob.glob('forget/root/meas/logits_sdxl_*.npz')):
    z = np.load(p); o = np.argsort(z['src']); V[os.path.basename(p)[7:-4]] = {k: z[k][o] for k in z.files}
sel = V[list(V)[0]]['src']; V = {t: v for t, v in V.items() if len(v['src']) == len(sel)}; y = y[sel]; pos = pos[sel]; n = len(y); I = np.arange(n)
sub = lambda d: dict(fx=d['fx'][pos], fa4=d['fa4'][:, pos], ctrl=d['ctrl'], val=d['val']); Os = sub(O); Gs = [sub(g) for g in G]; lab = dict(fx=y, ctrl=Y[W['ctrl']], val=Y[W['val']])
sm = lambda z, T=1.0: np.exp(SC.lsm(z, T)); ref = Gs[1:]; ALT = ['alt0_recadrage+flip', 'alt1_couleur_douce', 'alt2_flou_doux', 'alt3_bruit+recadrage']
gp = np.stack([g['fx'].argmax(1) for g in Gs]); gw = (gp != y).mean(0); wrong = gw >= 0.5
gm = np.array([np.bincount(gp[:, i][gp[:, i] != y[i]], minlength=100).argmax() if (gp[:, i] != y[i]).any() else -1 for i in range(n)])   # classe fausse la plus fréquente des golds
o2 = np.argsort(-Os['fx'], 1); pg = np.mean([sm(g['fx'], 2) for g in Gs], 0)
print(f"{n} images ; ratées par la majorité des golds : {wrong.sum()} ({wrong.mean():.3f})")
print('--- Classe d erreur du gold retrouvée (sur les images que la majorité des golds rate)')
print(f"  2e choix d orig sur l image elle-même                    : {np.mean(o2[wrong, 1] == gm[wrong]):.3f}   (choix 2 à 4 : {np.mean([gm[i] in o2[i, 1:4] for i in np.where(wrong)[0]]):.3f})")
for t, v in V.items():
    pv = v['propre'].argmax(1); e = pv != y; zz = v['propre'].copy(); zz[I, y] = -1e9; alt = zz.argmax(1)   # meilleure classe autre que la vraie
    print(f"  {t:14s} prédiction d orig sur la variation, si fausse : {np.mean(pv[wrong & e] == gm[wrong & e]):.3f} (sur {int((wrong & e).sum())} images)   | 1re classe hors vraie classe sur la variation : {np.mean(alt[wrong] == gm[wrong]):.3f}")
    print(f"  {'':14s} masse du gold moyen (T=2) sur la classe proposée : variation {pg[I, alt][wrong].mean():.3f} contre 2e choix d orig {pg[I, o2[:, 1]][wrong].mean():.3f}")
print('--- Erreur d orig sur la variation contre erreur du gold sur l image')
for t, v in V.items():
    e = v['propre'].argmax(1) != y; print(f"  {t:14s} P(gold rate | orig rate la variation) {wrong[e].mean():.3f} ({int(e.sum())} images) ; P(gold rate | orig réussit la variation) {wrong[~e].mean():.3f}")
print('--- Score officiel (360 images) : distribution de la variation sur les images sélectionnées, orig ailleurs')
z_of = lambda p: 2 * np.log(np.clip(p, 1e-9, None)); po = sm(Os['fx'], 2); poa = np.stack([sm(Os['fa4'][t], 2) for t in range(4)])
ent = -(po * np.log(np.clip(po, 1e-12, None))).sum(1)
def sc(p, pA): d = dict(Os); d['fx'] = z_of(p); d['fa4'] = z_of(pA); s = SC.score(d, Os, ref, lab); e = s['exact']; return f"oubli {s['forget']:.3f} | loss {s['loss']:.3f} préd {s['pred']:.3f} | m/s {e['L_mean']:.2f}/{e['L_std']:.2f} (gold {e['gold_mean']:.2f}/{e['gold_std']:.2f})"
for t, v in V.items():
    pv = sm(v['propre'], 2); pvA = np.stack([sm(v[a], 2) for a in ALT]); e = v['propre'].argmax(1) != y
    print(f"  {t} partout                                         {sc(pv, pvA)}")
    m = e; print(f"  {t} seulement là où orig rate la variation ({m.mean():.2f})   {sc(np.where(m[:, None], pv, po), np.where(m[None, :, None], pvA, poa))}")
    for q in (0.2, 0.35, 0.5):
        m = np.zeros(n, bool); m[np.argsort(-ent)[:int(n * q)]] = True; print(f"  {t} sur le top {q:.2f} d entropie d orig              {sc(np.where(m[:, None], pv, po), np.where(m[None, :, None], pvA, poa))}")
    m = wrong; print(f"  {t} sur les images que le gold rate (oracle)        {sc(np.where(m[:, None], pv, po), np.where(m[None, :, None], pvA, poa))}")
