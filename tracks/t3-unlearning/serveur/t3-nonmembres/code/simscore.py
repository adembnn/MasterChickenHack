"""Test de la chaîne complète dans le simulateur (vrais golds connus), sur 900 images du forget set simulé.
1. Calibration : orig_sim sur variation(forget) contre gold_sim sur forget ; orig_sim sur vrais non-membres ; orig_sim sur variation(non-membres).
2. Signal par image : la loss d'orig_sim sur les variations d'une image prédit-elle la loss du gold sur cette image ?
3. Score officiel (oubli = loss × prédiction) d'un élève qui reproduirait exactement la cible tirée des variations.
usage : simscore.py DOSSIER_SIM_LOCAL   (contient forget/root/meas, nonmembers/root/meas, forget/forget_images.npz, orig_s.npz, gold_s*.npz, world_A.npz, y.npy)"""
import sys, os, glob, json, numpy as np
sys.path.insert(0, 'E:/EAC-lab/t3/t3-simulateur'); import score as SC
S = sys.argv[1] if len(sys.argv) > 1 else '.'
W = np.load(S + '/world_A.npz'); Y = np.load(S + '/y.npy'); fi = np.load(S + '/forget/forget_images.npz'); idx = fi['idx']; y = fi['y']
pos = np.searchsorted(W['forget'], idx); assert (W['forget'][pos] == idx).all() and (Y[idx] == y).all()
O = dict(np.load(S + '/orig_s.npz')); G = [dict(np.load(S + f'/gold_s{k}.npz')) for k in (2, 3, 4, 5)]
sub = lambda d: dict(fx=d['fx'][pos], fa4=d['fa4'][:, pos], ctrl=d['ctrl'], val=d['val'])
Os = sub(O); Gs = [sub(g) for g in G]; lab = dict(fx=y, ctrl=Y[W['ctrl']], val=Y[W['val']])
sm = lambda z, T=1.0: np.exp(SC.lsm(z, T)); ce = SC.ce
def st(z, yy): c = ce(z, yy); return f"acc {np.mean(z.argmax(1) == yy):.3f}  CE {c.mean():.2f}/{c.std():.2f}  >1:{(c > 1).mean():.2f} >5:{(c > 5).mean():.2f}  maxpT2 {sm(z, 2).max(1).mean():.3f}"
ALT = ['alt0_recadrage+flip', 'alt1_couleur_douce', 'alt2_flou_doux', 'alt3_bruit+recadrage']
def load(root):
    d = {}
    for p in sorted(glob.glob(f'{S}/{root}/root/meas/logits_*.npz')):
        t = os.path.basename(p)[7:-4]; z = np.load(p); o = np.argsort(z['src']); d[t] = {k: (z[k][o] if z[k].shape[0] == len(o) else z[k]) for k in z.files}
    return d
F = load('forget'); N = load('nonmembers')
# les variations ne couvrent qu'une partie des 900 images (8 par classe) : tout est restreint à ces images
vts = [t for t in F if t != 'forget']; sel = F[vts[0]]['src']; vts = [t for t in vts if len(F[t]['src']) == len(sel) and (F[t]['src'] == sel).all()]
F = {t: F[t] for t in vts + ['forget']}; F['forget'] = {k: (v[sel] if v.shape[0] == 900 else v) for k, v in F['forget'].items()}
y = y[sel]; pos = pos[sel]; Os = sub(O); Gs = [sub(g) for g in G]; lab = dict(fx=y, ctrl=Y[W['ctrl']], val=Y[W['val']]); print('images utilisées :', len(sel))
print('=== 1. Calibration (900 images, 45 classes) ===')
print(f"{'GOLD sur le forget set (moyenne de 4 golds) : LA CIBLE':58s} acc {np.mean([np.mean(g['fx'].argmax(1) == y) for g in Gs]):.3f}  CE {np.mean([ce(g['fx'], y).mean() for g in Gs]):.2f}/{np.mean([ce(g['fx'], y).std() for g in Gs]):.2f}  >1:{np.mean([(ce(g['fx'], y) > 1).mean() for g in Gs]):.2f} >5:{np.mean([(ce(g['fx'], y) > 5).mean() for g in Gs]):.2f}  maxpT2 {np.mean([sm(g['fx'], 2) for g in Gs], 0).max(1).mean():.3f} (moyenne des golds)")
print(f"{'orig_sim sur le forget set (membres)':58s} {st(Os['fx'], y)}")
if 'forget' in N: print(f"{'orig_sim sur de vrais non-membres':58s} {st(N['forget']['propre'], N['forget']['y'])}")
for t in F:
    if t != 'forget': print(f"{'orig_sim sur ' + t + ' du forget set':58s} {st(F[t]['propre'], y)}")
for t in N:
    if t != 'forget': print(f"{'orig_sim sur ' + t + ' des NON-membres':58s} {st(N[t]['propre'], N[t]['y'])}")
print('\n=== 1 bis. Copies modifiées (4 familles du simulateur) ===')
print(f"{'GOLD sur les copies du forget set':58s} acc {np.mean([np.mean(g['fa4'].argmax(-1) == y) for g in Gs]):.3f}  CE {np.mean([np.mean([ce(g['fa4'][t], y).mean() for t in range(4)]) for g in Gs]):.2f}/{np.mean([np.mean([ce(g['fa4'][t], y).std() for t in range(4)]) for g in Gs]):.2f}")
for t in F:
    if t != 'forget': za = np.concatenate([F[t][a] for a in ALT]); print(f"{'orig_sim sur copies de ' + t:58s} {st(za, np.tile(y, 4))}")
print('\n=== 2. Signal par image : corrélation de rang (Spearman) avec la loss moyenne du gold sur l image ===')
def rank(a): r = np.empty(len(a)); r[np.argsort(a, kind='stable')] = np.arange(len(a)); return r
def sp(a, b): return float(np.corrcoef(rank(a), rank(b))[0, 1])
def auc(score, pos_mask):
    r = rank(score); n1 = pos_mask.sum(); n0 = len(r) - n1; return float((r[pos_mask].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)) if n1 and n0 else float('nan')
Lg = np.mean([ce(g['fx'], y) for g in Gs], 0); wrong = np.mean([g['fx'].argmax(1) != y for g in Gs], 0) >= 0.5
print(f"images que la majorité des golds rate : {wrong.mean():.3f} ; Spearman entre deux golds : {sp(ce(Gs[0]['fx'], y), ce(Gs[1]['fx'], y)):.3f}")
sig = {}
vt = [t for t in F if t != 'forget']
for t in vt: sig[t] = ce(F[t]['propre'], y)
fam = lambda pre: [t for t in vt if t.startswith(pre)]
for name, ts in (('moyenne i2i_70 (toutes graines)', fam('sdxl_i2i_70')), ('moyenne IP-Adapter', fam('sdxl_ipa')), ('moyenne i2i_85', fam('sdxl_i2i_85')), ('moyenne de tous les types', vt)):
    if len(ts) > 1: sig[name + f' [{len(ts)}]'] = np.mean([ce(F[t]['propre'], y) for t in ts], 0)
if 'forget' in F:
    Ff = F['forget']; aug = [k for k in Ff if k not in ('y', 'src', 'propre')]
    for k in aug: sig['SANS générateur : orig_sim sur ' + k] = ce(Ff[k], y)
    sig['SANS générateur : moyenne des augmentations fortes'] = np.mean([ce(Ff[k], y) for k in aug if any(s in k for s in ('couleur', 'flou_1.5', 'recadrage_0.5', 'bruit_0.10', 'jpeg_q20'))], 0)
for n, s in sig.items(): print(f"  {n:58s} Spearman {sp(s, Lg):+.3f}   AUC(gold se trompe) {auc(s, wrong):.3f}")
print('\n=== 3. Score officiel d un élève égal à la cible (rétention laissée à 1) ; référence = golds s3-s5, restreint aux 900 images ===')
ref = Gs[1:]; z_of = lambda p: 2 * np.log(np.clip(p, 1e-9, None))
def sc(fx, fa4): d = dict(Os); d['fx'] = fx; d['fa4'] = fa4; s = SC.score(d, Os, ref, lab); e = s['exact']; return f"oubli {s['forget']:.3f} | loss {s['loss']:.3f} préd {s['pred']:.3f} | L {e['L_mean']:.2f}/{e['L_std']:.2f} (gold {e['gold_mean']:.2f}/{e['gold_std']:.2f})"
def pv(ts, key='propre'): return np.mean([sm(F[t][key], 2) for t in ts], 0)
def pa(ts): return np.stack([pv(ts, a) for a in ALT])
print(f"  {'un autre gold (s2) : plafond d un modèle seul':62s} {sc(Gs[0]['fx'], Gs[0]['fa4'])}")
pe = np.mean([sm(g['fx'], 2) for g in ref], 0); pea = np.stack([np.mean([sm(g['fa4'][t], 2) for g in ref], 0) for t in range(4)])
print(f"  {'moyenne exacte des golds de référence : plafond':62s} {sc(z_of(pe), z_of(pea))}")
po = sm(Os['fx'], 2); poa = np.stack([sm(Os['fa4'][t], 2) for t in range(4)])
for t in vt[:3]: print(f"  {'sortie brute d orig_sim sur ' + t:62s} {sc(F[t]['propre'], np.stack([F[t][a] for a in ALT]))}")
sets = [('moyenne i2i_70 (toutes graines)', fam('sdxl_i2i_70')), ('moyenne IP-Adapter', fam('sdxl_ipa')), ('moyenne de tous les types', vt)]
for name, ts in sets:
    if not ts: continue
    p, pA = pv(ts), pa(ts); print(f"  {name + f' [{len(ts)}]':62s} {sc(z_of(p), z_of(pA))}")
    for w in (0.25, 0.5, 0.75): print(f"  {'   mélange ' + str(w) + ' cible + ' + str(1 - w) + ' orig':62s} {sc(z_of(w * p + (1 - w) * po), z_of(w * pA + (1 - w) * poa))}")
if 'forget' in F:
    Ff = F['forget']; strong = [k for k in Ff if any(s in k for s in ('couleur', 'flou_1.5', 'recadrage_0.5', 'bruit_0.10', 'jpeg_q20'))]
    p = np.mean([sm(Ff[k], 2) for k in strong], 0); print(f"  {'SANS générateur : moyenne d orig_sim sur augmentations fortes':62s} {sc(z_of(p), z_of(np.stack([p] * 4)))}")
    for w in (0.5,): print(f"  {'   mélange 0.5 + 0.5 orig':62s} {sc(z_of(w * p + (1 - w) * po), z_of(w * p[None] + (1 - w) * poa))}")
