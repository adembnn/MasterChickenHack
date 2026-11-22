"""Suite de sens.py : sous-modèles « oracles » construits à partir des golds du simulateur, pour savoir quelle forme de cible rapporte.
Un élève parfaitement distillé vers une distribution cible p (à température 2) a pour logits z = 2·log p."""
import sys, numpy as np
sys.path.insert(0, 'E:/EAC-lab/t3/t3-simulateur'); import score as SC
W = np.load('world_A.npz'); Y = np.load('y.npy'); lab = dict(fx=Y[W['forget']], ctrl=Y[W['ctrl']], val=Y[W['val']]); y = lab['fx']
O = dict(np.load('orig_s.npz')); G = [dict(np.load(f'gold_s{k}.npz')) for k in (2, 3, 4, 5)]
def mk(fx, fa4): d = dict(O); d['fx'] = fx; d['fa4'] = fa4; return d
def show(n, s): e = s['exact']; print(f"{n:62s} oubli {s['forget']:.3f} | loss {s['loss']:.3f} préd {s['pred']:.3f} | L {e['L_mean']:.2f}/{e['L_std']:.2f} (gold {e['gold_mean']:.2f}/{e['gold_std']:.2f}) acc {e['acc']:.3f}")
def pT2(zs): return np.mean([np.exp(SC.lsm(z, 2.0)) for z in zs], 0)
def z_of(p): return 2 * np.log(np.clip(p, 1e-9, None))
ref = G[1:]   # golds de référence : s3, s4, s5 ; s2 et (s2 seul) servent de sources indépendantes
pe = pT2([g['fx'] for g in ref]); pa = np.stack([pT2([g['fa4'][t] for g in ref]) for t in range(4)])
show('élève = moyenne exacte des golds de référence (par image)', SC.score(mk(z_of(pe), z_of(pa)), O, ref, lab))
p1 = pT2([G[0]['fx']]); pa1 = np.stack([pT2([G[0]['fa4'][t]]) for t in range(4)])
show('élève = un autre gold seul (s2), par image', SC.score(G[0], O, ref, lab))
def bycls(p): q = p.copy(); [q.__setitem__(np.where(y == c)[0], p[y == c].mean(0)) for c in np.unique(y)]; return q
show('élève = moyenne PAR CLASSE des golds de référence (oracle de classe)', SC.score(mk(z_of(bycls(pe)), z_of(np.stack([bycls(pa[t]) for t in range(4)]))), O, ref, lab))
show('élève = moyenne PAR CLASSE du gold s2 seul (indépendant)', SC.score(mk(z_of(bycls(p1)), z_of(np.stack([bycls(pa1[t]) for t in range(4)]))), O, ref, lab))
# cible de classe estimée depuis orig sur de VRAIS non-membres du simulateur (jeu de contrôle) : ce que la piste non-membres vise à fournir
yc = lab['ctrl']; pc = np.exp(SC.lsm(O['ctrl'], 2.0)); q = np.zeros_like(pe)
for c in np.unique(y): q[y == c] = pc[yc == c].mean(0)
show('élève = moyenne par classe d orig sur vrais non-membres (contrôle)', SC.score(mk(z_of(q), z_of(np.stack([q] * 4))), O, ref, lab))
# mélange : part w de la cible de classe + (1-w) d'orig, par image
po = np.exp(SC.lsm(O['fx'], 2.0)); poa = np.stack([np.exp(SC.lsm(O['fa4'][t], 2.0)) for t in range(4)])
for w in (0.3, 0.5, 0.7, 0.9): show(f'  mélange {w} cible de classe (contrôle) + {1 - w:.1f} orig', SC.score(mk(z_of(w * q + (1 - w) * po), z_of(w * q[None] + (1 - w) * poa)), O, ref, lab))
# par image : tirer pour chaque image à oublier UNE image de contrôle de la même classe et prendre la sortie d'orig dessus (imite « orig sur une variation non-membre »)
rng = np.random.default_rng(0)
for K in (1, 2, 4, 8):
    zz = np.zeros_like(pe)
    for c in np.unique(y):
        idx = np.where(y == c)[0]; pool = np.where(yc == c)[0]; zz[idx] = pc[rng.choice(pool, (len(idx), K))].mean(1)
    if K == 1:
        zl = np.zeros_like(O['fx'], dtype=np.float64)
        for c in np.unique(y): idx = np.where(y == c)[0]; pool = np.where(yc == c)[0]; zl[idx] = O['ctrl'][rng.choice(pool, len(idx))]
        show('élève = sortie brute d orig sur 1 non-membre de la classe (logits)', SC.score(mk(zl, np.stack([zl] * 4)), O, ref, lab))
    show(f'élève = moyenne d orig sur {K} non-membres de la classe (par image)', SC.score(mk(z_of(zz), z_of(np.stack([zz] * 4))), O, ref, lab))
