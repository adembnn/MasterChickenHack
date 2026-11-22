"""t3-verificateur : (A) ou part la KL(gold || fichier) a T=2 sur les images a oublier ; (B) essai de forme sans entrainement : tete multipliee par alpha (logits x alpha, exact), plancher de probabilite.
Gold de substitution : 6 redistillations hors pli a 6 et 8 blocs, logits cales a une loss moyenne de 5,1. Score de loss sur le gold ajuste aux envois (m 5,1 ; s 4,7), groupes exactes et copies legeres.
usage : python pgab.py [NOM]   (defaut SOUPs1)"""
import os, sys, numpy as np
ROOT = os.environ['T3ROOT']; ME = ROOT + '/t3-verificateur'; OUT = ME + '/out'; OOF = ME + '/oof'; NM = sys.argv[1] if len(sys.argv) > 1 else 'SOUPs1'; MG, SG = 5.1, 4.7
AF = ['a_shift', 'a_crop', 'a_rot', 'a_color', 'a_blur', 'a_noise', 'a_jpeg']; CF = ['c_mix', 'c_blur_strong', 'c_color_strong', 'c_crop_strong']


def lsm(z, t=1.0):
    z = np.asarray(z, np.float64) / t; z = z - z.max(-1, keepdims=True); return z - np.log(np.exp(z).sum(-1, keepdims=True))


def load(nm):
    d = {k: v.astype(np.float64) for k, v in np.load(f'{OUT}/{nm}.npz').items()}; i = np.arange(len(d['exact'])); d['alt'] = np.stack([d[a] for a in AF])[i % 7, i]; return d


O = load('orig'); S = load(NM); y = O['exact'].argmax(1); n = len(y); ar = np.arange(n); LO2 = lsm(O['exact'], 2); CEO = {g: -lsm(O[g])[ar, y] for g in ('exact', 'alt')}
M = [np.load(f'{OOF}/redistill_k6m4_logits_s{i}.npy').astype(np.float64) for i in range(4)] + [np.load(f'{OOF}/redistill_k8m2_logits_s{i}.npy').astype(np.float64) for i in range(2)]


def gam(Z, mc=5.1):
    lo, hi = 0.2, 30.0
    for _ in range(50):
        g = (lo + hi) / 2
        if (-lsm(Z * g)[ar, y]).mean() < mc: lo = g
        else: hi = g
    return g


P = np.mean([np.exp(lsm(Z * gam(Z), 2)) for Z in M], 0); LP = np.log(np.clip(P, 1e-12, None)); KOi = (P * (LP - LO2)).sum(1); KO = KOi.mean(); gtop = P.argmax(1); gwrong = gtop != y
def kli(z): return (P * (LP - lsm(z, 2))).sum(1)
def sl(z, g): ce = -lsm(z)[ar, y]; return 1 - min(1, (abs(ce.mean() - MG) + abs(ce.std() - SG)) / (abs(CEO[g].mean() - MG) + abs(CEO[g].std() - SG)))


z = S['exact']; kl = kli(z); pred = z.argmax(1); fl = pred != y; L2 = lsm(z, 2)
print(f'== A. {NM} : KL(gold||fichier) moyenne {kl.mean():.3f} ; KL(gold||orig) {KO:.3f} ; score de prediction local {1 - kl.mean() / KO:.3f} ; le gold de substitution a faux sur {gwrong.mean():.3f} des images ==')
o = np.argsort(-kl); cs = np.cumsum(kl[o]) / kl.sum(); print('A1. part de la KL totale par decile d images, des plus cheres aux moins cheres : ' + ' / '.join(f'{(cs[int((k + 1) * n / 10) - 1] - (cs[int(k * n / 10) - 1] if k else 0)):.3f}' for k in range(10)))
print(f'    KL moyenne du decile le plus cher {kl[o[:n // 10]].mean():.2f} ; mediane de toutes les images {np.median(kl):.2f}')
cells = {'1. basculee, gold faux, MEME classe': fl & gwrong & (pred == gtop), '2. basculee, gold faux, AUTRE classe': fl & gwrong & (pred != gtop), '3. basculee, gold JUSTE': fl & ~gwrong,
         '4. non basculee, gold FAUX': ~fl & gwrong, '5. non basculee, gold juste': ~fl & ~gwrong}
print('A2. cases'); print(f"{'case':38s} {'effectif':>8s} {'part':>6s} {'KL moy':>7s} {'part de la KL':>13s} {'KL orig':>8s} | p max T2 : gold  fichier | p_y T2 : gold  fichier | masse du gold sur la classe du fichier, et du fichier dessus")
for k, m in cells.items():
    if not m.sum(): continue
    pg = np.exp(LP[m]); ps = np.exp(L2[m]); c = pred[m]; i = np.arange(m.sum())
    print(f"{k:38s} {m.sum():8d} {m.mean():6.3f} {kl[m].mean():7.3f} {kl[m].sum() / kl.sum():13.3f} {KOi[m].mean():8.3f} | {pg.max(1).mean():15.3f} {ps.max(1).mean():8.3f} | {pg[i, y[m]].mean():13.3f} {ps[i, y[m]].mean():8.3f} | {pg[i, c].mean():6.3f} {ps[i, c].mean():6.3f}")
print('A3. forme ou information ? dans chaque case, KL si on remplacait la sortie du fichier par : (a) la meme sortie a la meilleure temperature de la case ; (b) la sortie du gold lui-meme avec la classe du fichier gardee en tete (borne : forme parfaite, information du fichier)')
tot_a = 0; tot_b = 0
for k, m in cells.items():
    if not m.sum(): continue
    best = min(((P[m] * (LP[m] - lsm(z[m] * a, 2))).sum(1).mean(), a) for a in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5))
    # (b) : distribution du gold, mais on echange la masse de la classe en tete du gold avec celle de la classe predite par le fichier (si elles different)
    Q = P[m].copy(); i = np.arange(m.sum()); c = pred[m]; g = gtop[m]; a_, b_ = Q[i, g].copy(), Q[i, c].copy(); Q[i, g] = b_; Q[i, c] = a_; klb = (P[m] * (LP[m] - np.log(np.clip(Q, 1e-12, None)))).sum(1).mean()
    tot_a += best[0] * m.mean(); tot_b += klb * m.mean(); print(f'   {k:38s} KL actuelle {kl[m].mean():.3f} ; (a) {best[0]:.3f} a alpha {best[1]} ; (b) {klb:.3f}')
print(f'   total : KL actuelle {kl.mean():.3f} (S_pred {1 - kl.mean() / KO:.3f}) ; (a) meilleure temperature par case {tot_a:.3f} (S_pred {1 - tot_a / KO:.3f}) ; (b) forme parfaite avec l information du fichier {tot_b:.3f} (S_pred {1 - tot_b / KO:.3f})')

print(f'\n== B. tete de {NM} multipliee par alpha (logits x alpha) ==')
print(f"{'alpha':>6s} | exactes m     s   | copies m     s   | S_loss ex  cop  moy | S_pred | produit (S_loss moy x S_pred) | KL(orig||fichier) T2 : melanges flou_fort coul_forte recadr_fort")
base = None
for a in (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2):
    ce = -lsm(S['exact'] * a)[ar, y]; ca = -lsm(S['alt'] * a)[ar, y]; s1, s2 = sl(S['exact'] * a, 'exact'), sl(S['alt'] * a, 'alt'); sp = 1 - kli(S['exact'] * a).mean() / KO; pr = (s1 + s2) / 2 * sp
    kc = [float((np.exp(lsm(O[c], 2)) * (lsm(O[c], 2) - lsm(S[c] * a, 2))).sum(1).mean()) for c in CF]
    if a == 1.0: base = pr
    print(f'{a:6.2f} | {ce.mean():7.2f} {ce.std():5.2f} | {ca.mean():7.2f} {ca.std():5.2f} | {s1:9.3f} {s2:.3f} {(s1 + s2) / 2:.3f} | {sp:6.3f} | {pr:7.3f} | ' + ' '.join(f'{v:8.3f}' for v in kc))
print(f'reference alpha 1 : produit {base:.3f}')
print('plancher de probabilite simule : sortie T=1 melangee avec l uniforme')
for eps in (0.0, 0.005, 0.02, 0.05, 0.10):
    def fl_(zz): p = np.exp(lsm(zz)); return np.log((1 - eps) * p + eps / 100)
    ze, za = fl_(S['exact']), fl_(S['alt']); ce = -lsm(ze)[ar, y]; s1, s2 = sl(ze, 'exact'), sl(za, 'alt'); sp = 1 - kli(ze).mean() / KO
    kc = [float((np.exp(lsm(O[c], 2)) * (lsm(O[c], 2) - lsm(fl_(S[c]), 2))).sum(1).mean()) for c in CF]
    print(f'  plancher {eps:5.3f} : m {ce.mean():5.2f} s {ce.std():5.2f} | S_loss moy {(s1 + s2) / 2:.3f} | S_pred {sp:.3f} | produit {(s1 + s2) / 2 * sp:.3f} | KL melanges {kc[0]:.3f}')
print('plancher sur la distribution T=2 seulement (borne : ce que gagnerait S_pred si la masse nulle etait relevee, loss inchangee)')
for eps in (0.02, 0.05, 0.10):
    q = (1 - eps) * np.exp(L2) + eps / 100; k2 = (P * (LP - np.log(q))).sum(1).mean(); print(f'  plancher T=2 {eps:4.2f} : S_pred {1 - k2 / KO:.3f}')
