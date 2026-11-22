"""Simulateur (régime entraîné de zéro, 3 500 images à oublier, formule officielle) : quel signal par image, disponible depuis orig seul, désigne les images que le gold rate,
et quel score d'oubli donne la recette « basculer les q % d'images les plus difficiles vers une autre classe, laisser les autres comme orig ».
Élève construit directement sur les logits (pas d'entraînement) : rétention laissée à 1."""
import sys, numpy as np
sys.path.insert(0, 'E:/EAC-lab/t3/t3-simulateur'); import score as SC
W = np.load('world_A.npz'); Y = np.load('y.npy'); lab = dict(fx=Y[W['forget']], ctrl=Y[W['ctrl']], val=Y[W['val']]); y = lab['fx']; N = len(y); I = np.arange(N)
O = dict(np.load('orig_s.npz')); G = [dict(np.load(f'gold_s{k}.npz')) for k in (2, 3, 4, 5)]
ce = SC.ce; sm = lambda z, T=1.0: np.exp(SC.lsm(z, T))
def rank(a): r = np.empty(len(a)); r[np.argsort(a, kind='stable')] = np.arange(len(a)); return r
def sp(a, b): return float(np.corrcoef(rank(a), rank(b))[0, 1])
def auc(s, m): r = rank(s); n1 = m.sum(); n0 = len(r) - n1; return float((r[m].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))
Lg = np.mean([ce(g['fx'], y) for g in G], 0); gw = np.mean([g['fx'].argmax(1) != y for g in G], 0); wrong = gw >= 0.5
yo = O['fx'].argmax(1); zs = np.sort(O['fx'], 1)
print(f"gold : précision moyenne {1 - gw.mean():.3f} ; images ratées par la majorité des golds {wrong.mean():.3f} ; orig juste sur {np.mean(yo == y):.4f} du forget set")
sig = {'loss d orig sur l image (pseudo-étiquette)': ce(O['fx'], yo), 'moins la marge logit top1 - top2 d orig': -(zs[:, -1] - zs[:, -2]),
       'moins le logit max d orig': -zs[:, -1], 'entropie d orig à T=2': -(sm(O['fx'], 2) * SC.lsm(O['fx'], 2)).sum(1),
       'loss moyenne d orig sur 4 copies légères': np.mean([ce(O['fa4'][t], yo) for t in range(4)], 0),
       'loss d orig sur copie couleur douce seule': ce(O['fa4'][1], yo), 'hasard': np.random.default_rng(0).random(N)}
print('--- A. Signal par image (aucune étiquette, aucune image externe) contre la loss moyenne du gold')
for n, s in sig.items():
    o = np.argsort(-s); print(f"  {n:44s} Spearman {sp(s, Lg):+.3f}  AUC {auc(s, wrong):.3f}  | part réellement ratée par le gold dans le top 10/20/30/40 % : " + ' '.join(f"{wrong[o[:int(N * q)]].mean():.2f}" for q in (.1, .2, .3, .4)))
print('--- B. Vers quelle classe ? Sur les images que la majorité des golds rate')
gm = np.array([np.bincount(np.array([g['fx'][i].argmax() for g in G]), minlength=100).argmax() for i in range(N)])   # classe majoritaire des golds
o2 = np.argsort(-O['fx'], 1)
for k in (1, 2, 3): print(f"  la classe la plus fréquente des golds est le choix n°{k + 1} d orig : {np.mean(gm[wrong] == o2[wrong, k]):.3f}")
print(f"  ... est dans les choix 2 à 4 d orig : {np.mean([gm[i] in o2[i, 1:4] for i in np.where(wrong)[0]]):.3f} ; deux golds qui se trompent sur la même image choisissent la même classe : {np.mean((G[0]['fx'].argmax(1) == G[1]['fx'].argmax(1))[(G[0]['fx'].argmax(1) != y) & (G[1]['fx'].argmax(1) != y)]):.3f}")
print('--- C. Score officiel de la recette « basculer le top q % vers une classe, loss visée L sur les basculées, le reste = orig » (référence : golds s3-s5)')
ref = G[1:]
def build(z, sel, tgt, L, yref):
    """Sur les images sel : la classe tgt reçoit le logit max, la classe d'origine est abaissée pour que la loss vaille environ L."""
    z = z.astype(np.float64).copy(); i = np.where(sel)[0]; top = z[i].max(1); z[i, tgt[i]] = top + 1e-3; z[i, yref[i]] = top - L; return z
def run(name, s, tgt, q, L):
    o = np.argsort(-s); sel = np.zeros(N, bool); sel[o[:int(N * q)]] = True
    d = dict(O); d['fx'] = build(O['fx'], sel, tgt, L, yo); d['fa4'] = np.stack([build(O['fa4'][t], sel, tgt, L, yo) for t in range(4)])
    r = SC.score(d, O, ref, lab); e = r['exact']; print(f"  {name:38s} q={q:.2f} L={L:4.1f} : oubli {r['forget']:.3f} | loss {r['loss']:.3f} préd {r['pred']:.3f} | m/s {e['L_mean']:.2f}/{e['L_std']:.2f} (gold {e['gold_mean']:.2f}/{e['gold_std']:.2f})")
    return r['forget']
t2 = o2[:, 1]; best = {}
for sn in ('hasard', 'moins la marge logit top1 - top2 d orig', 'loss moyenne d orig sur 4 copies légères'):
    for q in (0.2, 0.3, 0.4):
        for L in (3, 5, 8, 11): best[(sn, q, L)] = run(sn[:38], sig[sn], t2, q, L)
print('  --- plafonds : sélection parfaite (images que le gold rate) ; classe = 2e choix d orig, puis classe majoritaire des golds')
orc = Lg.copy()
for L in (3, 5, 8): run('oracle images, 2e choix d orig', orc, t2, wrong.mean(), L)
for L in (3, 5, 8): run('oracle images + classe des golds', orc, gm, wrong.mean(), L)
k = max(best, key=best.get); print('meilleure recette sans oracle :', k, round(best[k], 3))
