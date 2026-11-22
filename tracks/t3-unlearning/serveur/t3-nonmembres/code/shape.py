"""Simulateur (formule officielle, 3 500 images, rétention laissée à 1) : quelle FORME de sortie sur les images basculées et sur les autres maximise loss × prédiction,
en n'utilisant que ce qu'orig fournit (aucune image externe, aucune étiquette). Élèves construits directement sur les logits."""
import sys, itertools, numpy as np
sys.path.insert(0, 'E:/EAC-lab/t3/t3-simulateur'); import score as SC
W = np.load('world_A.npz'); Y = np.load('y.npy'); lab = dict(fx=Y[W['forget']], ctrl=Y[W['ctrl']], val=Y[W['val']]); y = lab['fx']; N = len(y); I = np.arange(N)
O = dict(np.load('orig_s.npz')); G = [dict(np.load(f'gold_s{k}.npz')) for k in (2, 3, 4, 5)]; ref = G[1:]
sm = lambda z, T=1.0: np.exp(SC.lsm(z, T)); yo = O['fx'].argmax(1)
p2 = sm(O['fx'], 2); ent = -(p2 * np.log(np.clip(p2, 1e-12, None))).sum(1)
gw = np.mean([g['fx'].argmax(1) != y for g in G], 0) >= 0.5
def student(z0, sel, kind, L, tau_u, par):
    z = z0.astype(np.float64).copy(); z = z - z.max(1, keepdims=True)       # logit max = 0
    zu = z / tau_u                                                          # images non basculées : adoucies par tau_u
    i = np.where(sel)[0]; zf = z[i].copy(); t = yo[i]; zf[np.arange(len(i)), t] = -1e9; o = np.argsort(-zf, 1)
    if kind == 'dur':            # tout sur le 2e choix d'orig
        out = np.full_like(zf, -30.0); out[np.arange(len(i)), o[:, 0]] = 0
    elif kind == 'etale':        # masse égale sur les par choix suivants d'orig
        out = np.full_like(zf, -30.0)
        for k in range(par): out[np.arange(len(i)), o[:, k]] = 0
    elif kind == 'rang':         # ordre d'orig sur les autres classes, adouci par par
        out = zf / par; out = out - out.max(1, keepdims=True)
    out[np.arange(len(i)), t] = -L; zu[i] = out; return zu
def run(sel, kind, L, tau_u, par):
    d = dict(O); d['fx'] = student(O['fx'], sel, kind, L, tau_u, par); d['fa4'] = np.stack([student(O['fa4'][t], sel, kind, L, tau_u, par) for t in range(4)])
    r = SC.score(d, O, ref, lab); e = r['exact']; return r['forget'], r['loss'], r['pred'], e['L_mean'], e['L_std']
def top(q, s): m = np.zeros(N, bool); m[np.argsort(-s)[:int(N * q)]] = True; return m
sels = {'entropie d orig, top 30 %': top(0.30, ent), 'entropie d orig, top 40 %': top(0.40, ent), 'oracle (images ratées par le gold)': gw}
kinds = [('dur', 0), ('etale', 3), ('etale', 6), ('rang', 1.0), ('rang', 2.0), ('rang', 4.0), ('rang', 8.0)]
for sn, sel in sels.items():
    res = []
    for (kind, par), L, tu in itertools.product(kinds, (3, 5, 7, 9), (1.0, 1.5, 2.0, 3.0)):
        f, lo, pr, m, s = run(sel, kind, L, tu, par); res.append((f, lo, pr, m, s, kind, par, L, tu))
    res.sort(reverse=True); print(f'=== sélection : {sn} ({sel.mean():.2f} des images ; précision de la sélection {gw[sel].mean():.2f}) ; gold m/s 1.36/2.52 ===')
    for f, lo, pr, m, s, kind, par, L, tu in res[:6]: print(f"  oubli {f:.3f} | loss {lo:.3f} préd {pr:.3f} | m/s {m:.2f}/{s:.2f} | basculées : {kind} {par} loss {L} ; autres : orig adouci tau {tu}")
    for kind, par in kinds:
        b = max(r for r in res if r[5] == kind and r[6] == par); print(f"    meilleur pour '{kind} {par}' : oubli {b[0]:.3f} (loss {b[1]:.2f}, préd {b[2]:.2f}) à L={b[7]}, tau={b[8]}")
