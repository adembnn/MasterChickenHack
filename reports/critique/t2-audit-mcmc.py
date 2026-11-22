"""Audit T2 : echantillonnage (recuit puis MCMC) des couples (verite, masque public) qui redonnent les 11 scores.
Independant des scripts de t2-critique : verites avec DOUBLONS (une variante servie par plusieurs guichets) et
sans aucun a priori tire des lecteurs (chaque guichet : base, variante propre, ou meme modele qu'un autre guichet).
Masques : 'subset' = paires entre guichets d'un ensemble S ; 'pairs' = sous-ensemble quelconque de paires (q libre).
python t2-audit-mcmc.py subset|pairs [chaines] [pas] [graine]"""
import sys, itertools, numpy as np, importlib.util
spec = importlib.util.spec_from_file_location("a", __file__.replace("t2-audit-mcmc.py", "t2-audit-score.py"))
a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
FAM, E, ix, famof, A, B, P, OBS, SUBS, mcc = a.FAM, a.E, a.ix, a.famof, a.A, a.B, a.P, a.OBS, a.SUBS, a.mcc
mode = sys.argv[1] if len(sys.argv) > 1 else "subset"
CONS = [c for c in sys.argv[6:]] if len(sys.argv) > 6 else []   # contraintes : nodup nodisgvar H4 max5 nobase
DISG = [ix[e] for e in ("33", "30", "31", "02", "08", "29", "23", "20", "16", "14", "07", "27")]   # T 1,0 / prompt / persona / tronque
C = int(sys.argv[2]) if len(sys.argv) > 2 else 3000; STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 6000
rng = np.random.default_rng(int(sys.argv[4]) if len(sys.argv) > 4 else 1)
famidx = [np.array([ix[e] for e in f]) for f in FAM]

def energy(L, S):
    T = (L[:, A] == L[:, B]).astype(np.float32)
    M = (S[:, A] & S[:, B]).astype(np.float32) if mode == "subset" else S.astype(np.float32)
    Mt = M * T; Mn = M * (1 - T)
    tp = Mt @ P.T; fp = Mn @ P.T; fn = Mt.sum(1, keepdims=True) - tp; tn = Mn.sum(1, keepdims=True) - fp
    sc = 0.1 + 0.9 * mcc(tp, fp, fn, tn)
    dev = np.maximum(0, np.abs(sc - OBS[None, :]) - 0.025); dev[:, 0] = np.maximum(0, sc[:, 0] - 0.125)
    pen = penalty(L)
    return (dev / 0.01) ** 2 @ np.ones(len(OBS)) + 30 * pen, np.maximum(dev.max(1), pen), sc

def penalty(L):
    var = L != famof[None, :]; pen = np.zeros(len(L))
    if "nodup" in CONS or "H4" in CONS:
        for f in famidx:
            v = np.where(var[:, f], L[:, f], -np.arange(11)[None, :] - 1000)
            srt = np.sort(v, 1); pen += (srt[:, 1:] == srt[:, :-1]).sum(1)
    if "nodisgvar" in CONS or "H4" in CONS: pen += var[:, DISG].sum(1)
    if "H4" in CONS:
        for f in famidx: pen += np.abs(var[:, f].sum(1) - 4)
    if "max5" in CONS:
        for f in famidx: pen += np.maximum(0, var[:, f].sum(1) - 5)
    if "nobase" in CONS:      # une famille sans aucun groupe de copies (au plus 1 guichet a l'etiquette de base)
        pen += np.maximum(0, (~var[:, famidx[0]]).sum(1) - 1)
    return pen

# etat : L (etiquette vraie : famille = base, 100+i = variante propre de i), S (bool 33 ou 165)
L = np.where(rng.random((C, 33)) < 0.5, 100 + np.arange(33)[None, :], famof[None, :]).astype(np.int32)
S = rng.random((C, 33 if mode == "subset" else 165)) < 0.5
en, _, _ = energy(L, S)
for step in range(STEPS):
    Tt = max(1.0, 10.0 * (1 - step / (0.6 * STEPS)))
    L2 = L.copy(); S2 = S.copy(); r = rng.random(C); n = np.arange(C)
    j = rng.integers(0, 33, C)
    m1 = r < 0.35                                                    # basculer la presence au public
    if mode == "subset": S2[n[m1], j[m1]] = ~S2[n[m1], j[m1]]
    else:
        jj = rng.integers(0, 165, C); S2[n[m1], jj[m1]] = ~S2[n[m1], jj[m1]]
    m2 = (r >= 0.35) & (r < 0.65); L2[n[m2], j[m2]] = np.where(L2[n[m2], j[m2]] == famof[j[m2]], 100 + j[m2], famof[j[m2]])
    m3 = r >= 0.65                                                   # rejoindre le modele d'un autre guichet de la famille
    other = np.array([famidx[famof[x]][rng.integers(11)] for x in j])
    L2[n[m3], j[m3]] = L[n[m3], other[m3]]
    e2, _, _ = energy(L2, S2)
    acc = rng.random(C) < np.exp(np.minimum(0, (en - e2) / Tt))
    L[acc] = L2[acc]; S[acc] = S2[acc]; en[acc] = e2[acc]
_, dv, sc = energy(L, S); ok = dv < 1e-9
L, S = L[ok], S[ok]
print(f"[{mode}] {ok.sum()} chaines sur {C} redonnent les 11 scores")
if ok.sum() == 0: sys.exit()
var = L != famof[None, :]
print("P(variante)            : " + " ".join(f"{e}={var[:, ix[e]].mean():.2f}" for e in E))
if mode == "subset":
    print("P(public)              : " + " ".join(f"{e}={S[:, ix[e]].mean():.2f}" for e in E))
    print("P(public ET variante)  : " + " ".join(f"{e}={(S[:, ix[e]] & var[:, ix[e]]).mean():.2f}" for e in E))
    print("guichets publics : moyenne %.1f (min %d, max %d)" % (S.sum(1).mean(), S.sum(1).min(), S.sum(1).max()))
    M = S[:, A] & S[:, B]
else:
    M = S
print("paires internes publiques : moyenne %.1f" % M.sum(1).mean())
dup = np.array([len(set(l[var[k]])) < var[k].sum() for k, l in enumerate(L)]); print(f"P(au moins un doublon de variante) = {dup.mean():.2f}")
print(f"P(17 et 18 meme modele) = {(L[:, ix['17']] == L[:, ix['18']]).mean():.2f}")
# erreurs publiques restantes de 1967 : guichets publics mal classes
lab = SUBS[2][1]
pred = (lab[A] == lab[B]); truth = (L[:, A] == L[:, B])
err = (pred[None, :] != truth) & M
cnt = np.zeros((len(L), 33))
for k in range(len(L)):
    for p in np.nonzero(err[k])[0]: cnt[k, A[p]] += 1; cnt[k, B[p]] += 1
print("paires publiques fausses dans 1967 : moyenne %.1f" % err.sum(1).mean())
print("guichets impliques dans une erreur publique de 1967 (P) : " + " ".join(f"{e}={(cnt[:, ix[e]] > 0).mean():.2f}" for e in E if (cnt[:, ix[e]] > 0).mean() > 0.15))
full = np.array([a.score_all(l)[2] for l in L])
print("score de 1967 sur toutes les paires : moyenne %.3f, p10 %.3f, p90 %.3f" % (full.mean(), np.percentile(full, 10), np.percentile(full, 90)))
nv = var.sum(1); print("guichets variantes : moyenne %.1f (p10 %d, p90 %d)" % (nv.mean(), np.percentile(nv, 10), np.percentile(nv, 90)))
np.savez(sys.argv[5] if len(sys.argv) > 5 else f"t2-audit-mcmc-{mode}.npz", L=L, S=S)
