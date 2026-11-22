"""t2-oracle : hypothese « gabarit » (b) par repondération exacte des tirages a posteriori « etats » (aucune requete).
Gabarit : 5 classes alignees de meme nature dans les 3 familles ; les 6 autres guichets de chaque famille : meme nombre de
variantes dans les 3 familles. b4 : 4 variantes en tout par famille ; ball : nombre libre (0 a 11)."""
import json, math, sys, numpy as np
L = "E:/EAC-lab/t2/reel/oracle/"
FAM = [["01","02","04","12","14","17","18","23","24","25","33"], ["03","06","07","08","20","21","22","26","28","30","32"], ["05","09","10","11","13","15","16","19","27","29","31"]]
E = [e for f in FAM for e in f]; ix = {e: i for i, e in enumerate(E)}
CL = [("33","30","31"), ("02","08","29"), ("23","20","16"), ("14","07","27"), ("25","06","19")]
REST = [[ix[e] for e in f if not any(e in c for c in CL)] for f in FAM]
X = np.load(L + "exact_etats_egale.npy"); Zet = json.load(open(L + "exact_etats_egale.json"))["Z"]
nS = sum(math.comb(33, k) for k in range(14, 23))
def C(n, k): return math.comb(n, k) if 0 <= k <= n else 0
def w_b(x, total):
    # classes : etats publics doivent concorder
    fixed = []; free = 0
    for c in CL:
        st = {int(x[ix[e]]) for e in c} - {0}
        if len(st) > 1: return 0
        if st: fixed.append(1 if st == {2} else 0)
        else: free += 1
    nf = sum(fixed); tot = 0
    for extra in range(free + 1):
        nc = nf + extra; ways = math.comb(free, extra)
        if total is None:
            for v in range(7):
                p = ways
                for f in range(3):
                    r = x[REST[f]]; p *= C(int((r == 0).sum()), v - int((r == 2).sum()))
                tot += p
        else:
            v = total - nc; p = ways
            for f in range(3):
                r = x[REST[f]]; p *= C(int((r == 0).sum()), v - int((r == 2).sum()))
            tot += p
    return tot
Hsize = {4: sum(math.comb(5, c) * C(6, 4 - c) ** 3 for c in range(6)), None: sum(math.comb(5, c) * sum(C(6, v) ** 3 for v in range(7)) for c in range(6))}
for tot, nm in ((4, "b4"), (None, "ball")):
    w = np.array([w_b(x, tot) for x in X], float)
    Z = Zet * w.mean(); print(f"{nm} : part des tirages compatibles {np.mean(w > 0):.3f} ; P(12 scores | {nm}, S) = {Z / (nS * Hsize[tot]):.3e} ; taille de l'hypothese {Hsize[tot]}")
    if w.sum() > 0:
        ww = w / w.sum(); print("   P(public)/P(var|pub) :", " ".join(f"{e}:{(ww * (X[:, i] > 0)).sum():.2f}/{(ww * (X[:, i] == 2)).sum() / max((ww * (X[:, i] > 0)).sum(), 1e-9):.2f}" for i, e in enumerate(E)))
        np.save(L + f"gabarit_{nm}_poids.npy", w)
        # nature des classes
        for c in CL:
            i = ix[c[0]]; pubany = np.array([any(x[ix[e]] > 0 for e in c) for x in X])
            var = np.array([any(x[ix[e]] == 2 for e in c) for x in X])
            print(f"   classe {c} : P(au moins un public) {(ww * pubany).sum():.2f} ; P(variante et visible) {(ww * var).sum():.2f}")
