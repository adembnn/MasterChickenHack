# Session relais : enumeration LIBRE en deux temps (aucune requete API).
# Verite : dans chaque famille, n'importe quel sous-ensemble des guichets possiblement publics est variante (variantes toutes distinctes),
#          le reste = base. Ensemble public : tout sous-ensemble (>= 2) des guichets non exclus. Contraintes : les 6 scores informatifs.
# Temps 1 : signatures sur les seuls scores connus, croisement des trois familles. Temps 2 : pour chaque triplet compatible,
#          effet de chaque envoi candidat (un changement dans UNE famille) en parcourant les etats de cette famille.
# Usage : python t2-relais-libre.py [maxvar]   (maxvar = nombre maximal de variantes publiques par famille, defaut 5)
import itertools, collections, sys
import numpy as np
MAXV = int(sys.argv[1]) if len(sys.argv) > 1 else 5
FAM = {"q3b": [1, 2, 4, 12, 14, 17, 18, 23, 24, 25, 33], "q15": [3, 6, 7, 8, 20, 21, 22, 26, 28, 30, 32], "lla": [5, 9, 10, 11, 13, 15, 16, 19, 27, 29, 31]}
HORS = {1, 3, 9, 11, 19, 28}   # 12 et 24 testes ENSEMBLE seulement : compensation possible, on les laisse libres
FOF = {e: f for f, es in FAM.items() for e in es}
def L67(e):
    if e in (5, 15, 22): return "v%d" % e
    if e in (17, 18): return "x"
    return "b"
def L82(e): return "v%d" % e if e in (5, 15, 22) else "b"
def Lsp(e): return "v%d" % e if e in (17, 18) else L67(e)
HISO = {3, 4, 5, 9, 11, 12, 15, 17, 18, 21, 22, 32}
def LH(e): return "v%d" % e if e in HISO else "b"
def B(e): return "v26" if e == 26 else L67(e)
def B2(e): return "v%d" % e if e in (10, 13) else B(e)
def L1224(e): return "v%d" % e if e in (12, 24) else L67(e)
SUBS = [("1967+12,24", L1224, 0.60), ("1882", L82, 0.40), ("1967", L67, 0.60), ("17|18", Lsp, 0.60), ("H", LH, 0.30), ("b26", B, 0.80), ("b26+10,13", B2, 0.60)]
IB = 5                                                    # indice du fichier retenu (0,80)
def iso(x): return lambda e: ("v%d" % e if e == x else B(e))
def enbase(x): return lambda e: ("b" if e == x else B(e))
def meme(a, b): return lambda e: ("g%d" % a if e in (a, b) else B(e))
CAND = [("isoler %02d" % x, iso(x), FOF[x]) for x in (24, 12, 2, 14, 23, 25, 33, 6, 7, 8, 20, 30)]
CAND += [("isoler 12 et 24", lambda e: ("v%d" % e if e in (12, 24) else B(e)), "q3b")]
CAND += [("17|18 separes", lambda e: ("v%d" % e if e in (17, 18) else B(e)), "q3b"), ("05=15 meme", meme(5, 15), "lla")]
def counts(pub, t, lab):
    tp = fp = tn = fn = 0
    for a, b in itertools.combinations(pub, 2):
        st = t[a] == t[b]; sp = lab(a) == lab(b)
        if st and sp: tp += 1
        elif st: fn += 1
        elif sp: fp += 1
        else: tn += 1
    return (tp, fp, tn, fn)
states = {}
for f, eps in FAM.items():
    cand = [e for e in eps if e not in HORS]
    sig = collections.defaultdict(list)
    for r in range(0, min(MAXV, len(cand) - 1) + 1):
        for var in itertools.combinations(cand, r):
            t = {e: (("v", e) if e in var else "b") for e in cand}
            for k in range(2, len(cand) + 1):
                for pub in itertools.combinations(cand, k):
                    base = tuple(counts(pub, t, lab) for _, lab, _ in SUBS)
                    extra = tuple(counts(pub, t, lab) if ff == f else None for _, lab, ff in CAND)
                    sig[base].append((pub, var, extra))
    states[f] = sig
    print(f, "signatures", len(sig), "etats", sum(len(v) for v in sig.values()), flush=True)
def score(c):
    tp, fp, tn, fn = [c[..., i].astype(float) for i in range(4)]
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    with np.errstate(divide="ignore", invalid="ignore"): m = np.where(den > 0, (tp * tn - fp * fn) / den, 0.0)
    return 0.1 + 0.9 * m
def sc1(c):
    tp, fp, tn, fn = [float(x) for x in c]
    den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return 0.1 + 0.9 * ((tp * tn - fp * fn) / den if den > 0 else 0.0)
keys = {f: list(states[f]) for f in FAM}; arr = {f: np.array(keys[f]) for f in FAM}
w = {f: np.array([len(states[f][k]) for k in keys[f]], float) for f in FAM}
tg = np.array([t for _, _, t in SUBS])
A, Bq, C = arr["q3b"], arr["q15"], arr["lla"]
tot = 0.0; up = np.zeros(len(CAND)); dn = np.zeros(len(CAND)); gain = np.zeros(len(CAND))
pubw = collections.Counter(); varw = collections.Counter(); nvar = collections.Counter(); ntri = 0
fam_idx = {"q3b": 0, "q15": 1, "lla": 2}
for i in range(len(A)):
    P = A[i][None, None] + Bq[:, None] + C[None, :]
    ok = np.all(np.abs(score(P) - tg) <= 0.0251, axis=-1)
    for b, c in zip(*np.nonzero(ok)):
        idx = (i, b, c); ww = w["q3b"][i] * w["q15"][b] * w["lla"][c]; tot += ww; ntri += 1
        cb = [np.array(keys["q3b"][i][IB]), np.array(keys["q15"][b][IB]), np.array(keys["lla"][c][IB])]
        s0 = sc1(cb[0] + cb[1] + cb[2])
        for f in FAM:
            lst = states[f][keys[f][idx[fam_idx[f]]]]; share = ww / len(lst)
            others = sum(cb[j] for j in range(3) if j != fam_idx[f])
            for pub, var, extra in lst:
                for e in pub:
                    pubw[e] += share
                    if e in var: varw[e] += share
                nvar[(f, sum(1 for e in var if e in pub))] += share
                for j, ex in enumerate(extra):
                    if ex is None: continue
                    d = sc1(others + np.array(ex)) - s0
                    if d > 0.0249: up[j] += share; gain[j] += share * d
                    elif d < -0.0249: dn[j] += share
print("\nvariantes publiques max par famille :", MAXV, "| triplets compatibles :", ntri, "| poids total :", int(tot))
if tot > 0:
    for f in FAM:
        print(" ", f, " ".join("%02d:pub %.2f var %.2f" % (e, pubw[e] / tot, (varw[e] / pubw[e] if pubw[e] else 0)) for e in FAM[f] if e not in HORS))
        print("     variantes publiques dans la famille :", " ".join("%d:%.2f" % (k, nvar[(f, k)] / tot) for k in range(0, 8) if nvar[(f, k)] > 0))
    print("envoi a un changement (base 0,80) : hausse | baisse | gain moyen si hausse")
    order = sorted(range(len(CAND)), key=lambda j: -(up[j] - dn[j]))
    for j in order: print("   %-14s %.2f | %.2f | %+.3f" % (CAND[j][0], up[j] / tot, dn[j] / tot, gain[j] / up[j] if up[j] > 0 else 0))
