# Session relais : test de coherence de l'hypothese « filigrane = modele different » (aucune requete API).
# Verite FIXEE : variantes = {17,18,25} / {22,26,6} / {5,15,19}, toutes distinctes ; le reste = base de sa famille.
# On enumere seulement l'ensemble public (tout sous-ensemble >= 2 des guichets non exclus) et on garde ceux
# qui reproduisent les 6 scores informatifs. Usage : python t2-relais-filigrane.py [alt]
#   alt = autre guichet deguise a la place de 25 (ex. 2) pour comparer les hypotheses.
import itertools, collections, sys
import numpy as np
ALT = int(sys.argv[1]) if len(sys.argv) > 1 else 25
FAM = {"q3b": [1, 2, 4, 12, 14, 17, 18, 23, 24, 25, 33], "q15": [3, 6, 7, 8, 20, 21, 22, 26, 28, 30, 32], "lla": [5, 9, 10, 11, 13, 15, 16, 19, 27, 29, 31]}
HORS = {1, 3, 9, 11, 12, 19, 24, 28}
VAR = {17, 18, ALT, 22, 26, 6, 5, 15, 19} if ALT == 25 else {17, 18, ALT, 22, 26, 5, 15}
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
def iso(*xs): return lambda e: ("v%d" % e if e in xs else B(e))
SUBS = [("1882", L82, 0.40), ("1967", L67, 0.60), ("17|18", Lsp, 0.60), ("H", LH, 0.30), ("b26", B, 0.80), ("b26+10,13", B2, 0.60)]
CAND = [("isoler 25", iso(25)), ("isoler 02", iso(2)), ("isoler 06", iso(6)), ("isoler 25+06", iso(25, 6)), ("isoler 14", iso(14)), ("isoler 23", iso(23)), ("isoler 33", iso(33))]
ALL = [l for _, l, _ in SUBS] + [l for _, l in CAND]
def counts(pub, lab):
    tp = fp = tn = fn = 0
    for a, b in itertools.combinations(pub, 2):
        st = (a not in VAR) and (b not in VAR); sp = lab(a) == lab(b)
        if st and sp: tp += 1
        elif st: fn += 1
        elif sp: fp += 1
        else: tn += 1
    return (tp, fp, tn, fn)
sig = {}
for f, eps in FAM.items():
    cand = [e for e in eps if e not in HORS]
    d = collections.defaultdict(list)
    for k in range(0, len(cand) + 1):
        for pub in itertools.combinations(cand, k):
            d[tuple(counts(pub, l) for l in ALL)].append(pub)
    sig[f] = d
def sc(c):
    tp, fp, tn, fn = [c[..., i].astype(float) for i in range(4)]
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    with np.errstate(divide="ignore", invalid="ignore"): m = np.where(den > 0, (tp * tn - fp * fn) / den, 0.0)
    return 0.1 + 0.9 * m
keys = {f: list(sig[f]) for f in FAM}; arr = {f: np.array(keys[f]) for f in FAM}
w = {f: np.array([len(sig[f][k]) for k in keys[f]], float) for f in FAM}
tg = np.array([t for _, _, t in SUBS]); ns = len(SUBS)
tot = 0; pubw = collections.Counter(); out = collections.defaultdict(collections.Counter); ntri = 0
A, Bq, C = arr["q3b"], arr["q15"], arr["lla"]
for i in range(len(A)):
    P = A[i][None, None] + Bq[:, None] + C[None, :]; S = sc(P)
    ok = np.all(np.abs(S[..., :ns] - tg) <= 0.0251, axis=-1)
    for b, c in zip(*np.nonzero(ok)):
        ww = w["q3b"][i] * w["q15"][b] * w["lla"][c]; tot += ww; ntri += 1
        for j, (n, _) in enumerate(CAND): out[n][round(float(S[b, c, ns + j]), 2)] += ww
        for f, idx in (("q3b", i), ("q15", b), ("lla", c)):
            lst = sig[f][keys[f][idx]]
            for pub in lst:
                for e in pub: pubw[e] += ww / len(lst)
print("hypothese : variantes =", sorted(VAR), "| triplets compatibles :", ntri, "| poids :", int(tot))
if tot:
    for f in FAM: print(" ", f, " ".join("%02d:%.2f" % (e, pubw[e] / tot) for e in FAM[f] if e not in HORS), "(P public)")
    for n, _ in CAND: print("  %-14s" % n, " ".join("%.2f:%.0f%%" % (s, 100 * v / tot) for s, v in sorted(out[n].items())))
