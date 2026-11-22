# Session relais : enumeration sous le GABARIT (5 deguisements visibles = copies ; variantes parmi les guichets sans deguisement).
# Qwen-3B et Qwen-1.5B fixes par les scores publics ; en Llama, 4 variantes parmi {05,09,10,11,13,15} (15 choix).
# Ensemble public libre (tout sous-ensemble des guichets non exclus). Aucune requete API.
import itertools, collections, sys
import numpy as np
FAM = {"q3b": [1, 2, 4, 12, 14, 17, 18, 23, 24, 25, 33], "q15": [3, 6, 7, 8, 20, 21, 22, 26, 28, 30, 32], "lla": [5, 9, 10, 11, 13, 15, 16, 19, 27, 29, 31]}
HORS = {1, 3, 9, 11, 12, 19, 24, 28}
NV = int(sys.argv[1]) if len(sys.argv) > 1 else 4          # nombre de variantes parmi les 6 guichets Llama sans deguisement
def L67(e):
    if e in (5, 15, 22): return "v%d" % e
    if e in (17, 18): return "x"
    return "b"
def L82(e): return "v%d" % e if e in (5, 15, 22) else "b"
def Lsp(e): return "v%d" % e if e in (17, 18) else L67(e)
HISO = {3, 4, 5, 9, 11, 12, 15, 17, 18, 21, 22, 32}
def LH(e): return "v%d" % e if e in HISO else "b"
def B(e): return "v26" if e == 26 else L67(e)               # fichier retenu (0,80)
def plus(*iso):
    s = set(iso); return lambda e: ("v%d" % e if e in s else B(e))
def enbase(x): return lambda e: ("b" if e == x else B(e))
SUBS = [("1882", L82, 0.40), ("1967", L67, 0.60), ("17|18", Lsp, 0.60), ("H", LH, 0.30), ("b26", B, 0.80)]
CAND = [("+10", plus(10)), ("+13", plus(13)), ("+10,13", plus(10, 13)), ("15->base", enbase(15)), ("05->base", enbase(5)),
        ("17|18 separes", lambda e: ("v%d" % e if e in (17, 18) else B(e))), ("+25", plus(25)), ("+27", plus(27)), ("+29", plus(29))]
ALL = SUBS + [(n, f, None) for n, f in CAND]
def truths(f):
    if f == "q3b": yield {e: (("v", e) if e in (17, 18) else "b") for e in FAM[f]}, "17,18"
    elif f == "q15": yield {e: (("v", e) if e in (22, 26) else "b") for e in FAM[f]}, "22,26"
    else:
        for var in itertools.combinations([5, 9, 10, 11, 13, 15], NV):
            yield {e: (("v", e) if e in var else "b") for e in FAM[f]}, ",".join("%02d" % v for v in var)
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
    for t, name in truths(f):
        for r in range(2, len(cand) + 1):
            for pub in itertools.combinations(cand, r):
                sig[tuple(counts(pub, t, lab) for _, lab, _ in ALL)].append((pub, name))
    states[f] = sig
    print(f, "signatures", len(sig), "etats", sum(len(v) for v in sig.values()), flush=True)
def score(c):
    tp, fp, tn, fn = [c[..., i].astype(float) for i in range(4)]
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    with np.errstate(divide="ignore", invalid="ignore"): m = np.where(den > 0, (tp * tn - fp * fn) / den, 0.0)
    return 0.1 + 0.9 * m
keys = {f: list(states[f]) for f in FAM}; arr = {f: np.array(keys[f]) for f in FAM}
w = {f: np.array([len(states[f][k]) for k in keys[f]], float) for f in FAM}
tg = np.array([t for _, _, t in SUBS]); ns = len(SUBS)
tot = 0.0; up = np.zeros(len(CAND)); dn = np.zeros(len(CAND)); gain = np.zeros(len(CAND)); val = collections.defaultdict(lambda: np.zeros(len(CAND)))
truthw = collections.Counter(); sizew = collections.Counter(); pubw = collections.Counter()
A, Bq, C = arr["q3b"], arr["q15"], arr["lla"]
for i in range(len(A)):
    sc = score(A[i][None, None] + Bq[:, None] + C[None, :])
    ok = np.all(np.abs(sc[..., :ns] - tg) <= 0.0251, axis=-1)
    for b, c in zip(*np.nonzero(ok)):
        ww = w["q3b"][i] * w["q15"][b] * w["lla"][c]; tot += ww
        d = sc[b, c, ns:] - sc[b, c, ns - 1]
        up += ww * (d > 0.0249); dn += ww * (d < -0.0249); gain += ww * np.where(d > 0.0249, d, 0)
        lst = states["lla"][keys["lla"][c]]
        for pub, name in lst:
            truthw[name] += ww / len(lst)
            for e in pub: pubw[e] += ww / len(lst)
        for f, idx in (("q3b", i), ("q15", b)):
            l2 = states[f][keys[f][idx]]
            for pub, _ in l2:
                for e in pub: pubw[e] += ww / len(l2)
print("\npoids total des etats compatibles :", int(tot))
if tot > 0:
    print("verite Llama (les", NV, "variantes) : probabilite")
    for n, v in truthw.most_common(15): print("   {%s} %.3f" % (n, v / tot))
    print("P(public) :", " ".join("%02d:%.2f" % (e, pubw[e] / tot) for f in FAM for e in FAM[f] if e not in HORS))
    print("envoi (base 0,80) : hausse | baisse | gain moyen si hausse")
    for j, (n, _) in enumerate(CAND): print("   %-14s %.2f | %.2f | %+.3f" % (n, up[j] / tot, dn[j] / tot, gain[j] / up[j] if up[j] > 0 else 0))
