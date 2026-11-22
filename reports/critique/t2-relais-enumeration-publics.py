# Session relais : enumeration exhaustive « k guichets publics par famille » contre les scores publics T2.
# Usage : python t2-relais-enumeration-publics.py [k]   (k = guichets publics par famille, defaut 6)
# Aucune requete API. Les guichets hors public connus (inchanges quand on les isole) sont exclus d'office.
import itertools, sys, collections
import numpy as np

K = int(sys.argv[1]) if len(sys.argv) > 1 else 6
PRIOR = sys.argv[2] if len(sys.argv) > 2 else "base"   # "base" = une base + variantes uniques ; "part" = toute partition
FAM = {
    "q3b": [1, 2, 4, 12, 14, 17, 18, 23, 24, 25, 33],
    "q15": [3, 6, 7, 8, 20, 21, 22, 26, 28, 30, 32],
    "lla": [5, 9, 10, 11, 13, 15, 16, 19, 27, 29, 31],
}
HORS = {1, 3, 9, 11, 12, 19, 24, 28}

def lab_1967(e):
    if e in (5, 15, 22): return "v%d" % e
    if e in (17, 18): return "x"
    return "b"
def lab_1882(e):
    return "v%d" % e if e in (5, 15, 22) else "b"
def lab_split(e):
    return "v%d" % e if e in (17, 18) else lab_1967(e)
H_ISO = {3, 4, 5, 9, 11, 12, 15, 17, 18, 21, 22, 32}
def lab_H(e):
    return "v%d" % e if e in H_ISO else "b"
def plus(*iso):
    s = set(iso)
    return lambda e: ("v%d" % e if e in s else lab_1967(e))

SUBS = [("1882", lab_1882, 0.40), ("1967", lab_1967, 0.60), ("split17|18", lab_split, 0.60), ("H", lab_H, 0.30), ("+26", plus(26), 0.80)]
CAND = [("+25", plus(26, 25)), ("+10", plus(26, 10)), ("+13", plus(26, 13)), ("+10,13", plus(26, 10, 13)), ("+06", plus(26, 6)),
        ("+29", plus(26, 29)), ("+33", plus(26, 33)), ("+02", plus(26, 2)),
        ("33=17", lambda e: "x17" if e in (17, 33) else ("v18" if e == 18 else plus(26)(e)))]
ALL = SUBS + [(n, f, None) for n, f in CAND]

def partitions(items):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for p in partitions(rest):
        for i in range(len(p)):
            yield p[:i] + [[first] + p[i]] + p[i + 1:]
        yield [[first]] + p

def truths(pub):
    if PRIOR == "part":
        for p in partitions(list(pub)):
            t = {}
            for gi, g in enumerate(p):
                for e in g: t[e] = gi
            yield t
    else:  # une base (non vide) + variantes toutes distinctes
        for r in range(0, len(pub)):
            for var in itertools.combinations(pub, r):
                yield {e: (("v", e) if e in var else "b") for e in pub}

def counts(pub, truth, lab):
    tp = fp = tn = fn = 0
    for a, b in itertools.combinations(pub, 2):
        same_t = truth[a] == truth[b]
        same_p = lab(a) == lab(b)
        if same_t and same_p: tp += 1
        elif same_t: fn += 1
        elif same_p: fp += 1
        else: tn += 1
    return (tp, fp, tn, fn)

states = {}
for f, eps in FAM.items():
    cand = [e for e in eps if e not in HORS]
    sig = collections.defaultdict(list)
    for pub in itertools.combinations(cand, min(K, len(cand))):
        for t in truths(pub):
            s = tuple(counts(pub, t, lab) for _, lab, _ in ALL)
            sig[s].append((pub, t))
    states[f] = sig
    print(f, "candidats", cand, "| signatures", len(sig), "| etats", sum(len(v) for v in sig.values()))

def mcc(c):
    tp, fp, tn, fn = [c[..., i].astype(float) for i in range(4)]
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.where(den > 0, (tp * tn - fp * fn) / den, 0.0)
    return 0.1 + 0.9 * m

keys = {f: list(states[f].keys()) for f in FAM}
arr = {f: np.array(keys[f]) for f in FAM}            # (n, nsub, 4)
w = {f: np.array([len(states[f][k]) for k in keys[f]], dtype=float) for f in FAM}
targets = np.array([t for _, _, t in SUBS])
ns = len(SUBS)
tot = 0.0
pub_w = collections.Counter(); var_w = collections.Counter()
cand_up = np.zeros(len(CAND)); cand_dn = np.zeros(len(CAND)); cand_gain = np.zeros(len(CAND))
A, B, C = arr["q3b"], arr["q15"], arr["lla"]
ok_triples = 0
for i in range(len(A)):
    P = A[i][None, None] + B[:, None] + C[None, :]          # (nb, nc, nsub_all, 4)
    sc = mcc(P)                                              # (nb, nc, nsub_all)
    ok = np.all(np.abs(sc[..., :ns] - targets) <= 0.0251, axis=-1)
    if not ok.any(): continue
    jb, jc = np.nonzero(ok)
    for b, c in zip(jb, jc):
        ww = w["q3b"][i] * w["q15"][b] * w["lla"][c]
        tot += ww; ok_triples += 1
        base = sc[b, c, 4]
        d = sc[b, c, ns:] - base
        cand_up += ww * (d > 0.0249); cand_dn += ww * (d < -0.0249); cand_gain += ww * np.where(d > 0.0249, d, 0)
        for f, idx in (("q3b", i), ("q15", b), ("lla", c)):
            lst = states[f][keys[f][idx]]
            share = ww / len(lst)
            for pub, t in lst:
                for e in pub:
                    pub_w[e] += share
                    if t[e] != "b" and (PRIOR != "part"): var_w[e] += share
print("\nk =", K, "| a priori :", PRIOR, "| triplets de signatures compatibles :", ok_triples, "| poids total :", int(tot))
if tot > 0:
    print("\nguichet : P(public) | P(variante sachant public)")
    for f, eps in FAM.items():
        print(" ", f, " ".join("%02d:%.2f|%s" % (e, pub_w[e] / tot, ("%.2f" % (var_w[e] / pub_w[e]) if pub_w[e] > 0 and PRIOR != "part" else "-")) for e in eps if e not in HORS))
    print("\nenvoi candidat (base 1967 + ep-26, 0,80) : P(hausse) | P(baisse) | gain moyen si hausse")
    for j, (n, _) in enumerate(CAND):
        print("  %-10s %.2f | %.2f | %+.3f" % (n, cand_up[j] / tot, cand_dn[j] / tot, (cand_gain[j] / cand_up[j]) if cand_up[j] > 0 else 0))
