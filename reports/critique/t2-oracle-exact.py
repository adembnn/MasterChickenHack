"""t2-oracle : enumeration EXACTE des etats (S, nature des guichets publics) compatibles avec les 12 scores publics T2. Aucune requete.
Etats d'un guichet : 0 hors public, 1 copie publique, 2 variante publique (son propre modele).
Algorithme : (1) par famille, les 3^11 etats et leurs comptes de paires pour chaque fichier envoye ; (2) projection sur
(|S_f|, paires vraies, paires fausses, VP et FP du fichier 1967) ; triplets de projections compatibles avec 1967 et |S| ;
(3) pour chaque triplet, filtre des etats de chaque famille par les fichiers qui ne changent que cette famille ;
(4) combinaison des trois familles sur les fichiers couples (1803, H, 2491) ; (5) tirage de N etats exacts a posteriori.
Poids a priori (factorises par famille) : U (chaque guichet hors public copie ou variante : 2^hors), k4/k5/k6/k3 (exactement k
variantes par famille : C(hors, k - variantes publiques)), ou « etats » (chaque etat public compte 1).
Usage : python t2-oracle-exact.py <poids> [--alt2707] [--N 200000] [--taille uniforme|egale]"""
import sys, itertools, argparse, collections, json, time, numpy as np
from math import comb
ap = argparse.ArgumentParser(); ap.add_argument("poids"); ap.add_argument("--alt2707", action="store_true"); ap.add_argument("--N", type=int, default=200000)
ap.add_argument("--taille", default="egale"); ap.add_argument("--smin", type=int, default=14); ap.add_argument("--smax", type=int, default=22)
a = ap.parse_args(); rng = np.random.default_rng(0); t0 = time.time()
FAM = [["01","02","04","12","14","17","18","23","24","25","33"], ["03","06","07","08","20","21","22","26","28","30","32"], ["05","09","10","11","13","15","16","19","27","29","31"]]
E = [e for f in FAM for e in f]
FP8 = ["05","15","22"]; G = [["17","18"]]
SUBS = [("1803", ["14","23","07","28","16"], [], .10), ("1882", FP8, [], .40), ("1967", FP8, G, .60), ("2052", FP8 + ["17","18"], [], .60),
        ("2176", FP8 + ["11"], G, .60), ("2383", FP8 + ["11","19"], G, .60), ("2491", FP8 + ["11","28","03"], G, .60),
        ("2595", FP8 + ["09","11"], G, .60), ("2707", FP8 + ["12","24"] + (["11"] if a.alt2707 else []), G, .60),
        ("2795", FP8 + ["17","18","04","12","03","21","32","09","11"], [], .30), ("2899", FP8 + ["01"], G, .60),
        ("2998", FP8 + ["26"], G, .80), ("3102", FP8 + ["26","10","13"], G, .60),
        ("3200", FP8 + ["26","24"], G, .80)]
K = len(SUBS); NM = [s[0] for s in SUBS]; K0 = NM.index("1967"); OBS = np.array([s[3] for s in SUBS])
IA, IB = map(np.array, zip(*itertools.combinations(range(11), 2)))
def predpairs(f, iso, groups):
    lab = {e: 0 for e in FAM[f]}
    for e in iso:
        if e in lab: lab[e] = 100 + int(e)
    for k, g in enumerate(groups):
        for e in g:
            if e in lab: lab[e] = 200 + k
    return np.array([lab[FAM[f][i]] == lab[FAM[f][j]] for i, j in zip(IA, IB)])
def score(Pt, Pn, tp, fp):
    Pt, Pn, tp, fp = (np.asarray(v, float) for v in (Pt, Pn, tp, fp)); fn = Pt - tp; tn = Pn - fp
    d = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.1 + 0.9 * np.maximum(0, np.where(d > 0, (tp * tn - fp * fn) / np.maximum(d, 1e-9), 0))
def okk(Pt, Pn, tp, fp, k): return np.abs(score(Pt, Pn, tp, fp) - OBS[k]) <= 0.025 + 1e-9
NL = 4 if a.poids in ("c4", "etats4") else 3
X = np.array(list(itertools.product(range(NL), repeat=11)), dtype=np.int8)
if NL == 4: X = X[(X == 3).sum(1) <= 2]
def C(n, k): return comb(int(n), int(k)) if 0 <= k <= n else 0
def wfun(x):
    off = (x == 0).sum(1); v = (x == 2).sum(1)
    if a.poids == "etats": return np.ones(len(x))
    if a.poids == "U": return 2.0 ** off
    if a.poids == "etats4": return np.ones(len(x))
    if a.poids == "c4":   # 4 variantes par famille dont une paire sert le meme modele (etat 3 = membre de la paire)
        p3 = (x == 3).sum(1)
        return np.array([C(o, 2 - v) if q == 2 else (o * C(o - 1, 2 - v) if q == 1 else C(o, 2) * C(o - 2, 2 - v)) for o, v, q in zip(off, v, p3)], float)
    if a.poids.startswith("fix:"):   # verite fixee : liste des variantes
        V = set(a.poids[4:].split(",")); f = FAMI[0]
        tv = np.array([2 if e in V else 1 for e in FAM[f]]); return ((x == 0) | (x == tv[None, :])).all(1).astype(float)
    k = int(a.poids[1:]); return np.array([comb(int(o), k - int(vv)) if vv <= k else 0 for o, vv in zip(off, v)], float)
FAMD = []
FAMI = [0]
for f in range(3):
    FAMI[0] = f
    pub = X > 0; m = pub[:, IA] & pub[:, IB]; same = (X[:, IA] == X[:, IB]) & ((X[:, IA] == 1) | (X[:, IA] == 3))
    PR = np.stack([predpairs(f, s[1], s[2]) for s in SUBS]).astype(np.int32)
    tp = (m & same).astype(np.int32) @ PR.T; fp = (m & ~same).astype(np.int32) @ PR.T
    touch = [k for k in range(K) if not np.array_equal(PR[k], PR[K0])]
    proj = np.stack([pub.sum(1), (m & same).sum(1), (m & ~same).sum(1), tp[:, K0], fp[:, K0]], 1)
    w = wfun(X); keep = w > 0
    U, inv = np.unique(proj[keep], axis=0, return_inverse=True); idx = np.nonzero(keep)[0]
    groups = [idx[inv.ravel() == g] for g in range(len(U))]
    FAMD.append(dict(tp=tp, fp=fp, touch=touch, U=U, groups=groups, w=w))
    print(f"famille {f + 1} : fichiers qui la changent {[NM[k] for k in touch]} ; projections {len(U)}", flush=True)
COUPLED = [k for k in range(K) if sum(k in d["touch"] for d in FAMD) >= 2]
print("fichiers couples :", [NM[k] for k in COUPLED], flush=True)
U1, U2, U3 = (d["U"] for d in FAMD)
tri = []
for i1 in range(len(U1)):
    s = U1[i1][None, None, :] + U2[:, None, :] + U3[None, :, :]
    n, Pt, Pn, tp, fp = (s[..., j] for j in range(5))
    ok = okk(Pt, Pn, tp, fp, K0) & (n >= a.smin) & (n <= a.smax)
    for i2, i3 in zip(*np.nonzero(ok)): tri.append((i1, i2, i3))
print(f"triplets de projections compatibles avec 1967 et |S| : {len(tri)} ({time.time() - t0:.0f} s)", flush=True)
def sprior(n): return 1.0 if a.taille == "egale" else 1.0 / comb(33, int(n))
combos = []   # (poids, [etats compatibles par famille sous forme (indices, poids, cle couplee)], cles)
Z = 0.0; marg = np.zeros((33, 4)); nstates = 0.0; sizeh = collections.Counter()
for (i1, i2, i3) in tri:
    T = U1[i1].astype(int) + U2[i2] + U3[i3]; n, Pt, Pn, TP0, FP0 = T
    fam_sets = []
    for f, ig in enumerate((i1, i2, i3)):
        d = FAMD[f]; st = d["groups"][ig]; good = np.ones(len(st), bool)
        for k in d["touch"]:
            if k in COUPLED: continue
            dtp = d["tp"][st, k] - d["tp"][st, K0]; dfp = d["fp"][st, k] - d["fp"][st, K0]
            good &= okk(Pt, Pn, TP0 + dtp, FP0 + dfp, k)
        st = st[good]
        if len(st) == 0: break
        key = np.stack([np.where(k in d["touch"], d["tp"][st, k] - d["tp"][st, K0], 0) for k in COUPLED] +
                       [np.where(k in d["touch"], d["fp"][st, k] - d["fp"][st, K0], 0) for k in COUPLED], 1)
        fam_sets.append((st, key))
    if len(fam_sets) < 3: continue
    # combinaison sur les cles couplees
    agg = []
    for st, key in fam_sets:
        uk, inv = np.unique(key, axis=0, return_inverse=True); inv = inv.ravel()
        ww = np.bincount(inv, weights=FAMD[len(agg)]["w"][st], minlength=len(uk)); agg.append((uk, ww, inv, st))
    (k1, w1, _, _), (k2, w2, _, _), (k3, w3, _, _) = agg
    S3 = k1[:, None, None, :] + k2[None, :, None, :] + k3[None, None, :, :]
    good = np.ones(S3.shape[:3], bool); nc = len(COUPLED)
    for j, k in enumerate(COUPLED): good &= okk(Pt, Pn, TP0 + S3[..., j], FP0 + S3[..., nc + j], k)
    if not good.any(): continue
    W = good * w1[:, None, None] * w2[None, :, None] * w3[None, None, :] * sprior(n)
    tw = W.sum(); Z += tw; sizeh[int(n)] += tw
    # marges : poids de chaque cle de chaque famille
    for f, ax in enumerate(((1, 2), (0, 2), (0, 1))):
        wk = W.sum(axis=ax); uk, ww, inv, st = agg[f]
        per_state = wk[inv] / np.maximum(ww[inv], 1e-300) * FAMD[f]["w"][st]   # part de chaque etat
        for lab in range(4): marg[f * 11:(f + 1) * 11, lab] += (per_state[:, None] * (X[st] == lab)).sum(0)
    combos.append((tw, W, agg))
print(f"triplets retenus apres tous les fichiers : {len(combos)} ; masse totale (poids « {a.poids} ») : {Z:.6g} ({time.time() - t0:.0f} s)")
if Z == 0: sys.exit()
marg /= Z
print("taille de S :", " ".join(f"{s}:{v / Z:.2f}" for s, v in sorted(sizeh.items())))
out = {}
print("guichet P(public) P(copie pub) P(var pub) P(var|pub)")
for i, e in enumerate(E):
    pp = 1 - marg[i, 0]; out[e] = dict(pub=pp, copie=marg[i, 1], var=marg[i, 2] + marg[i, 3], dbl=marg[i, 3])
    print(f"ep-{e}  {pp:5.3f}  {marg[i, 1]:5.3f}  {marg[i, 2] + marg[i, 3]:5.3f}  {(marg[i, 2] + marg[i, 3]) / pp if pp > 1e-12 else float('nan'):5.3f}")
# tirages exacts a posteriori
tw = np.array([c[0] for c in combos]); pick = rng.choice(len(combos), a.N, p=tw / tw.sum())
SAMP = np.zeros((a.N, 33), np.int8)
for ci, cnt in zip(*np.unique(pick, return_counts=True)):
    _, W, agg = combos[ci]; flat = W.ravel(); cells = rng.choice(len(flat), cnt, p=flat / flat.sum())
    j1, j2, j3 = np.unravel_index(cells, W.shape); rows = np.nonzero(pick == ci)[0]
    for f, jj in enumerate((j1, j2, j3)):
        uk, ww, inv, st = agg[f]
        for r, j in zip(rows, jj):
            cand = st[inv == j]; pw = FAMD[f]["w"][cand]; SAMP[r, f * 11:(f + 1) * 11] = X[rng.choice(cand, p=pw / pw.sum())]
tag = "15s_" + f"{a.poids.replace(':', '_').replace(',', '-')}_{a.taille}" + ("_alt2707" if a.alt2707 else "")
np.save(f"E:/EAC-lab/t2/reel/oracle/exact_{tag}.npy", SAMP)
json.dump(dict(Z=Z, marg=out, tailles={str(k): v / Z for k, v in sizeh.items()}), open(f"E:/EAC-lab/t2/reel/oracle/exact_{tag}.json", "w"), indent=1)
print(f"{a.N} tirages exacts enregistres : exact_{tag}.npy ({time.time() - t0:.0f} s)")
