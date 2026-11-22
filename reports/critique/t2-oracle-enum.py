"""t2-oracle : enumeration EXACTE des verites compatibles avec les 12 scores publics T2 (aucune requete, aucun .env).
Modele « S » : le public = toutes les paires internes aux familles entre guichets d'un sous-ensemble S (|S| entre SMIN et SMAX).
Etat d'un guichet : 0 = hors public, 1 = public et copie (modele de base), 2 = public et variante seule,
3 = public et variante en double (les guichets a l'etat 3 d'une meme famille servent UN meme modele modifie).
Score = 0,1 + 0,9 x max(0, MCC paires internes publiques), compare a l'observe a +-0,025 (arrondi a 0,05).
Familles supposees justes (5 equipes a 1,00 ; 1803 = 0,10).
Usage : python t2-oracle-enum.py <classe> [--alt2707] [--smin 14 --smax 22]
  classe : simple (0/1/2) | double (0/1/2/3, 0 ou 2 ou 3 guichets a l'etat 3 par famille)"""
import sys, itertools, collections, argparse, numpy as np, json
ap = argparse.ArgumentParser(); ap.add_argument("classe", default="simple", nargs="?"); ap.add_argument("--alt2707", action="store_true")
ap.add_argument("--alt2595", action="store_true"); ap.add_argument("--smin", type=int, default=14); ap.add_argument("--smax", type=int, default=22)
ap.add_argument("--out", default=None); ap.add_argument("--sans", default="")
a = ap.parse_args()
FAM = [["01","02","04","12","14","17","18","23","24","25","33"], ["03","06","07","08","20","21","22","26","28","30","32"], ["05","09","10","11","13","15","16","19","27","29","31"]]
FP8 = ["05","15","22"]; G = [["17","18"]]
def P(iso, groups=G): return (list(iso), [list(g) for g in groups])
# fichiers reconstitues (voir rapport) : iso = guichets seuls dans leur etiquette ; groups = groupes hors base
SUBS = [("1803 tempB",   P(["14","23","07","28","16"], []), .10),
        ("1882 fp8",     P(FP8, []), .40),
        ("1967 retenu",  P(FP8), .60),
        ("2052 17|18",   P(FP8 + ["17","18"], []), .60),
        ("2176 +11",     P(FP8 + ["11"]), .60),
        ("2383 +11+19",  P(FP8 + ["11","19"]), .60),
        ("2491 +11+28+03", P(FP8 + ["11","28","03"]), .60),
        ("2595 +09+11",  P(FP8 + (["09"] if a.alt2595 else ["09","11"])), .60),
        ("2707 +12+24",  P(FP8 + ["12","24"] + (["11"] if a.alt2707 else [])), .60),
        ("2795 H",       P(FP8 + ["17","18","04","12","03","21","32","09","11"], []), .30),
        ("2899 +01",     P(FP8 + ["01"]), .60)]
if a.sans: SUBS = [s for s in SUBS if not s[0].startswith(a.sans)]
OBS = np.array([s[2] for s in SUBS]); K = len(SUBS)
def predpairs(f, sub):
    iso, groups = sub; lab = {}
    for e in FAM[f]: lab[e] = 0
    for e in iso: lab[e] = 100 + int(e)
    for k, g in enumerate(groups):
        for e in g: lab[e] = 200 + k
    return np.array([lab[x] == lab[y] for x, y in itertools.combinations(FAM[f], 2)])
PAIRS = list(itertools.combinations(range(11), 2)); IA = np.array([p[0] for p in PAIRS]); IB = np.array([p[1] for p in PAIRS])
def fam_states(f):
    nl = 3 if a.classe == "simple" else 4
    X = np.array(list(itertools.product(range(nl), repeat=11)), dtype=np.int8)
    if nl == 4:
        n3 = (X == 3).sum(1); X = X[(n3 == 0) | (n3 == 2) | (n3 == 3)]
    pub = X > 0
    m = pub[:, IA] & pub[:, IB]
    same = (X[:, IA] == X[:, IB]) & ((X[:, IA] == 1) | (X[:, IA] == 3))
    Pt = (m & same).sum(1); Pn = (m & ~same).sum(1)
    PR = np.stack([predpairs(f, s[1]) for s in SUBS])            # K x 55
    tp = (m & same).astype(np.int32) @ PR.T.astype(np.int32); fp = (m & ~same).astype(np.int32) @ PR.T.astype(np.int32)
    V = np.concatenate([pub.sum(1)[:, None], Pt[:, None], Pn[:, None], tp, fp], 1).astype(np.int16)
    U, inv, cnt = np.unique(V, axis=0, return_inverse=True, return_counts=True)
    return X, U, inv.ravel(), cnt
F = [fam_states(f) for f in range(3)]
print(f"classe {a.classe} ; etats par famille : {[len(x[0]) for x in F]} ; vecteurs distincts : {[len(x[1]) for x in F]}", flush=True)
def ok_scores(tot):
    Pt, Pn, tp, fp = tot[:, 1:2], tot[:, 2:3], tot[:, 3:3 + K], tot[:, 3 + K:3 + 2 * K]
    fn = Pt - tp; tn = Pn - fp
    d = np.sqrt((tp + fp).astype(float) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.where(d > 0, (tp.astype(float) * tn - fp.astype(float) * fn) / np.maximum(d, 1e-9), 0.0)
    s = 0.1 + 0.9 * np.maximum(0, mcc)
    return (np.abs(s - OBS[None, :]) <= 0.025 + 1e-9).all(1), s
# combinaison F1 + F2, puis F3, par blocs ; filtre sur la taille de S
U1, U2, U3 = F[0][1], F[1][1], F[2][1]
c1, c2, c3 = F[0][3], F[1][3], F[2][3]
S12 = (U1[:, None, :].astype(np.int32) + U2[None, :, :]).reshape(-1, U1.shape[1]); w12 = (c1[:, None] * c2[None, :]).ravel()
i12 = np.arange(len(S12))
keep = (S12[:, 0] + U3[:, 0].min() <= a.smax) & (S12[:, 0] + U3[:, 0].max() >= a.smin); S12, w12, i12 = S12[keep], w12[keep], i12[keep]
U12, inv12 = np.unique(S12, axis=0, return_inverse=True); inv12 = inv12.ravel()
print(f"sommes F1+F2 distinctes : {len(U12)}", flush=True)
hits = []   # (index U12, index U3)
B = max(1, 4_000_000 // len(U3))
for st in range(0, len(U12), B):
    blk = U12[st:st + B]
    tot = (blk[:, None, :] + U3[None, :, :].astype(np.int32)).reshape(-1, U3.shape[1])
    ns = tot[:, 0]; good = (ns >= a.smin) & (ns <= a.smax)
    okk = np.zeros(len(tot), bool); okk[good] = ok_scores(tot[good])[0]
    for j in np.nonzero(okk)[0]: hits.append((st + j // len(U3), j % len(U3)))
print(f"combinaisons (vecteurs) compatibles : {len(hits)}", flush=True)
if not hits: sys.exit()
# poids et marges par guichet
w_u12 = np.bincount(inv12, weights=w12.astype(float), minlength=len(U12))
tot_w = 0.0; marg = np.zeros((33, 4)); size_hist = collections.Counter()
# marges par famille : somme des etats par vecteur distinct
def margs(f):
    X, U, inv, cnt = F[f]; M = np.zeros((len(U), 11, 4))
    for lab in range(4): np.add.at(M[:, :, lab], inv, (X == lab).astype(float))
    return M
M1, M2, M3 = margs(0), margs(1), margs(2)
# pour F1+F2 il faut separer : poids par (u1,u2) qui tombent dans U12[k]
n2 = len(U2)
pairs_of = collections.defaultdict(list)
for idx, k in zip(i12, inv12): pairs_of[k].append(idx)
H12 = collections.defaultdict(float); H3 = collections.defaultdict(float)
hk = collections.defaultdict(list)
for k, j in hits: hk[k].append(j)
for k, js in hk.items():
    js = np.array(js); w3 = c3[js].astype(float)
    for idx in pairs_of[k]:
        u1, u2 = divmod(idx, n2); w = float(c1[u1]) * c2[u2]
        W = w * w3.sum(); tot_w += W
        marg[0:11] += M1[u1] * c2[u2] * w3.sum()
        marg[11:22] += M2[u2] * c1[u1] * w3.sum()
        marg[22:33] += w * (M3[js] * (w3 / c3[js])[:, None, None]).sum(0)
        for jj, ww in zip(js, w3): size_hist[int(U1[u1, 0] + U2[u2, 0] + U3[jj, 0])] += w * ww
print(f"nombre d'etats (verite x S) compatibles : {tot_w:.4g}")
print("taille de S :", " ".join(f"{s}:{v / tot_w:.2f}" for s, v in sorted(size_hist.items())))
E = [e for f in FAM for e in f]; marg /= tot_w
print("guichet  P(public)  P(copie publique)  P(variante seule)  P(variante double)  P(variante|public)")
res = {}
for i, e in enumerate(E):
    pp = 1 - marg[i, 0]; pv = marg[i, 2] + marg[i, 3]
    res[e] = dict(pub=pp, copie=marg[i, 1], var=marg[i, 2], dbl=marg[i, 3])
    print(f"ep-{e}   {pp:5.2f}      {marg[i,1]:5.2f}            {marg[i,2]:5.2f}             {marg[i,3]:5.2f}             {pv / pp if pp > 1e-9 else float('nan'):5.2f}")
if a.out:
    json.dump(dict(total=tot_w, marg=res, sizes={str(k): v / tot_w for k, v in size_hist.items()}), open(a.out, "w"), indent=1)
