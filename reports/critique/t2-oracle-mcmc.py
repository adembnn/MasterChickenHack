"""t2-oracle : echantillonnage UNIFORME des etats (S, verite) compatibles avec les 12 scores publics T2 (aucune requete).
Etape 1 : recuit jusqu'a un etat compatible. Etape 2 : marche de Metropolis contrainte (un guichet change d'etat, on accepte
seulement si les 12 scores restent exacts et |S| dans [SMIN, SMAX]) : la loi limite est uniforme sur les etats compatibles.
Etats : 0 hors public, 1 copie publique, 2 variante seule publique, 3/4 = deux modeles modifies servis chacun par plusieurs guichets
(classe « double »). Usage : python t2-oracle-mcmc.py <simple|double> <graine> [--alt2707] [--test]"""
import sys, itertools, argparse, collections, json, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("classe"); ap.add_argument("seed", type=int); ap.add_argument("--alt2707", action="store_true")
ap.add_argument("--C", type=int, default=3000); ap.add_argument("--burn", type=int, default=3000); ap.add_argument("--keep", type=int, default=3000)
ap.add_argument("--smin", type=int, default=14); ap.add_argument("--smax", type=int, default=22); ap.add_argument("--out", default=None)
a = ap.parse_args(); rng = np.random.default_rng(a.seed); NL = 3 if a.classe == "simple" else 5
FAM = [["01","02","04","12","14","17","18","23","24","25","33"], ["03","06","07","08","20","21","22","26","28","30","32"], ["05","09","10","11","13","15","16","19","27","29","31"]]
E = [e for f in FAM for e in f]; ix = {e: i for i, e in enumerate(E)}
pairs = [(ix[x], ix[y]) for f in FAM for x, y in itertools.combinations(f, 2)]; A = np.array([p[0] for p in pairs]); B = np.array([p[1] for p in pairs])
FAMID = np.array([i // 11 for i in range(33)])
def pred(iso, groups=(["17", "18"],)):
    lab = np.zeros(33, int)
    for e in iso: lab[ix[e]] = 100 + ix[e]
    for k, g in enumerate(groups):
        for e in g: lab[ix[e]] = 200 + k
    return (lab[A] == lab[B]).astype(np.float32)
FP8 = ["05", "15", "22"]
SUBS = [("1803", pred(["14","23","07","28","16"], ()), .10), ("1882", pred(FP8, ()), .40), ("1967", pred(FP8), .60),
        ("2052", pred(FP8 + ["17","18"], ()), .60), ("2176", pred(FP8 + ["11"]), .60), ("2383", pred(FP8 + ["11","19"]), .60),
        ("2491", pred(FP8 + ["11","28","03"]), .60), ("2595", pred(FP8 + ["09","11"]), .60),
        ("2707", pred(FP8 + ["12","24"] + (["11"] if a.alt2707 else [])), .60),
        ("2795", pred(FP8 + ["17","18","04","12","03","21","32","09","11"], ()), .30), ("2899", pred(FP8 + ["01"]), .60)]
PF = np.stack([s[1] for s in SUBS]); OBS = np.array([s[2] for s in SUBS], dtype=np.float32)
def truth_same(x):
    # 1 = base de la famille ; 3, 4 = modeles modifies partages (meme famille) ; 2 = seul
    xa, xb = x[:, A], x[:, B]
    return (xa == xb) & (xa != 2) & (xa != 0)
def scores(x, PFm=PF):
    pub = x > 0; m = pub[:, A] & pub[:, B]; t = truth_same(x)
    Mt = (m & t).astype(np.float32); Mn = (m & ~t).astype(np.float32)
    tp = Mt @ PFm.T; fp = Mn @ PFm.T; fn = Mt.sum(1, keepdims=True) - tp; tn = Mn.sum(1, keepdims=True) - fp
    d = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.1 + 0.9 * np.maximum(0, np.where(d > 0, (tp * tn - fp * fn) / np.maximum(d, 1e-9), 0))
def dev(x):
    s = scores(x); d = np.maximum(0, np.abs(s - OBS[None, :]) - 0.025 - 1e-6)
    n = (x > 0).sum(1); dn = np.maximum(0, a.smin - n) + np.maximum(0, n - a.smax)
    return (d / 0.01) ** 2 @ np.ones(len(SUBS)) + dn, d.max(1) + dn
def valid_dup(x):   # un label 3 ou 4 doit avoir >= 2 guichets dans une meme famille
    if NL == 3: return np.ones(len(x), bool)
    ok = np.ones(len(x), bool)
    for lab in (3, 4):
        for f in range(3):
            c = (x[:, f * 11:(f + 1) * 11] == lab).sum(1); ok &= (c == 0) | (c >= 2)
    return ok
C = a.C; x = rng.integers(0, 3, (C, 33)).astype(np.int8); en, _ = dev(x)
for step in range(4000):
    T = max(0.3, 6.0 * (1 - step / 3000))
    y = x.copy(); j = rng.integers(0, 33, C); y[np.arange(C), j] = rng.integers(0, NL, C)
    e2, _ = dev(y); e2 = np.where(valid_dup(y), e2, 1e9)
    acc = rng.random(C) < np.exp(np.minimum(0, (en - e2) / T)); x[acc] = y[acc]; en[acc] = e2[acc]
_, d0 = dev(x); good = (d0 < 1e-9) & valid_dup(x)
x = x[good]; C2 = len(x)
print(f"{C2} chaines sur {C} atteignent un etat compatible", flush=True)
if C2 == 0: sys.exit()
samples = []
for step in range(a.burn + a.keep):
    y = x.copy(); j = rng.integers(0, 33, C2); y[np.arange(C2), j] = rng.integers(0, NL, C2)
    if NL == 5 and rng.random() < 0.5:      # echange de deux guichets d'une meme famille (melange)
        f = rng.integers(0, 3, C2); p = f * 11 + rng.integers(0, 11, C2); q = f * 11 + rng.integers(0, 11, C2)
        y = x.copy(); r = np.arange(C2); tmp = y[r, p].copy(); y[r, p] = y[r, q]; y[r, q] = tmp
    _, d = dev(y); acc = (d < 1e-9) & valid_dup(y); x[acc] = y[acc]
    if step >= a.burn and step % 10 == 0: samples.append(x.copy())
X = np.concatenate(samples)
# canonicaliser les labels 3/4 (echange 3<->4 equivalent) pour compter les etats distincts
U = np.unique(X, axis=0)
print(f"echantillons {len(X)} ; etats distincts visites {len(U)} ; |S| moyen {(X > 0).sum(1).mean():.1f} (min {(X>0).sum(1).min()}, max {(X>0).sum(1).max()})")
out = {}
print("guichet P(public) P(copie pub) P(var pub) P(var|pub)")
for e in E:
    c = X[:, ix[e]]; pp = (c > 0).mean(); pv = ((c >= 2)).mean()
    out[e] = dict(pub=float(pp), copie=float((c == 1).mean()), var=float(pv))
    print(f"ep-{e} {pp:5.2f} {(c==1).mean():5.2f} {pv:5.2f} {pv / pp if pp > 0 else float('nan'):5.2f}")
np.save(f"E:/EAC-lab/t2/reel/oracle/mcmc_{a.classe}_{a.seed}.npy", X[::max(1, len(X) // 20000)])
if a.out: json.dump(out, open(a.out, "w"), indent=1)
