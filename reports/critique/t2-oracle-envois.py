"""t2-oracle : issue attendue des envois candidats (un changement par rapport a 1967) sur les etats echantillonnes.
Usage : python t2-oracle-envois.py <fichiers .npy d'echantillons...>"""
import sys, itertools, collections, numpy as np
FAM = [["01","02","04","12","14","17","18","23","24","25","33"], ["03","06","07","08","20","21","22","26","28","30","32"], ["05","09","10","11","13","15","16","19","27","29","31"]]
E = [e for f in FAM for e in f]; ix = {e: i for i, e in enumerate(E)}
pairs = [(ix[x], ix[y]) for f in FAM for x, y in itertools.combinations(f, 2)]; A = np.array([p[0] for p in pairs]); B = np.array([p[1] for p in pairs])
FP8 = ["05", "15", "22"]
def lab1967():
    lab = np.zeros(33, int)
    for e in FP8: lab[ix[e]] = 100 + ix[e]
    lab[ix["17"]] = lab[ix["18"]] = 200
    return lab
def pp(lab): return (lab[A] == lab[B]).astype(np.float32)
def scores(x, PFm):
    pub = x > 0; m = pub[:, A] & pub[:, B]; xa, xb = x[:, A], x[:, B]; t = (xa == xb) & (xa != 2) & (xa != 0)
    Mt = (m & t).astype(np.float32); Mn = (m & ~t).astype(np.float32)
    tp = Mt @ PFm.T; fp = Mn @ PFm.T; fn = Mt.sum(1, keepdims=True) - tp; tn = Mn.sum(1, keepdims=True) - fp
    d = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.1 + 0.9 * np.maximum(0, np.where(d > 0, (tp * tn - fp * fn) / np.maximum(d, 1e-9), 0))
def iso(*es):
    lab = lab1967()
    for e in es: lab[ix[e]] = 300 + ix[e]
    return lab
def base(*es):
    lab = lab1967()
    for e in es: lab[ix[e]] = 0
    return lab
def grp(es):
    lab = lab1967()
    for e in es: lab[ix[e]] = 400
    return lab
CANDS = {}
for f in FAM:
    for e in f:
        if e not in ("05", "15", "22", "17", "18"): CANDS[f"iso {e}"] = iso(e)
for e in ("05", "15", "22", "17", "18"): CANDS[f"base {e}"] = base(e)
CANDS["grp 05+15"] = grp(["05", "15"])
for nm, es in {"T=1 {33,30,31}": ("33","30","31"), "prompt {02,08,29}": ("02","08","29"), "persona {23,20,16}": ("23","20","16"),
               "tronque {14,07,27}": ("14","07","27"), "filigrane {25,06,19}": ("25","06","19"), "{10,13}": ("10","13"),
               "simples {04,32,10}": ("04","32","10"), "{26,10}": ("26", "10")}.items(): CANDS["classe " + nm] = iso(*es)
lab = lab1967()
for e in ("12","17","18","24","03","22","26","28","05","09","11","15"): lab[ix[e]] = 500 + ix[e]
CANDS["candidat brief (12 var. seules)"] = lab
if __name__ == "__main__":
    X = np.concatenate([np.load(p) for p in sys.argv[1:]]); print(f"{len(X)} etats")
    PF = np.stack(list(CANDS.values()) and [pp(l) for l in CANDS.values()])
    s0 = np.round(scores(X, pp(lab1967())[None])[:, 0] * 20) / 20
    print("controle 1967 :", collections.Counter(np.round(s0, 2)))
    S = np.round(scores(X, PF) * 20) / 20
    rows = []
    for k, nm in enumerate(CANDS):
        s = S[:, k]; up = (s > .625).mean(); dn = (s < .575).mean(); eq = 1 - up - dn
        ps = np.array([up, eq, dn]); H = -(ps[ps > 0] * np.log2(ps[ps > 0])).sum()
        rows.append((np.maximum(s, .60).mean(), up, eq, dn, H, s.mean(), nm, collections.Counter(np.round(s, 2)).most_common(4)))
    rows.sort(key=lambda r: (-r[0], -r[4]))
    print("envoi                              E[max(s,0.60)] P(hausse) P(inch.) P(baisse) info(bits) E[s]  scores frequents")
    for r in rows: print(f"{r[6]:34s} {r[0]:.3f}  {r[1]:.2f}  {r[2]:.2f}  {r[3]:.2f}  {r[4]:.2f}  {r[5]:.2f}  " + " ".join(f"{k:.2f}:{v / len(S):.2f}" for k, v in r[7]))

def poids(X, nvar):
    """poids exact de l'hypothese « exactement nvar variantes par famille (seules) » : on place les variantes manquantes
    parmi les guichets hors public. nvar=None : uniforme sur les etats."""
    from math import comb
    if nvar is None: return np.ones(len(X))
    w = np.ones(len(X))
    for f in range(3):
        xf = X[:, f * 11:(f + 1) * 11]; v = (xf >= 2).sum(1); off = (xf == 0).sum(1)
        w *= np.array([comb(int(o), nvar - int(vv)) if vv <= nvar else 0 for o, vv in zip(off, v)])
    return w
def classer(X, w, top=45):
    PF = np.stack([pp(l) for l in CANDS.values()]); S = np.round(scores(X, PF) * 20) / 20; w = w / w.sum(); rows = []
    for k, nm in enumerate(CANDS):
        s = S[:, k]; up = w[s > .625].sum(); dn = w[s < .575].sum(); eq = 1 - up - dn
        ps = np.array([up, eq, dn]); H = -(ps[ps > 1e-12] * np.log2(ps[ps > 1e-12])).sum()
        rows.append(((np.maximum(s, .60) * w).sum(), up, eq, dn, H, (s * w).sum(), nm))
    rows.sort(key=lambda r: (-r[0], -r[4]))
    for r in rows[:top]: print(f"{r[6]:34s} {r[0]:.3f}  {r[1]:.2f}  {r[2]:.2f}  {r[3]:.2f}  {r[4]:.2f}  {r[5]:.2f}")
