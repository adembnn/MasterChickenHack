"""t2-oracle (15 scores) : plan pour trouver la variante publique manquante, base = t2_1967_plus_ep26.csv (0,80). Aucune requete.
Tirages exacts a posteriori de t2-oracle-exact.py (15 scores) sous k4 (4 variantes par famille, chacune son modele),
etats (chaque etat public compte 1) et etats4/c4 (une paire de guichets d'une famille peut servir la meme variante, etat 3).
Pour chaque envoi : P(hausse/inchange/baisse) et, pour les paires {X,Y}, la table score -> nature de X et Y (decodable ?)."""
import sys, os, itertools, collections, numpy as np
sys.argv = ["x"]
exec(open("E:/EAC/reports/critique/t2-oracle-envois.py").read().replace('if __name__ == "__main__":', 'if False:'))
L = "E:/EAC-lab/t2/reel/oracle/"
def base26():
    l = lab1967(); l[ix["26"]] = 326; return l
def sc(lab, Xs): return np.round(scores(Xs, pp(lab)[None])[:, 0] * 20) / 20
NAT = {0: "hors", 1: "copie", 2: "VAR", 3: "VARdbl"}
SETS = [s for s in ("15s_k4_egale", "15s_etats_egale", "15s_c4_egale", "15s_etats4_egale") if os.path.exists(L + f"exact_{s}.npy")]
CANDX = ["02","14","23","25","33","06","07","08","20","30"]
for tag in SETS:
    X = np.load(L + f"exact_{tag}.npy"); s0 = sc(base26(), X); assert (s0 == .80).all(), collections.Counter(s0)
    print(f"\n######## {tag} : {len(X)} tirages ; controle base 0,80 OK")
    # nombre de variantes publiques encore dans la base
    miss = np.zeros(len(X), int); fam_miss = collections.Counter()
    for f in range(3):
        xf = X[:, f * 11:(f + 1) * 11]; b = base26()[f * 11:(f + 1) * 11]
        mf = ((xf >= 2) & (b[None, :] == 0)).sum(1); miss += mf
        for k in range(1, 4): fam_miss[(f, k)] += (mf == k).mean()
    print("   variantes publiques encore en base :", dict(collections.Counter(miss)), "| par famille (famille, nombre) : ", {k: round(v, 3) for k, v in fam_miss.items() if v > 0})
    print("   isolees a tort (copie publique isolee) :", {e: round(float((X[:, ix[e]] == 1).mean()), 3) for e in ("05","15","17","18","22","26")})
    rows = []
    def test(nm, l):
        s = sc(l, X); up = (s > .825).mean(); dn = (s < .775).mean(); c = collections.Counter(np.round(s, 2))
        p = np.array(list(c.values())) / len(s); H = -(p * np.log2(p)).sum()
        rows.append((up, H, nm, dn, " ".join(f"{k:.2f}:{v / len(s):.2f}" for k, v in sorted(c.items()))))
    for e in CANDX:
        l = base26(); l[ix[e]] = 300 + ix[e]; test(f"iso {e}", l)
    for e in ("05","15","17","18","22"):
        l = base26(); l[ix[e]] = 0; test(f"base {e}", l)
    for a_, b_ in (("05","15"),):
        l = base26(); l[ix[a_]] = l[ix[b_]] = 777; test(f"grp {a_}={b_}", l)
    for e in ("02","14","23","25","33"):
        l = base26(); l[ix[e]] = 200; test(f"{e} avec 17,18", l)
    for e in ("06","07","08","20","30"):
        l = base26(); l[ix[e]] = 326; test(f"{e} avec 26", l)
    for nm, es in {"T=1 {33,30}": ("33","30"), "prompt {02,08}": ("02","08"), "persona {23,20}": ("23","20"), "tronque {14,07}": ("14","07"), "filigrane {25,06}": ("25","06")}.items():
        l = base26()
        for e in es: l[ix[e]] = 300 + ix[e]
        test("classe " + nm, l)
    rows.sort(key=lambda r: (-r[0], -r[1]))
    print("   envoi                 P(hausse) P(baisse) info(bits)  scores")
    for r in rows: print(f"   {r[2]:20s}  {r[0]:.2f}      {r[3]:.2f}     {r[1]:.2f}     {r[4]}")
    # paires decodables
    print("   paires {X,Y} isolees ensemble (meme famille ou non) : P(hausse), et decodage score -> (nature X, nature Y)")
    best = []
    for a_, b_ in itertools.combinations(CANDX, 2):
        l = base26(); l[ix[a_]] = 300 + ix[a_]; l[ix[b_]] = 300 + ix[b_]; s = sc(l, X)
        na = np.vectorize(NAT.get)(X[:, ix[a_]]); nb = np.vectorize(NAT.get)(X[:, ix[b_]])
        # ambiguite : pour chaque score, la nature (VAR ou non) de X et de Y est-elle unique ?
        amb = 0.0; tab = {}
        for v in np.unique(s):
            m = s == v; va = (X[m, ix[a_]] >= 2); vb = (X[m, ix[b_]] >= 2)
            fa, fb = va.mean(), vb.mean(); tab[round(float(v), 2)] = (round(float(m.mean()), 2), round(float(fa), 2), round(float(fb), 2))
            amb += m.mean() * (min(fa, 1 - fa) + min(fb, 1 - fb))
        best.append((-(s > .825).mean(), amb, a_, b_, tab))
    best.sort()
    for nb_, amb, a_, b_, tab in best[:12]:
        print(f"   {{{a_},{b_}}} hausse {-nb_:.2f} ambiguite {amb:.3f} | score: (proba, P(X var), P(Y var)) " + " ".join(f"{k:.2f}:{v}" for k, v in sorted(tab.items())))
    np.save(L + f"b26_{tag}_ok.npy", np.array([1]))
