"""t2-oracle (15 scores) : recherche de la variante publique manquante en un minimum de creneaux (aucune requete).
Base t2_1967_plus_ep26.csv (0,80). Tirages exacts k4 (15 scores). Tests = isoler un groupe de guichets candidats.
Politique gloutonne : a chaque creneau, le test qui maximise l'information sur le score exact ; on simule toutes les issues."""
import sys, itertools, collections, numpy as np
sys.argv = ["x"]
exec(open("E:/EAC/reports/critique/t2-oracle-envois.py").read().replace('if __name__ == "__main__":', 'if False:'))
L = "E:/EAC-lab/t2/reel/oracle/"; tag = sys.argv[1] if len(sys.argv) > 1 else "15s_k4_egale"
X = np.load(L + f"exact_{tag}.npy")
def base26():
    l = lab1967(); l[ix["26"]] = 326; return l
BASE = base26()
CAND = ["02","14","23","25","33","06","07","08","20","30"]
# localisation de la variante manquante
loc = np.full(len(X), "", dtype=object)
for e in CAND + [x for x in E if x not in CAND]:
    m = (X[:, ix[e]] >= 2) & (BASE[ix[e]] == 0); loc[m] = e
print("ou est la variante publique manquante :", {k: round(v / len(X), 3) for k, v in collections.Counter(loc).most_common()})
TESTS = {}
for r in (1, 2, 3, 4, 5):
    for g in itertools.combinations(CAND, r): TESTS["iso " + ",".join(g)] = g
SC = {}
for nm, g in TESTS.items():
    l = BASE.copy()
    for e in g: l[ix[e]] = 300 + ix[e]
    SC[nm] = np.round(scores(X, pp(l)[None])[:, 0] * 20) / 20
def H(s, m):
    c = collections.Counter(s[m]); p = np.array(list(c.values())) / m.sum(); return -(p * np.log2(p)).sum()
def best_test(m, k=5):
    rows = sorted(((H(SC[nm], m), -(SC[nm][m] > .825).mean(), nm) for nm in TESTS), reverse=True)
    return rows[:k]
print("\nmeilleurs tests maintenant (info en bits, P(hausse)) :")
for h, up, nm in best_test(np.ones(len(X), bool), 8): print(f"   {nm:24s} info {h:.2f}  hausse {-up:.2f}")
def decode(nm, m):
    s = SC[nm]; out = []
    for v in sorted(set(s[m])):
        mm = m & (s == v); c = collections.Counter(loc[mm]); out.append((v, mm.mean() / m.mean(), dict((k, round(n / mm.sum(), 2)) for k, n in c.most_common(4))))
    return out
for nm in ("iso 02", "iso 02,25,33", "iso 06,08,20,30", "iso 02,25,33,06,08", "iso 02,33"):
    print(f"\n{nm} : score -> (proba, ou est la variante)")
    for v, p, d in decode(nm, np.ones(len(X), bool)): print(f"   {v:.2f} ({p:.2f}) {d}")
# politique gloutonne : nombre de creneaux pour localiser (toutes les issues), depart = ep-02 deja programme
def run(m, depth, first=None, log=""):
    c = collections.Counter(loc[m])
    if len(c) == 1 or depth == 0: return [(m.mean(), depth, log, c.most_common(1)[0][0])]
    nm = first or best_test(m, 1)[0][2]; res = []
    for v in sorted(set(SC[nm][m])):
        mm = m & (SC[nm] == v); res += run(mm, depth - 1, None, log + f"{nm}->{v:.2f} | ")
    return res
res = run(np.ones(len(X), bool), 5, "iso 02")
print("\narbre glouton apres « iso 02 » (proba de la branche, creneaux restants, chemin, variante localisee) :")
for p, d, lg, where in sorted(res, key=lambda r: -r[0])[:30]:
    print(f"   {p:.3f}  {5 - d} envois  {lg}=> {where}")
print("esperance du nombre d'envois pour localiser :", round(sum(p * (5 - d) for p, d, _, _ in res), 2))
