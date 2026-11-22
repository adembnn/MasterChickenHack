"""t2-oracle : arbre d'envois. Loi a posteriori = melange a parts egales des hypotheses k3, k4, k5 et gabarit b4
(chaque tirage exact « etats » repondere par sum_h poids_h / taille_h : le melange pondere donc par la vraisemblance).
Pour la base courante, classe les envois a un changement (ou par classe) par E[max(score, meilleur)] puis par information,
puis, pour le premier envoi, recommence sur chaque issue (score exact observe) : arbre a 3 niveaux.
Aucune requete. Usage : python t2-oracle-arbre.py"""
import sys, math, json, itertools, collections, numpy as np
sys.argv = ["x"]
exec(open("E:/EAC/reports/critique/t2-oracle-envois.py").read().replace('if __name__ == "__main__":', 'if False:'))
exec(open("E:/EAC/reports/critique/t2-oracle-gabarit.py").read().split("Hsize =")[0].replace('X = np.load(L + "exact_etats_egale.npy")', 'X = np.load(L + "exact_etats_300k.npy")'))
def C2(n, k): return math.comb(n, k) if 0 <= k <= n else 0
def wk(X, k):
    w = np.ones(len(X))
    for f in range(3):
        xf = X[:, f * 11:(f + 1) * 11]; off = (xf == 0).sum(1); v = (xf == 2).sum(1)
        w *= np.array([C2(int(o), k - int(vv)) for o, vv in zip(off, v)])
    return w
HS = {3: math.comb(11, 3) ** 3, 4: math.comb(11, 4) ** 3, 5: math.comb(11, 5) ** 3}
W = sum(wk(X, k) / HS[k] for k in (3, 4, 5))
Hb4 = sum(math.comb(5, c) * C2(6, 4 - c) ** 3 for c in range(6))
W = W + np.array([w_b(x, 4) for x in X]) / Hb4
W = W / W.sum(); print(f"{len(X)} tirages exacts ; taille effective {1 / (W ** 2).sum():.0f}")
np.save(L + "melange_poids.npy", W)
print("P(public) | P(variante|public) sous le melange :")
print(" ".join(f"{e}:{(W * (X[:, i] > 0)).sum():.2f}|{(W * (X[:, i] == 2)).sum() / max((W * (X[:, i] > 0)).sum(), 1e-9):.2f}" for i, e in enumerate(E)))
CLS = {"T=1 {33,30,31}": ("33","30","31"), "prompt {02,08,29}": ("02","08","29"), "persona {23,20,16}": ("23","20","16"),
       "tronque {14,07,27}": ("14","07","27"), "filigrane {25,06,19}": ("25","06","19"), "{10,13}": ("10","13"), "{26,10}": ("26","10"), "{10,13,26}": ("10","13","26")}
def moves(lab):
    mv = {}
    for e in E:
        i = ix[e]
        if (lab == lab[i]).sum() > 1 and lab[i] == 0: l2 = lab.copy(); l2[i] = 300 + i; mv[f"iso {e}"] = l2
        elif lab[i] != 0 and e not in ("17", "18"): l2 = lab.copy(); l2[i] = 0; mv[f"base {e}"] = l2
    for nm, es in CLS.items():
        if all(lab[ix[e]] == 0 for e in es):
            l2 = lab.copy()
            for e in es: l2[ix[e]] = 300 + ix[e]
            mv["classe " + nm] = l2
    return mv
def sc(lab, Xs): return np.round(scores(Xs, pp(lab)[None])[:, 0] * 20) / 20
def rank(lab, best, w, Xs, top=8):
    rows = []
    for nm, l2 in moves(lab).items():
        s = sc(l2, Xs); ww = w / w.sum()
        up = ww[s > best + .025].sum(); dn = ww[s < best - .025].sum(); eq = 1 - up - dn
        vals = collections.defaultdict(float)
        for v, x in zip(s, ww): vals[round(float(v), 2)] += x
        p = np.array([x for x in vals.values() if x > 1e-12]); H = -(p * np.log2(p)).sum()
        rows.append(((np.maximum(s, best) * ww).sum(), up, eq, dn, H, nm, l2, s))
    rows.sort(key=lambda r: (-round(r[0], 3), -r[4]))
    return rows[:top]
base = lab1967(); s0 = sc(base, X); assert (s0 == .60).all()
print("\n=== Niveau 1 (base 1967, meilleur 0,60) : E[max] P(hausse) P(inch.) P(baisse) info(bits sur le score exact)")
R1 = rank(base, .60, W, X, 10)
for r in R1: print(f"{r[5]:28s} {r[0]:.3f} {r[1]:.2f} {r[2]:.2f} {r[3]:.2f} {r[4]:.2f}")
tree = {}
for r in R1[:2]:
    nm, l2, s = r[5], r[6], r[7]
    print(f"\n=== Si on envoie « {nm} » :")
    for v in sorted(set(np.round(s, 2))):
        m = np.abs(s - v) < 1e-6; pv = W[m].sum()
        if pv < 0.02: continue
        nb = l2 if v > .625 else base; best = max(.60, v)
        R2 = rank(nb, best, W[m], X[m], 3)
        print(f"  score {v:.2f} (proba {pv:.2f}) -> base {'nouvelle' if v > .625 else '1967'} ; suivants : " +
              " ; ".join(f"{q[5]} (E {q[0]:.3f}, hausse {q[1]:.2f}, baisse {q[3]:.2f})" for q in R2))
        tree[f"{nm} -> {v:.2f}"] = [q[5] for q in R2]
json.dump(tree, open(L + "arbre.json", "w"), indent=1, ensure_ascii=False)
