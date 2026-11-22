"""t2-oracle : a priori « deguisement » de la session relais (aucune requete). Chaque guichet est variante independamment avec
p_sans = r x p_avec (sans deguisement visible) ; 6 p_sans + 5 p_avec = 4 variantes par famille en moyenne.
Repondération exacte des tirages « etats » (300 000). Sorties : envois a un changement, issues de ep-26, fichier « MAP » glouton."""
import sys, numpy as np, collections, itertools, json
sys.argv = ["x"]
exec(open("E:/EAC/reports/critique/t2-oracle-envois.py").read().replace('if __name__ == "__main__":', 'if False:'))
L = "E:/EAC-lab/t2/reel/oracle/"; X = np.load(L + "exact_etats_300k.npy")
SANS = set("01 04 12 17 18 24 03 21 22 26 28 32 05 09 10 11 13 15".split())
def poids_r(r):
    pa = 4 / (6 * r + 5); ps = r * pa
    p = np.array([ps if e in SANS else pa for e in E])
    w = np.where(X == 2, p[None, :], np.where(X == 1, 1 - p[None, :], 1.0)).prod(1); return w / w.sum()
def sc(lab, Xs): return np.round(scores(Xs, pp(lab)[None])[:, 0] * 20) / 20
def table(w, base=None, best=.60, top=12, Xs=X):
    base = lab1967() if base is None else base; rows = []
    cands = {}
    for e in E:
        i = ix[e]
        if base[i] == 0: l = base.copy(); l[i] = 300 + i; cands[f"iso {e}"] = l
    for nm, es in {"{10,13}": ("10","13"), "{10,13,26}": ("10","13","26"), "prompt {02,08,29}": ("02","08","29"), "T=1 {33,30,31}": ("33","30","31"),
                   "filigrane {25,06,19}": ("25","06","19"), "persona {23,20,16}": ("23","20","16"), "tronque {14,07,27}": ("14","07","27")}.items():
        l = base.copy()
        for e in es: l[ix[e]] = 300 + ix[e]
        cands["classe " + nm] = l
    l = base.copy(); l[ix["33"]] = 200; cands["33 avec 17,18"] = l
    for nm, l in cands.items():
        s = sc(l, Xs); up = w[s > best + .025].sum(); dn = w[s < best - .025].sum()
        rows.append(((w * np.maximum(s, best)).sum(), up, dn, (w * s).sum(), nm))
    rows.sort(key=lambda r: -r[0])
    for r in rows[:top]: print(f"   {r[4]:24s} E[max] {r[0]:.3f} hausse {r[1]:.2f} baisse {r[2]:.2f} E[s] {r[3]:.2f}")
    return rows
out = {}
for r in (1, 3, 10):
    w = poids_r(r); print(f"\n===== a priori r = {r} (p_sans {4 * r / (6 * r + 5):.2f}, p_avec {4 / (6 * r + 5):.2f}) ; taille effective {1 / (w ** 2).sum():.0f}")
    print("   P(var|public) :", " ".join(f"{e}:{(w * (X[:, ix[e]] == 2)).sum() / max((w * (X[:, ix[e]] > 0)).sum(), 1e-9):.2f}" for e in E))
    table(w)
    l26 = lab1967(); l26[ix["26"]] = 326; s26 = sc(l26, X)
    for v in sorted(set(s26)):
        mk = s26 == v; pv = w[mk].sum()
        if pv < .02: continue
        print(f"   -- si ep-26 -> {v:.2f} (proba {pv:.2f}) ; suivants :")
        nb = l26 if v > .625 else lab1967()
        table(w[mk] / pv, nb, max(.6, v), 3, X[mk])
    # fichier MAP glouton : maximise P(score > 0,60) puis E[s], en isolant / remettant en base / groupant 33 avec 17-18
    cur = lab1967(); 
    def val(l): s = sc(l, X); return (w[s > .625].sum(), (w * s).sum())
    bestv = val(cur); log = []
    for it in range(8):
        cand = []
        for e in E:
            i = ix[e]
            if e in ("17", "18"): continue
            l = cur.copy(); l[i] = 0 if cur[i] != 0 else 300 + i; cand.append((val(l), ("-" if cur[i] else "+") + e, l))
        cand.sort(key=lambda c: c[0], reverse=True)
        if cand[0][0] <= bestv: break
        bestv, mv, cur = cand[0]; log.append(f"{mv} -> P(>0,60) {bestv[0]:.2f} E[s] {bestv[1]:.3f}")
    print("   fichier MAP glouton :", " ; ".join(log) if log else "aucun changement ne bat 1967")
    out[r] = [E[i] for i in range(33) if cur[i] != 0 and E[i] not in ("17", "18")]
json.dump(out, open(L + "map_deguise.json", "w"))
