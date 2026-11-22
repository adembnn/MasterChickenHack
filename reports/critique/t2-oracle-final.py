"""t2-oracle : quel fichier final maximise l'esperance du score sur les paires CACHEES (et sur toutes les paires) ?
Tirages exacts a posteriori (melange k3/k4/k5/gabarit b4), completes : la nature des guichets hors public est tiree selon
l'hypothese du tirage (k variantes par famille au total, placees au hasard parmi les guichets hors public ; gabarit : classes liees).
Score cache = 0,1 + 0,9 x MCC sur les paires internes dont au moins un guichet est hors S. Aucune requete.
LIMITE : sans les lecteurs, la nature d'un guichet hors public ne vient que de la loi a priori. Ce calcul dit combien vaut
chaque guichet a isoler, il ne remplace pas les lecteurs (voir PRIOR_LECTEUR pour brancher leurs probabilites)."""
import sys, math, json, itertools, collections, numpy as np
sys.argv = ["x"]
exec(open("E:/EAC/reports/critique/t2-oracle-arbre.py").read().split("CLS = {")[0])
rng = np.random.default_rng(1)
# probabilite a priori « variante » apportee par les lecteurs, guichet par guichet (None = neutre). A remplir par t2-lecteur.
PRIOR_LECTEUR = {}
N = 60000; pick = rng.choice(len(X), N, p=W)
Xs = X[pick]; full = np.zeros((N, 33), bool)   # vrai = variante
which = rng.random(N)
# choix de l'hypothese de completion par tirage, proportionnel a sa part du poids
parts = np.stack([wk(Xs, k) / HS[k] for k in (3, 4, 5)] + [np.array([w_b(x, 4) for x in Xs]) / Hb4], 1)
parts = parts / parts.sum(1, keepdims=True); hyp = (parts.cumsum(1) < which[:, None]).sum(1)
for r in range(N):
    x = Xs[r]; v = x == 2
    if hyp[r] < 3:
        k = (3, 4, 5)[hyp[r]]
        for f in range(3):
            sl = slice(f * 11, (f + 1) * 11); off = np.nonzero(x[sl] == 0)[0]; need = k - int(v[sl].sum())
            ch = rng.choice(off, need, replace=False) if need > 0 else []
            vv = v[sl].copy(); vv[ch] = True; v[sl] = vv
    else:   # gabarit : classes liees puis reste a 4 par famille
        vv = v.copy(); nc = 0
        for c in CL:
            st = {int(x[ix[e]]) for e in c} - {0}
            isv = (2 in st) if st else (rng.random() < 0.5)
            for e in c: vv[ix[e]] = isv
            nc += isv
        ok = True
        for f in range(3):
            rest = np.array(REST[f]); off = rest[x[rest] == 0]; need = 4 - nc - int((x[rest] == 2).sum())
            if need < 0 or need > len(off): ok = False; break
            vv[rng.choice(off, need, replace=False)] = True
        if not ok: vv = v   # rare : rejet doux
        v = vv
    full[r] = v
pub = Xs > 0
lab_true = np.where(full, 100 + np.arange(33)[None, :], 0)
same_true = lab_true[:, A] == lab_true[:, B]
hidden = ~(pub[:, A] & pub[:, B])
def score_on(pred, mask):
    m = mask; t = same_true
    tp = (m & t & pred).sum(1); fp = (m & ~t & pred).sum(1); fn = (m & t & ~pred).sum(1); tn = (m & ~t & ~pred).sum(1)
    tp, fp, fn, tn = (z.astype(float) for z in (tp, fp, fn, tn))
    d = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.1 + 0.9 * np.maximum(0, np.where(d > 0, (tp * tn - fp * fn) / np.maximum(d, 1e-9), 0))
def file_pred(iso, grp=(("17", "18"),)):
    lab = np.zeros(33, int)
    for e in iso: lab[ix[e]] = 300 + ix[e]
    for k, g in enumerate(grp):
        for e in g: lab[ix[e]] = 400 + k
    return (lab[A] == lab[B])[None, :]
pv = full.mean(0)
print("P(variante) tous guichets (public + loi a priori pour les caches) :")
print(" ".join(f"{e}:{pv[i]:.2f}" for i, e in enumerate(E)))
FILES = {"1967": ["05", "15", "22"], "1967 + 24": ["05", "15", "22", "24"],
         "1967 + {01,12,24}": ["05","15","22","01","12","24"], "1967 + {03,28}": ["05","15","22","03","28"],
         "1967 + {09,11,19}": ["05","15","22","09","11","19"],
         "1967 + {01,12,24,03,28,09,11,19}": ["05","15","22","01","12","24","03","28","09","11","19"],
         "tout seul sauf copies sures": None}
pvthr = [e for i, e in enumerate(E) if pv[i] > 0.5 and e not in ("17", "18")]
FILES["isoler P(var)>0,5"] = pvthr
pvthr4 = [e for i, e in enumerate(E) if pv[i] > 0.4 and e not in ("17", "18")]
FILES["isoler P(var)>0,4"] = pvthr4
print("\nfichier                                   E[cache]  E[tout]  E[public]  (17 et 18 groupes)")
res = {}
for nm, iso in FILES.items():
    if iso is None: continue
    p = file_pred(iso)
    h = score_on(p, hidden).mean(); al = score_on(p, np.ones_like(hidden)).mean(); pu = score_on(p, ~hidden).mean()
    res[nm] = (h, al, pu); print(f"{nm:40s}  {h:.3f}    {al:.3f}    {pu:.3f}")
p = file_pred(FILES["isoler P(var)>0,5"], ()); print(f"{'meme, 17 | 18 separes':40s}  {score_on(p, hidden).mean():.3f}    {score_on(p, np.ones_like(hidden)).mean():.3f}")
# glouton : partir de 1967, ajouter / retirer un guichet tant que E[cache] monte
cur = list(FILES["1967"]); best = score_on(file_pred(cur), hidden).mean(); log = []
while True:
    cand = []
    for e in E:
        if e in ("17", "18"): continue
        nxt = [z for z in cur if z != e] if e in cur else cur + [e]
        cand.append((score_on(file_pred(nxt), hidden).mean(), e, nxt))
    cand.sort(reverse=True)
    if cand[0][0] <= best + 1e-4: break
    best, e, cur = cand[0]; log.append(f"{'+' if e in cur else '-'}{e} -> {best:.3f}")
print("\nglouton sur E[cache] (sans lecteurs) :", " ; ".join(log)); print("isoles :", sorted(cur))
json.dump(dict(pvar={e: float(pv[i]) for i, e in enumerate(E)}, fichiers=res, glouton=sorted(cur)), open(L + "final.json", "w"), indent=1)
