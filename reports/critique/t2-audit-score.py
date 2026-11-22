"""Audit T2 (session t2-audit d'Adem). Aucune requete API, aucun .env.
1. score exact : 0,1 x MCC famille (toutes les paires) + 0,9 x MCC modele (paires de meme famille VRAIE), MCC ramene a 0.
2. plafond du fichier retenu (1967) selon le nombre k de variantes non isolees.
3. ABC : quelles (verites, masques publics) redonnent les 11 scores observes ? Compare plusieurs modeles de masque.
python t2-audit-score.py [nb_tirages]"""
import sys, itertools, numpy as np
rng = np.random.default_rng(7)
FAM = [["01","02","04","12","14","17","18","23","24","25","33"], ["03","06","07","08","20","21","22","26","28","30","32"],
       ["05","09","10","11","13","15","16","19","27","29","31"]]
E = [e for f in FAM for e in f]; ix = {e: i for i, e in enumerate(E)}; famof = np.array([k for k, f in enumerate(FAM) for _ in f])
WP = [(ix[a], ix[b]) for f in FAM for a, b in itertools.combinations(f, 2)]          # 165 paires internes
A = np.array([p[0] for p in WP]); B = np.array([p[1] for p in WP])

def mcc(tp, fp, fn, tn):
    d = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return np.maximum(0, np.where(d > 0, (tp * tn - fp * fn) / np.maximum(d, 1e-12), 0.0))

def lab(iso=(), groups=()):
    """etiquettes predites : base de famille = -1-famille ; isoles = propre etiquette ; groupes = etiquette commune"""
    l = -1 - famof.copy()
    for e in iso: l[ix[e]] = 100 + ix[e]
    for k, g in enumerate(groups):
        for e in g: l[ix[e]] = 300 + k
    return l

FP8 = ["05", "15", "22"]; G = (["17", "18"],)
SUBS = [("1803 B_temperature", lab(["14", "23", "07", "28", "16"]), 0.10), ("1882 fp8", lab(FP8), 0.40), ("1967 +{17,18}", lab(FP8, G), 0.60),
        ("2052 17|18", lab(FP8 + ["17", "18"]), 0.60), ("2176 +11", lab(FP8 + ["11"], G), 0.60), ("2383 +11+19", lab(FP8 + ["11", "19"], G), 0.60),
        ("2491 +28+03", lab(FP8 + ["28", "03"], G), 0.60), ("2595 +09+11", lab(FP8 + ["09", "11"], G), 0.60),
        ("2707 +12+24", lab(FP8 + ["12", "24"], G), 0.60), ("2795 H", lab(FP8 + ["17", "18", "04", "12", "03", "21", "32", "09", "11"]), 0.30),
        ("2899 +01", lab(FP8 + ["01"], G), 0.60)]
P = np.stack([(s[1][A] == s[1][B]) for s in SUBS]).astype(np.float32)         # (11, 165) predit "meme modele"
OBS = np.array([s[2] for s in SUBS])

def score_all(truth_lab):
    """score exact sur toutes les paires (familles justes => MCC famille = 1)"""
    t = (truth_lab[A] == truth_lab[B]).astype(np.float32)
    tp = P @ t; fp = P.sum(1) - tp; fn = t.sum() - tp; tn = len(t) - tp - fp - fn
    return 0.1 + 0.9 * mcc(tp, fp, fn, tn)

# ---------- 2. plafond ----------
def truth_from(var_groups):
    l = -1 - famof.copy()
    for k, g in enumerate(var_groups):
        for e in g: l[ix[e]] = 500 + k
    return l

def plafond():
    print("== 2. Score EXACT (toutes les paires) du fichier 1967 si ses 5 isoles sont justes ; b_f = copies par famille")
    print("   k = nombre de vraies variantes laissees dans la base par 1967 ; s1718 = 17 et 18 meme modele ?")
    res = {}
    for b1 in range(3, 10):
        for b2 in range(3, 10):
            for b3 in range(3, 10):
                vg = [["05"], ["15"], ["22"]]
                for f, b, known in zip(FAM, (b1, b2, b3), (["17", "18"], ["22"], ["05", "15"])):
                    rest = [e for e in f if e not in known]
                    vg += [[e] for e in rest[: 11 - b - len(known)]]
                k = 33 - b1 - b2 - b3 - 5
                for same in (True, False):
                    g = vg + ([["17", "18"]] if same else [["17"], ["18"]])
                    res.setdefault((k, same), []).append(score_all(truth_from(g))[2])
    for k in sorted({k for k, _ in res}):
        a = res[(k, True)]; c = res[(k, False)]
        print(f"   k={k:2d}  17=18 : {min(a):.3f}-{max(a):.3f}   17!=18 : {min(c):.3f}-{max(c):.3f}")

# ---------- 3. ABC ----------
KNOWN = [ix[e] for e in ("05", "15", "22", "17", "18")]
def draw_truth(n, dup, pk=0.85):
    """a priori commun a tous les masques : par famille, chaque guichet est variante avec une proba p_f ~ U(0,15 ; 0,75) ;
    les 5 isoles de 1967 le sont avec pk. Doublons : avec proba dup (deux fois), deux guichets d'une famille, s'ils sont
    tous deux variantes, partagent le meme modele."""
    pf = rng.uniform(0.15, 0.75, (n, 3))[:, famof]
    pf[:, KNOWN] = pk
    V = rng.random((n, 33)) < pf
    L = np.where(V, 100 + np.arange(33)[None, :], famof[None, :]).astype(np.int32)
    for _ in range(2):
        for k, f in enumerate(FAM):
            do = rng.random(n) < dup
            a = rng.integers(0, 11, n); b = (a + rng.integers(1, 11, n)) % 11
            ia = np.array([ix[f[0]]]) * 0 + np.array([ix[x] for x in f])[a]; ib = np.array([ix[x] for x in f])[b]
            m = do & V[np.arange(n), ia] & V[np.arange(n), ib]
            L[np.arange(n)[m], ib[m]] = L[np.arange(n)[m], ia[m]]
    return L

def masks(n, model):
    if model[0] == "pairs":
        q = model[1] if model[1] else rng.uniform(0.1, 0.6, (n, 1))
        return rng.random((n, 165)) < q
    s = rng.integers(model[1], model[2] + 1, n)
    M = rng.random((n, 33)).argsort(1) < s[:, None]
    return (M[:, A] & M[:, B]) if model[0] == "subset" else (M[:, A] | M[:, B])

def abc(model, dup, NT, R=40, chunk=5000):
    keepL, keepW, keepM = [], [], []; tot = 0; acc = 0
    while tot < NT:
        L = draw_truth(chunk, dup); T = (L[:, A] == L[:, B]).astype(np.float32)
        T = np.repeat(T, R, 0); Mk = masks(chunk * R, model).astype(np.float32)
        Mt = Mk * T; Mn = Mk * (1 - T)
        tp = Mt @ P.T; fp = Mn @ P.T; fn = Mt.sum(1, keepdims=True) - tp; tn = Mn.sum(1, keepdims=True) - fp
        sc = 0.1 + 0.9 * mcc(tp, fp, fn, tn)
        ok = np.all(np.abs(sc[:, 1:] - OBS[None, 1:]) <= 0.025 + 1e-9, 1) & (sc[:, 0] < 0.125)
        w = ok.reshape(chunk, R).sum(1); sel = w > 0
        keepL.append(L[sel]); keepW.append(w[sel]); keepM.append(Mk[ok]); acc += ok.sum(); tot += chunk
    return np.concatenate(keepL), np.concatenate(keepW).astype(float), np.concatenate(keepM), acc / (tot * R)

def resume(name, L, W, M, rate):
    n = len(L); print(f"-- {name}: taux d'acceptation {rate:.2e} ({int(W.sum())} couples verite-masque, {n} verites)")
    if n == 0: return
    w = W / W.sum(); var = L != famof[None, :]
    pv = (var * w[:, None]).sum(0)
    nvar = var.sum(1); print("   variantes (guichets) : moyenne %.1f ; P(>=15) %.2f" % ((nvar * w).sum(), (w * (nvar >= 15)).sum()))
    print("   P(variante) : " + " ".join(f"{e}={pv[ix[e]]:.2f}" for e in E))
    print(f"   P(17 et 18 meme modele) = {(w * (L[:, ix['17']] == L[:, ix['18']])).sum():.2f}")
    full = np.array([score_all(l)[2] for l in L])
    o = np.argsort(full); cw = np.cumsum(w[o])
    print("   score de 1967 sur TOUTES les paires : moyenne %.3f, mediane %.3f, p10 %.3f, p90 %.3f" % ((full * w).sum(), full[o][np.searchsorted(cw, .5)], full[o][np.searchsorted(cw, .1)], full[o][np.searchsorted(cw, .9)]))
    print("   paires internes publiques : moyenne %.1f" % M.sum(1).mean())
    for s, (nm, l, ob) in enumerate(SUBS[3:], 3):
        d = (P[s] != P[2]).astype(np.float32); print(f"     {nm:16s} paires changees vs 1967 : {int(d.sum()):3d} ; dont publiques (moy.) : {(M @ d).mean():.2f}")

if __name__ == "__main__":
    NT = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    if "--noabc" not in sys.argv: pass
    plafond()
    print("== 3. ABC, meme a priori sur les verites pour tous les masques ; 40 masques par verite")
    out = {}
    for model in [("pairs", 0.3), ("pairs", None), ("subset", 10, 26), ("touch", 3, 15)]:
        for dup in (0.0, 0.4):
            L, W, M, rate = abc(model, dup, NT); out[(model, dup)] = rate
            resume(f"masque {model} doublons {dup}", L, W, M, rate)
    print("== Rapport de vraisemblance (taux d'acceptation relatifs, a priori identique) :")
    ref = max(out.values()) or 1
    for k, v in out.items(): print(f"   {k}: {v:.2e}  ({v / ref:.3f})")
