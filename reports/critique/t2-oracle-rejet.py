"""t2-oracle : echantillonnage EXACT par rejet (aucune requete). On tire (verite, masque public) selon la loi a priori d'une
hypothese, on calcule les 12 scores des fichiers envoyes, on garde les tirages qui les redonnent tous (a +-0,025).
- La part gardee = vraisemblance de l'hypothese (comparable d'une hypothese a l'autre : facteurs de Bayes).
- Les tirages gardes = loi a posteriori exacte (P(public), P(variante), issue des envois candidats).
Masques : « S » = guichets publics, |S| uniforme dans [14, 22] ; « paires » = chaque paire interne publique avec proba 0,3.
Hypotheses de verite (chaque variante est son propre modele sauf mention) :
  U     : chaque guichet variante avec proba 1/2, independamment
  a4    : 4 variantes par famille (comme les ref : 5 copies + 4 variantes sur 9 -> 7 + 4 sur 11), placees au hasard
  e5/e6 : 5 ou 6 variantes par famille
  e3    : 3 variantes par famille
  b     : gabarit, 5 classes alignees (T=1, prompt, persona, tronque, filigrane) de meme nature dans les 3 familles ;
          les 6 autres guichets de chaque famille : meme nombre de variantes dans les 3 familles (4 variantes en tout par famille)
  c4    : 4 variantes par famille dont deux servent le MEME modele modifie (variante en double)
  d4    : comme a4 mais les variantes peuvent etre deguisees (aucune contrainte sur les classes : identique a a4 cote score)
Usage : python t2-oracle-rejet.py <hyp> <masque S|paires> <millions de tirages> [graine]"""
import sys, itertools, collections, numpy as np, json, time
hyp, mask, M = sys.argv[1], sys.argv[2], float(sys.argv[3]); seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
rng = np.random.default_rng(seed)
FAM = [["01","02","04","12","14","17","18","23","24","25","33"], ["03","06","07","08","20","21","22","26","28","30","32"], ["05","09","10","11","13","15","16","19","27","29","31"]]
E = [e for f in FAM for e in f]; ix = {e: i for i, e in enumerate(E)}
pairs = [(ix[x], ix[y]) for f in FAM for x, y in itertools.combinations(f, 2)]; A = np.array([p[0] for p in pairs]); B = np.array([p[1] for p in pairs])
CLASSES = [("33","30","31"), ("02","08","29"), ("23","20","16"), ("14","07","27"), ("25","06","19")]
REST = [[e for e in f if not any(e in c for c in CLASSES)] for f in FAM]
def pred(iso, groups=(["17", "18"],)):
    lab = np.zeros(33, int)
    for e in iso: lab[ix[e]] = 100 + ix[e]
    for k, g in enumerate(groups):
        for e in g: lab[ix[e]] = 200 + k
    return (lab[A] == lab[B]).astype(np.float32)
FP8 = ["05", "15", "22"]
SUBS = [("1803", pred(["14","23","07","28","16"], ()), .10), ("1882", pred(FP8, ()), .40), ("1967", pred(FP8), .60),
        ("2052", pred(FP8 + ["17","18"], ()), .60), ("2176", pred(FP8 + ["11"]), .60), ("2383", pred(FP8 + ["11","19"]), .60),
        ("2491", pred(FP8 + ["11","28","03"]), .60), ("2595", pred(FP8 + ["09","11"]), .60), ("2707", pred(FP8 + ["12","24"]), .60),
        ("2795", pred(FP8 + ["17","18","04","12","03","21","32","09","11"], ()), .30), ("2899", pred(FP8 + ["01"]), .60)]
PF = np.stack([s[1] for s in SUBS]); OBS = np.array([s[2] for s in SUBS], dtype=np.float32)
def kofn(N, n, k):     # N lignes, k vrais parmi n (k peut etre un tableau)
    r = rng.random((N, n)); k = np.broadcast_to(np.asarray(k), (N,))
    rk = np.argsort(np.argsort(r, 1), 1); return rk < k[:, None]
def draw_truth(N):
    """renvoie lab (N,33) : etiquette de modele ; 0 = base de la famille, autre = variante"""
    var = np.zeros((N, 33), bool); dup = None
    if hyp == "U": var = rng.random((N, 33)) < 0.5
    elif hyp in ("a4", "d4", "e5", "e6", "e3", "c4"):
        k = {"a4": 4, "d4": 4, "e5": 5, "e6": 6, "e3": 3, "c4": 4}[hyp]
        for f in range(3): var[:, f * 11:(f + 1) * 11] = kofn(N, 11, k)
    elif hyp == "b":
        cv = rng.random((N, 5)) < 0.5; nc = cv.sum(1)
        for c, cls in enumerate(CLASSES):
            for e in cls: var[:, ix[e]] = cv[:, c]
        k = 4 - nc; okk = k >= 0; k = np.maximum(k, 0)
        for f in range(3):
            sub = kofn(N, 6, k)
            for j, e in enumerate(REST[f]): var[:, ix[e]] = sub[:, j]
        var[~okk] = False; var[~okk, 0] = True; var[~okk, 1] = True   # tirage impossible : rendu incompatible plus bas
        bad = ~okk
    lab = np.where(var, 100 + np.arange(33)[None, :], 0)
    if hyp == "c4":     # deux des 4 variantes de chaque famille servent le meme modele
        for f in range(3):
            vi = np.argsort(~var[:, f * 11:(f + 1) * 11], 1, kind="stable")[:, :2] + f * 11
            r = np.arange(N); lab[r, vi[:, 1]] = lab[r, vi[:, 0]]
    if hyp == "b": lab[bad] = -1
    return lab
def draw_mask(N):
    if mask == "S":
        k = rng.integers(14, 23, N); pub = kofn(N, 33, k); return pub[:, A] & pub[:, B], pub
    return rng.random((N, len(A))) < 0.3, None
def scores(lab, m, PFm=PF):
    t = lab[:, A] == lab[:, B]
    Mt = (m & t).astype(np.float32); Mn = (m & ~t).astype(np.float32)
    tp = Mt @ PFm.T; fp = Mn @ PFm.T; fn = Mt.sum(1, keepdims=True) - tp; tn = Mn.sum(1, keepdims=True) - fp
    d = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.1 + 0.9 * np.maximum(0, np.where(d > 0, (tp * tn - fp * fn) / np.maximum(d, 1e-9), 0))
if __name__ == "__main__":
    N = 200_000; tot = 0; keep_lab, keep_pub, keep_m = [], [], []; t0 = time.time()
    while tot < M * 1e6:
        lab = draw_truth(N); m, pub = draw_mask(N)
        s = scores(lab, m); ok = (np.abs(s - OBS[None, :]) <= 0.025 + 1e-6).all(1)
        if hyp == "b": ok &= lab[:, 0] >= 0
        keep_lab.append(lab[ok]); keep_m.append(m[ok]); keep_pub.append(pub[ok] if pub is not None else np.zeros((ok.sum(), 33), bool)); tot += N
    L = np.concatenate(keep_lab); P = np.concatenate(keep_pub); Mm = np.concatenate(keep_m); n = len(L)
    print(f"hyp {hyp} masque {mask} : {n} tirages gardes sur {tot:.3g} -> vraisemblance {n / tot:.3e} ({time.time() - t0:.0f} s)")
    np.savez_compressed(f"E:/EAC-lab/t2/reel/oracle/rejet_{hyp}_{mask}_{seed}.npz", lab=L, pub=P, m=Mm, tot=tot)
    if n and mask == "S":
        isv = L != 0
        print("guichet P(public) P(variante) P(variante|public)")
        print(" ".join(f"{e}:{P[:, i].mean():.2f}/{isv[:, i].mean():.2f}/{(isv[:, i] & P[:, i]).mean() / max(P[:, i].mean(), 1e-9):.2f}" for i, e in enumerate(E)))
    elif n:
        isv = L != 0; print("P(variante) :", " ".join(f"{e}:{isv[:, i].mean():.2f}" for i, e in enumerate(E)))
