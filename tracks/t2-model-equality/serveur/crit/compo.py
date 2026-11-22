"""Point 1 : structures cachees compatibles avec les envois deja notes (aucune requete).
Hypothese de composition : dans chaque famille de 11, k endpoints sont des variantes de poids (chacune son modele), les autres
sont des copies de la base. Public = 30 % des paires tire au hasard (le meme masque pour tous les envois), score arrondi a 0,05.
Echantillonnage (structure, masque) puis rejet sur les 6 scores observes. Usage : compo.py <k ou 'mix'> <uniforme|indices> [N]"""
import sys, itertools, json, numpy as np
rng = np.random.default_rng(int(sys.argv[4]) if len(sys.argv) > 4 else 0)
FAM = [["01","02","04","12","14","17","18","23","24","25","33"], ["03","06","07","08","20","21","22","26","28","30","32"], ["05","09","10","11","13","15","16","19","27","29","31"]]
E = [e for f in FAM for e in f]; ix = {e: i for i, e in enumerate(E)}
pairs = [(ix[a], ix[b]) for f in FAM for a, b in itertools.combinations(f, 2)]
A = np.array([p[0] for p in pairs]); B = np.array([p[1] for p in pairs])
def pred(iso, groups=()):
    lab = np.full(33, -1)
    for e in iso: lab[ix[e]] = 100 + ix[e]
    for k, g in enumerate(groups):
        for e in g: lab[ix[e]] = 200 + k
    return lab[A] == lab[B]
FP8 = ["05", "15", "22"]
SUBS = [("1803", pred(["14","23","07","28","16"]), 0.10), ("1882", pred(FP8), 0.40), ("1967", pred(FP8, (["17","18"],)), 0.60),
        ("2052", pred(FP8 + ["17","18"]), 0.60), ("2176", pred(FP8 + ["11"], (["17","18"],)), 0.60), ("2383", pred(FP8 + ["11","19"], (["17","18"],)), 0.60), ("2491", pred(FP8 + ["28","03"], (["17","18"],)), 0.60), ("2595", pred(FP8 + ["09","11"], (["17","18"],)), 0.60)]
# a priori tire de mes mesures locales : probabilite qu'un endpoint soit une variante de poids
PRI = {e: 0.25 for e in E}
PRI.update({"02": .03, "23": .03, "08": .03, "20": .03, "29": .03, "16": .03, "33": .05, "30": .05, "31": .05,
            "22": .95, "15": .95, "05": .95, "17": .5, "18": .5, "11": .6, "09": .5, "28": .5, "03": .45, "12": .35})
mode, prior = sys.argv[1], sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 4_000_000
logw = np.array([np.log(PRI[e] / (1 - PRI[e])) if prior == "indices" else 0.0 for e in E])
def mcc(ts, ps, m):
    tp = (ts & ps & m).sum(1).astype(float); fp = (~ts & ps & m).sum(1); fn = (ts & ~ps & m).sum(1); tn = (~ts & ~ps & m).sum(1)
    d = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)); return 0.1 + 0.9 * np.maximum(0, np.where(d > 0, (tp * tn - fp * fn) / np.maximum(d, 1e-9), 0))
keepV, keepM, keepP = [], [], []
MASK = sys.argv[5] if len(sys.argv) > 5 else "paires"
CH = 400_000
for c in range(0, N, CH):
    V = np.zeros((CH, 33), bool)
    for fi in range(3):
        ks = rng.integers(3, 7, CH) if mode == "mix" else np.full(CH, int(mode))
        g = logw[fi * 11:(fi + 1) * 11] + rng.gumbel(size=(CH, 11))
        rank = np.argsort(np.argsort(-g, 1), 1); V[:, fi * 11:(fi + 1) * 11] = rank < ks[:, None]
    ts = ~V[:, A] & ~V[:, B]                      # meme modele = les deux dans la base
    if MASK == "paires":
        m = rng.random((CH, len(pairs))) < 0.30
    else:                                          # masque par endpoints : paires dont les deux endpoints sont publics
        npub = int(MASK)
        pub = np.argsort(rng.random((CH, 33)), 1) < npub
        m = pub[:, A] & pub[:, B]
        keepP.append(None)
    ok = np.ones(CH, bool)
    for name, ps, obs in SUBS:
        s = mcc(ts, ps[None, :], m); ok &= (s >= obs - 0.025) & (s < obs + 0.025)
    keepV.append(V[ok]); keepM.append(m[ok])
V = np.concatenate(keepV); M = np.concatenate(keepM)
print(f"== k={mode}, a priori {prior}, masque {MASK} : {len(V)} structures compatibles avec les {len(SUBS)} scores sur {N} tirages ({len(V) / N:.2e})")
if len(V) < 30: sys.exit()
vis = np.zeros((len(M), 33))
np.add.at(vis, (np.repeat(np.arange(len(M)), M.sum(1)), A[np.nonzero(M)[1]]), 1); np.add.at(vis, (np.repeat(np.arange(len(M)), M.sum(1)), B[np.nonzero(M)[1]]), 1)
print("  P(au moins une paire publique) :", " ".join(f"{e}={(vis[:, ix[e]] > 0).mean():.2f}" for e in E))
print("  paires publiques internes par endpoint (moyenne) :", " ".join(f"{e}={vis[:, ix[e]].mean():.1f}" for e in E))
nv = V.reshape(-1, 3, 11).sum(2)
if mode == "mix": print("  nombre de variantes par famille (Qwen-3B, Qwen-1.5B, Llama) :", " | ".join(" ".join(f"{k}:{(nv[:, f] == k).mean():.2f}" for k in range(3, 7)) for f in range(3)))
print("  P(variante) :", " ".join(f"{e}={V[:, ix[e]].mean():.2f}" for e in E))
import collections
for fi, name in enumerate(("Qwen-3B", "Qwen-1.5B", "Llama")):
    c = collections.Counter(tuple(e for e in FAM[fi] if v[ix[e]]) for v in V)
    print(f"  {name}, jeux de variantes les plus frequents :", " ; ".join(f"{'+'.join(k)} {n / len(V):.3f}" for k, n in c.most_common(6)))
ts = ~V[:, A] & ~V[:, B]; full = np.ones_like(ts)
def report(name, ps):
    f = mcc(ts, ps[None, :], full); p = np.round(mcc(ts, ps[None, :], M) * 20) / 20
    d = collections.Counter(np.round(p, 2)); H = -sum(n / len(p) * np.log2(n / len(p)) for n in d.values())
    print(f"  {name:<34} final attendu {f.mean():.3f} | public : " + " ".join(f"{k:.2f}:{n / len(p):.2f}" for k, n in sorted(d.items())) + f" | information {H:.2f} bit")
j = collections.Counter((bool(v[ix["28"]]), bool(v[ix["03"]])) for v in V)
print("  ep-28 / ep-03 (variante ?) :", " ; ".join(f"28={'V' if a else 'c'} 03={'V' if b else 'c'} : {n / len(V):.2f}" for (a, b), n in sorted(j.items())))
j = collections.Counter((bool(v[ix["11"]]), bool(v[ix["09"]])) for v in V)
print("  ep-11 / ep-09 :", " ; ".join(f"11={'V' if a else 'c'} 09={'V' if b else 'c'} : {n / len(V):.2f}" for (a, b), n in sorted(j.items())))
print("  envois candidats (base = fichier 2176 sans ep-11, soit fp8 + groupe 17,18) :")
base_iso, grp = FP8, (["17", "18"],)
report("fichier sur (1967)", pred(base_iso, grp)); report("2176 (+ep-11)", pred(base_iso + ["11"], grp))
for add in (["09", "11", "19", "03", "28"], ["12", "24"], ["01", "04", "25"], ["06", "21", "26", "32"], ["28"], ["03"], ["11", "09"], ["09"], ["12"], ["24"], ["01"], ["12", "24"], ["01", "25"], ["10", "13"], ["06", "21"], ["26", "32"], ["04", "25"], ["07", "14", "27"], ["11", "09", "28", "03"], ["12", "24", "01"], ["06", "21", "26", "32"], ["01", "04", "12", "24", "25"]):
    report("+ " + ",".join(add) + " isoles", pred(base_iso + add, grp))
report("17 et 18 separes", pred(base_iso + ["17", "18"]))
