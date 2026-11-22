"""Grille de lecture de l'envoi H : score public observe -> verites probables -> meilleur fichier suivant. Aucune requete.
Verites (H) : Qwen-3B = 17, 18 + deux variantes parmi {01, 04, 12, 24, 25} ; Qwen-1.5B = 22 + trois de {03, 21, 26, 32} (une base) ;
Llama = {05, 09, 11, 15}, ou {05, 11, 15}, ou {05, 15}. Public = 30 % des paires au hasard, arrondi a 0,05."""
import csv, itertools, math, hashlib, collections, numpy as np
rng = np.random.default_rng(0)
F = {"qwen3b": ["01","02","04","12","14","17","18","23","24","25","33"], "qwen15b": ["03","06","07","08","20","21","22","26","28","30","32"], "llama1b": ["05","09","10","11","13","15","16","19","27","29","31"]}
E = [e for f in F.values() for e in f]; ix = {e: i for i, e in enumerate(E)}
P = [(ix[a], ix[b]) for f in F.values() for a, b in itertools.combinations(f, 2)]; A = np.array([p[0] for p in P]); B = np.array([p[1] for p in P])
ref = list(csv.DictReader(open("t2_fp8_loraQ3.csv", encoding="utf-8")))
baseL = {r["family"]: r["model"] for r in ref if r["model"].endswith("_base")}
def make(name, var, groups=()):
    rows = []
    for r in ref:
        e = r["endpoint"][-2:]; m = baseL[r["family"]]
        if e in var: m = f'{r["family"]}_var_ep{e}'
        for g in groups:
            if e in g: m = f'{r["family"]}_grp_ep{g[0]}'
        rows.append((r["endpoint"], r["family"], m))
    with open(name, "w", encoding="utf-8", newline="") as f:
        f.write("endpoint,family,model\n"); [f.write(",".join(x) + "\n") for x in rows]
    lab = np.array([hash(m) for _, _, m in rows]); order = [ix[r[0][-2:]] for r in rows]; L = np.zeros(33, dtype=np.int64); L[order] = lab
    return (L[A] == L[B]), hashlib.sha256(open(name, "rb").read()).hexdigest()
L3 = ["05", "09", "11", "15"]
FILES = collections.OrderedDict()
FILES["envoye : base 01 / 26"] = ("t2_H_base_ep01_ep26.csv", ["04","12","17","18","03","21","22","32"] + L3)
FILES["a : base Qwen-1.5B = ep-21"] = ("t2_H_base15b_ep21.csv", ["04","12","17","18","03","26","22","32"] + L3)
FILES["b : base Qwen-1.5B = ep-32"] = ("t2_H_base15b_ep32.csv", ["04","12","17","18","03","21","22","26"] + L3)
FILES["c : base Qwen-1.5B = ep-03"] = ("t2_H_base15b_ep03.csv", ["04","12","17","18","26","21","22","32"] + L3)
FILES["d : base Qwen-3B = ep-04"] = ("t2_H_base3b_ep04.csv", ["01","12","17","18","03","21","22","32"] + L3)
FILES["e : base Qwen-3B = ep-12"] = ("t2_H_base3b_ep12.csv", ["01","04","17","18","03","21","22","32"] + L3)
FILES["g : ep-24 isole a la place d'ep-12"] = ("t2_H_3b_ep24_pour_ep12.csv", ["04","24","17","18","03","21","22","32"] + L3)
FILES["h : ep-25 isole a la place d'ep-12"] = ("t2_H_3b_ep25_pour_ep12.csv", ["04","25","17","18","03","21","22","32"] + L3)
FILES["i : sans ep-09 (Llama)"] = ("t2_H_sans_ep09.csv", ["04","12","17","18","03","21","22","32","05","11","15"])
PRED, SHA = {}, {}
for k, (name, var) in FILES.items():
    if k.startswith("envoye"):
        rows = list(csv.DictReader(open(name, encoding="utf-8"))); L = np.zeros(33, dtype=np.int64)
        for r in rows: L[ix[r["endpoint"][-2:]]] = hash(r["model"])
        PRED[k] = (L[A] == L[B]); SHA[k] = hashlib.sha256(open(name, "rb").read()).hexdigest()
    else:
        PRED[k], SHA[k] = make(name, set(var))
rows = list(csv.DictReader(open("t2_fp8_loraQ3.csv", encoding="utf-8"))); L = np.zeros(33, dtype=np.int64)
for r in rows: L[ix[r["endpoint"][-2:]]] = hash(r["model"])
PRED["1967 (retenu)"] = (L[A] == L[B])
PRED["f : repli 1967 + {09, 11}"], SHA["f : repli 1967 + {09, 11}"] = make("t2_repli_1967_plus_ep09_ep11.csv", {"05","15","22","09","11"}, groups=(["17","18"],))
# verites et a priori
W1 = {"04": .60, "12": .60, "01": .15, "24": .30, "25": .25}
TR = []
for v1 in itertools.combinations(W1, 2):
    p1 = np.prod([W1[e] if e in v1 else 1 - W1[e] for e in W1])
    for b2, p2 in (("26", .34), ("32", .30), ("21", .28), ("03", .08)):
        for v3, p3 in ((("05","09","11","15"), .75), (("05","11","15"), .15), (("05","15"), .10)):
            var = {"17","18","22"} | set(v1) | ({"03","21","26","32"} - {b2}) | set(v3)
            t = np.array([(E[a] not in var) and (E[b] not in var) for a, b in P]); TR.append((p1 * p2 * p3, v1, b2, v3, t))
Z = sum(t[0] for t in TR); TR = [(p / Z, *r) for p, *r in TR]
def mcc(t, p, m):
    tp = (t & p & m).sum(-1).astype(float); fp = (~t & p & m).sum(-1); fn = (t & ~p & m).sum(-1); tn = (~t & ~p & m).sum(-1)
    d = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)); return 0.1 + 0.9 * np.maximum(0, np.where(d > 0, (tp * tn - fp * fn) / np.maximum(d, 1e-9), 0))
MASK = rng.random((3000, len(P))) < 0.30; FULL = np.ones((1, len(P)), bool)
k0 = "envoye : base 01 / 26"
print("== 1. Score exact du fichier envoye sur toutes les paires, et public attendu (30 % des paires, arrondi a 0,05) selon la verite")
seen = collections.OrderedDict()
for p, v1, b2, v3, t in sorted(TR, key=lambda r: -float(mcc(r[4], PRED[k0], FULL)[0])):
    s = float(mcc(t, PRED[k0], FULL)[0]); pub = np.round(mcc(t[None, :], PRED[k0][None, :], MASK) * 20) / 20
    seen.setdefault(round(s, 3), []).append((p, v1, b2, v3, np.percentile(pub, 10), np.median(pub), np.percentile(pub, 90)))
for s, L_ in seen.items():
    pp = sum(x[0] for x in L_); ex = max(L_, key=lambda x: x[0])
    print(f"  exact {s:.3f} | public 10-50-90 % : {min(x[4] for x in L_):.2f}-{np.median([x[5] for x in L_]):.2f}-{max(x[6] for x in L_):.2f} | a priori {pp:.2f} | {len(L_)} verites, la plus probable : Qwen-3B +{'+'.join(ex[1])}, base Qwen-1.5B ep-{ex[2]}, Llama {'+'.join(ex[3])}")
print("== 2. Selon le score public observe : verites les plus probables et meilleur fichier suivant (score exact attendu)")
pubs = {i: np.round(mcc(t[None, :], PRED[k0][None, :], MASK) * 20) / 20 for i, (_, _, _, _, t) in enumerate(TR)}
for obs in ("pas de New best (<= 0,60)", 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00):
    like = np.array([(pubs[i] <= 0.60).mean() if isinstance(obs, str) else (np.abs(pubs[i] - obs) < 1e-6).mean() for i in range(len(TR))])
    post = like * np.array([t[0] for t in TR])
    if post.sum() < 1e-9: print(f"  observe {obs} : impossible sous ces verites"); continue
    post /= post.sum(); top = np.argsort(-post)[:2]
    exp = {k: float(sum(post[i] * mcc(TR[i][4], PRED[k], FULL)[0] for i in range(len(TR)))) for k in PRED}
    best = sorted(exp, key=exp.get, reverse=True)[:3]
    desc = " ; ".join(f"Qwen-3B +{'+'.join(TR[i][1])}, base 1.5B ep-{TR[i][2]}, Llama {len(TR[i][3])} var. ({post[i]:.2f})" for i in top)
    print(f"  observe {obs} (P = {float((like * np.array([t[0] for t in TR])).sum()):.2f}) : {desc}")
    print("      fichier suivant : " + " | ".join(f"{k} -> {exp[k]:.3f}" for k in best))
print("== 3. Fichiers deposes (sha256)")
for k in FILES:
    if not k.startswith("envoye"): print(f"  {FILES[k][0]:36s} {SHA[k]}  [{k}]")
print(f"  {'t2_repli_1967_plus_ep09_ep11.csv':36s} {SHA['f : repli 1967 + {09, 11}']}  [f : repli]")
