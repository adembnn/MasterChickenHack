"""Tests sans modele entre endpoints (aucune requete, CPU). Reponses aux memes questions, 50 premiers caracteres.
Ressemblance S(x, y) = moyenne sur les questions de la longueur du prefixe commun (plafonnee a 50) / 50.
(1) un echantillon : ressemblance moyenne de l'endpoint aux faux endpoints « copie » de meme temperature, comparee a celle
    d'un faux endpoint aux autres (z < 0 : repond moins comme la base qu'une copie).
(2) deux echantillons : S(i, j) comparee a S entre deux faux endpoints aux memes temperatures (z > 0 : i et j se ressemblent
    plus que deux copies de la base, donc partagent autre chose que la base).
Usage : python pairs.py <journal> <nul> <ep:T,ep:T,...> [hamming|prefixe]"""
import sys, json, collections, random, statistics, itertools
data, null, spec = sys.argv[1:4]
kind = sys.argv[4] if len(sys.argv) > 4 else "prefixe"
Tst = {x.split(":")[0]: float(x.split(":")[1]) for x in spec.split(",")}
eps = list(Tst)
def load(path, keep):
    by = collections.defaultdict(dict)
    for l in open(path, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            if r.get("type", "answer") == "answer" and r.get("response") and keep(r["endpoint"]) and r.get("probe_id") not in ("P000", "P001"):
                by[r["endpoint"]][(r["probe_id"], str(r.get("repeat", "0")))] = r["response"][:50]
    return by
tx = load(data, lambda e: e in eps); tn = load(null, lambda e: True)
def sim(u, v):
    if kind == "hamming":
        return sum(a == b for a, b in zip(u, v)) / 50
    n = 0
    for a, b in zip(u, v):
        if a != b: break
        n += 1
    return n / 50
def S(x, y):
    ks = [k for k in x if k in y]
    return sum(sim(x[k], y[k]) for k in ks) / max(1, len(ks))
bands = (0.7, 0.8, 0.9, 1.0)
bT = lambda T: min(bands, key=lambda b: abs(b - T))
byT = collections.defaultdict(list)
for f in tn: byT[float(f.split("-")[1])].append(f)
rng = random.Random(0)
# (1) un echantillon
print("(1) ressemblance a la base :", kind)
for e in eps:
    b = byT[bT(Tst[e])]
    me = statistics.mean(S(tx[e], tn[f]) for f in b)
    ref = [statistics.mean(S(tn[g], tn[f]) for f in b if f != g) for g in b]
    z = (me - statistics.mean(ref)) / (statistics.pstdev(ref) or 1e-9)
    p = (1 + sum(v <= me for v in ref)) / (1 + len(ref))
    print(f"  {e} T*={Tst[e]:.2f}  S={me:.3f}  copies={statistics.mean(ref):.3f}  z={z:+5.1f}  p(moins ressemblant)={p:.3f}")
# (2) deux echantillons
nullS = {}
for b1, b2 in itertools.combinations_with_replacement(bands, 2):
    vs = [S(tn[f], tn[g]) for f, g in (((rng.choice(byT[b1]), rng.choice(byT[b2]))) for _ in range(600)) if f != g]
    nullS[(b1, b2)] = vs
print("(2) z de S(i, j) contre deux copies ; * = p <= 0,01 (plus ressemblants que deux copies)")
print("      " + " ".join(f"{t[-2:]:>6}" for t in eps))
for i in eps:
    row = []
    for j in eps:
        if i == j: row.append("     -"); continue
        vs = nullS[tuple(sorted((bT(Tst[i]), bT(Tst[j]))))]
        s = S(tx[i], tx[j]); z = (s - statistics.mean(vs)) / (statistics.pstdev(vs) or 1e-9)
        p = (1 + sum(v >= s for v in vs)) / (1 + len(vs))
        row.append(f"{z:+5.1f}" + ("*" if p <= 0.01 else " "))
    print(f"  {i[-2:]}  " + " ".join(row))
