"""Analyse des sorties de gl_reader.py : p-valeurs par endpoint (rang parmi les faux endpoints « copie » du jumeau local),
matrices entre endpoints, regle de decision pre-enregistree, CSV.
Usage : python gl_analyse.py gl_<tag>.json <journal reel> <journal nul> <famille> [--csv sortie.csv]
Regle (figee avant lecture des resultats) :
  variante fp8      si p_fp8 <= 0,01 (meilleur des schemas fp8, p corrigee du nombre de schemas) ;
  variante LoRA     si p_M  <= 0,01, ou M = plus grand z, sur les directions apprises sur les AUTRES endpoints, du gain de
                     vraisemblance profilee (max sur la dose s) ; z et p calibres sur les faux endpoints de meme temperature ;
  copie deguisee    si seul le lecteur appris sur l'endpoint lui-meme sort (p_self <= 0,05) : rattachee a la base ;
  copie             sinon.
"""
import sys, json, math, random, collections, itertools, statistics, argparse
ap = argparse.ArgumentParser()
ap.add_argument("res"); ap.add_argument("data"); ap.add_argument("null"); ap.add_argument("fam")
ap.add_argument("--csv", default=None); ap.add_argument("--alpha", type=float, default=0.01)
a = ap.parse_args()
R = json.load(open(a.res)); TS = R["TS"]; SC = [str(s) for s in R["scales"]]; eps = R["eps"]; n = R["ntok"]
dT = TS[1] - TS[0]


def pmax(v):
    i = max(range(len(v)), key=lambda k: v[k])
    if 0 < i < len(v) - 1:
        x, y, z = v[i - 1], v[i], v[i + 1]
        d = x - 2 * y + z
        if d < 0:
            s = 0.5 * (x - z) / d
            return y - 0.25 * (x - z) * s, TS[i] + s * dT
    return v[i], TS[i]


L0 = {e: pmax(R["base"][e]) for e in R["base"]}
fk = sorted(e for e in R["base"] if e.startswith("fk-"))
Tf = lambda e: float(e.split("-")[1])
Tst = {e: (L0[e][1] if not e.startswith("fk-") else Tf(e)) for e in L0}


def band(x, pool):
    b = [f for f in pool if abs(Tf(f) - Tst[x]) <= 0.1001 and f != x]
    return b if len(b) >= 8 else [f for f in pool if f != x]


def zp(val, x, pool):
    """z robuste de val[x] parmi les faux endpoints de temperature voisine."""
    b = band(x, pool)
    vs = [val[f] for f in b]
    med = statistics.median(vs)
    mad = statistics.median(abs(v - med) for v in vs) * 1.4826 or 1e-9
    return (val[x] - med) / mad


def zall(val, pool):
    """z de tous les endpoints reels et faux ; p-valeur = rang du z parmi TOUS les faux endpoints (z comparables entre temperatures)."""
    z = {x: zp(val, x, pool) for x in list(eps) + list(pool) if x in val}
    zf = [z[f] for f in pool]
    p = {x: (1 + sum(v >= z[x] for v in zf)) / (1 + len(zf)) for x in z}
    return z, p


out = {e: {} for e in eps}
# --- fp8
fp = {}
for sc, P in R.get("fp8", {}).items():
    fp[sc] = {e: (pmax(P[e])[0] - L0[e][0]) / n[e] for e in P}
ZF = {sc: zall(fp[sc], fk) for sc in fp}
for e in eps:
    best = min(((ZF[sc][1][e], -ZF[sc][0][e], sc) for sc in fp), default=(1.0, 0.0, "-"))
    out[e]["p_fp8"] = min(1.0, best[0] * max(1, len(fp)))
    out[e]["z_fp8"] = -best[1]
    out[e]["fp8_sc"] = best[2]
# --- directions apprises sur les autres endpoints
dirs = [d for d in eps if d in R.get("dir", {})]
if dirs:
    SC = sorted(R["dir"][dirs[0]].keys(), key=float)
lam, shat = {}, {}
for d in dirs:
    lam[d], shat[d] = {}, {}
    for t in R["dir"][d][SC[0]]:
        vals = [(pmax(R["dir"][d][s][t])[0] - L0[t][0]) / n[t] for s in SC]
        k = max(range(len(SC)), key=lambda i: vals[i])
        lam[d][t] = vals[k]
        shat[d][t] = float(SC[k]) if vals[k] > 0 else 0.0
fkd = sorted(set.intersection(*[set(f for f in lam[d] if f.startswith("fk-")) for d in dirs])) if dirs else []
Z = {d: {} for d in dirs}
for d in dirs:
    for t in list(eps) + fkd:
        if t != d and t in lam[d]:
            Z[d][t] = zp(lam[d], t, fkd)
Mf = {f: max(Z[d][f] for d in dirs) for f in fkd}
mfs = sorted(Mf.values())
for e in eps:
    cand = [(Z[d][e], d) for d in dirs if d != e and e in Z[d]]
    if not cand:
        continue
    m, d = max(cand)
    out[e].update(M=m, M_dir=d, M_s=shat[d][e], p_M=(1 + sum(v >= m for v in mfs)) / (1 + len(mfs)))
if mfs:
    print(f"loi nulle du M (max sur {len(dirs)} directions) : mediane {statistics.median(mfs):+.1f}, 95 % {mfs[int(0.95 * len(mfs))]:+.1f}, max {mfs[-1]:+.1f}")
# --- lecteur appris sur l'endpoint lui-meme (2 plis)
sl = {e: max((pmax(R["self"][e][s])[0] - L0[e][0]) / n[e] for s in SC) for e in R.get("self", {})}
fks = [f for f in sl if f.startswith("fk-")]
for e in eps:
    if e in sl and fks:
        vs = [sl[f] for f in fks]
        out[e]["self"] = sl[e]
        out[e]["p_self"] = (1 + sum(v >= sl[e] for v in vs)) / (1 + len(vs))
print(f"== {a.fam} : {len(fk)} faux endpoints (fp8), {len(fkd)} (directions), {len(fks)} (lecteur propre)")
print("endpoint  T*    nll/tok | fp8 z      p     | propre gain   p    | autres: M(z)  p     direction dose")
for e in eps:
    o = out[e]
    print(f"{e}  {L0[e][1]:.2f}  {-L0[e][0] / n[e]:.3f}  | {o.get('z_fp8', 0):+6.1f} {o.get('p_fp8', 1):.3f} {o.get('fp8_sc', '-')[:4]} | "
          f"{o.get('self', float('nan')):+.4f} {o.get('p_self', float('nan')):.3f} | {o.get('M', float('nan')):+6.1f} {o.get('p_M', float('nan')):.3f}  {o.get('M_dir', '-')}  {o.get('M_s', 0)}")
if dirs:
    print("\nmatrice z : ligne = direction apprise sur d, colonne = endpoint lu t (z du gain parmi les faux endpoints ; dose entre parentheses si gain > 0)")
    print("        " + " ".join(f"{t[-2:]:>9}" for t in eps))
    for d in dirs:
        print(f"dir {d[-2:]}  " + " ".join((f"{Z[d][t]:+5.1f}({shat[d][t]:.1f})" if t in Z[d] else "     -   ") for t in eps))
# --- test a deux echantillons sans modele : ressemblance de Hamming sur les 50 premiers caracteres, memes questions
def loadtxt(path, keep):
    by = collections.defaultdict(dict)
    for l in open(path, encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            if r.get("type", "answer") == "answer" and r.get("response") and keep(r["endpoint"]) and r.get("probe_id") not in ("P000", "P001"):
                by[r["endpoint"]][(r["probe_id"], str(r.get("repeat", "0")))] = r["response"][:50]
    return by


tx = loadtxt(a.data, lambda e: e in eps)
tn = loadtxt(a.null, lambda e: True)


def ham(x, y):
    ks = [k for k in x if k in y]
    return sum(sum(c1 == c2 for c1, c2 in zip(x[k], y[k])) / 50 for k in ks) / max(1, len(ks))


rng = random.Random(0)
bandT = lambda T: min((0.7, 0.8, 0.9, 1.0), key=lambda b: abs(b - T))
byT = collections.defaultdict(list)
for f in tn:
    byT[round(Tf(f), 2)].append(f)
nullH = {}
for b1, b2 in itertools.combinations_with_replacement((0.7, 0.8, 0.9, 1.0), 2):
    vs = []
    for _ in range(400):
        f, g = rng.choice(byT[b1]), rng.choice(byT[b2])
        if f != g:
            vs.append(ham(tn[f], tn[g]))
    nullH[(b1, b2)] = (statistics.mean(vs), statistics.pstdev(vs) or 1e-9)
print("\nmatrice Hamming (z de la ressemblance des 50 premiers caracteres, par rapport a deux copies du jumeau aux memes temperatures ; z > 0 = plus ressemblants que deux copies)")
print("        " + " ".join(f"{t[-2:]:>5}" for t in eps))
for i in eps:
    row = []
    for j in eps:
        if i == j:
            row.append("    -")
            continue
        k = tuple(sorted((bandT(Tst[i]), bandT(Tst[j]))))
        row.append(f"{(ham(tx[i], tx[j]) - nullH[k][0]) / nullH[k][1]:+5.1f}")
    print(f"    {i[-2:]}  " + " ".join(row))
# --- decision
lab = {}
for e in eps:
    o = out[e]
    if o.get("p_fp8", 1) <= a.alpha:
        lab[e] = ("variante fp8", e)
    elif o.get("p_M", 1) <= a.alpha:
        lab[e] = ("variante LoRA", e)
    elif o.get("p_self", 1) <= 0.05:
        lab[e] = ("copie deguisee", "base")
    else:
        lab[e] = ("copie", "base")
print("\ndecision :", " ; ".join(f"{e[-2:]}={lab[e][0]}" for e in eps))
json.dump({"out": out, "lab": lab}, open(a.res.replace(".json", "_analyse.json"), "w"), indent=1)
if a.csv:
    with open(a.csv, "w", encoding="utf-8", newline="") as f:
        f.write("endpoint,family,model\n")
        for e in eps:
            f.write(f"{e},{a.fam},{a.fam}_{lab[e][1]}\n")
