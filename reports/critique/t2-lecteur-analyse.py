"""Analyse du lecteur a deguisement ajuste (fitdisg.py) + lecteur existant (gl_reader.py), etalonnee sur le reel.
Session t2-lecteur, aucune requete.
Pour chaque guichet : meilleur deguisement (contexte, penalite, T, troncature), gain du deguisement, ecarts residuels apres
ajustement (tests de calibration queue / milieu / tete, LLR des quantifications), z calibres sur TOUS les faux endpoints
(copies simples + copies deguisees : la loi nulle elargie). Puis AUC copies confirmees / variantes confirmees, P(variante).
Usage : analyse.py fd_F1.json fd_F2.json fd_F3.json [--gl gl50_F1.json,...] [--csv candidat.csv]"""
import sys, json, math, statistics, argparse, random, itertools

ap = argparse.ArgumentParser()
ap.add_argument("fd", nargs="+")
ap.add_argument("--gl", default="")
ap.add_argument("--glan", default="", help="sorties *_analyse.json de gl_analyse.py")
ap.add_argument("--out", default="lecteur_stats.json")
a = ap.parse_args()
COP = ["ep-04", "ep-21", "ep-32", "ep-07", "ep-14", "ep-23"]
VAR = ["ep-17", "ep-18", "ep-22", "ep-05", "ep-15"]


def pmax(v, TS):
    i = max(range(len(v)), key=lambda k: v[k])
    if 0 < i < len(v) - 1:
        x, y, z = v[i - 1], v[i], v[i + 1]
        d = x - 2 * y + z
        if d < 0:
            s = 0.5 * (x - z) / d
            return y - 0.25 * (x - z) * s, TS[i] + s * (TS[i + 1] - TS[i])
    return v[i], TS[i]


def robz(vals, pool):
    vs = [vals[f] for f in pool if f in vals]
    med = statistics.median(vs)
    mad = statistics.median(abs(v - med) for v in vs) * 1.4826 or 1e-9
    return {e: (vals[e] - med) / mad for e in vals}


ALLROWS = {}
for path in a.fd:
    R = json.load(open(path))
    TS, TA, THS, CFG, n = R["TS"], R.get("TA", R["TS"]), R["THS"], R["CFG"], R["ntok"]
    eps = R["eps"]
    ends = list(R["grid"]) if R["grid"] else list(R["ctxA"]) if R["ctxA"] else list(R["base"])
    fk = [e for e in ends if e.startswith("fk-")]
    st = {}
    for e in ends:
        o = {}
        o["ctx"] = R["best_ctx"][e]
        if R["ctxA"]:
            l_none = pmax(R["ctxA"][e]["none"], TA)[0]
            l_ctx = pmax(R["ctxA"][e][o["ctx"]], TA)[0]
            o["gain_ctx"] = (l_ctx - l_none) / n[e]
        else:
            o["gain_ctx"] = 0.0
        if R["grid"]:
            G = R["grid"][e]
            best = max(((G[i][j][k][0], i, j, k) for i in range(len(THS)) for j in range(len(TS)) for k in range(len(CFG))))
            _, i, j, k = best
            plain = max(G[0][jj][0][0] for jj in range(len(TS)))
            o["disg"] = f"rep{THS[i]}/T{TS[j]}/{CFG[k]}"
            o["T"] = TS[j]
            o["gain_dec"] = (best[0] - plain) / n[e]
            s = G[i][j][k]
            N = s[9]
            o["nll"] = -s[0] / N
            o["z_ll"] = (s[0] + s[1]) / math.sqrt(max(s[2], 1e-9))
            o["z_tail"] = (s[3] - s[4]) / math.sqrt(max(s[10], 1e-9))
            o["z_mid"] = (s[5] - s[6]) / math.sqrt(max(s[6] * (1 - s[6] / N), 1e-9))
            o["z_top"] = (s[7] - s[8]) / math.sqrt(max(s[8] * (1 - s[8] / N), 1e-9))
        if R["base"]:
            lb = pmax(R["base"][e], TA)[0]
            for sc, Q in R["quant"].items():
                if e in Q:
                    o["llr_" + sc] = (pmax(Q[e], TA)[0] - lb) / n[e]
        st[e] = o
    keys = sorted({k for o in st.values() for k in o if isinstance(o[k], float) and k not in ("T",)})
    Z = {}
    for k in keys:
        vals = {e: st[e][k] for e in st if k in st[e]}
        Z[k] = robz(vals, fk)
    for e in st:
        for k in keys:
            if e in Z[k]:
                st[e]["Z_" + k] = Z[k][e]
                zf = [Z[k][f] for f in fk if f in Z[k]]
                st[e]["p_" + k] = (1 + sum(abs(v) >= abs(Z[k][e]) for v in zf)) / (1 + len(zf))
    # tableau des faux endpoints par deguisement : l'ecart residuel disparait-il ?
    print(f"\n== {path} : {len(eps)} guichets, {len(fk)} faux endpoints")
    groups = {}
    for f in fk:
        g = f.rsplit("-", 1)[0][3:]
        groups.setdefault(g, []).append(f)
    show = ["Z_z_tail", "Z_z_mid", "Z_z_top"] + [k for k in ["Z_llr_nf4_b64", "Z_llr_int8_ch", "Z_llr_fp8_ch", "Z_llr_int4_g128", "Z_llr_fp16", "Z_llr_fp8_t"] if any(k in st[e] for e in st)]
    print("faux par deguisement (mediane des z) | " + " ".join(s[2:] for s in show) + " | deguisement retrouve (le plus frequent)")
    for g, fs in sorted(groups.items()):
        meds = [statistics.median(st[f][s] for f in fs if s in st[f]) if any(s in st[f] for f in fs) else float("nan") for s in show]
        dg = statistics.mode([st[f].get("disg", "-") + "|" + st[f]["ctx"] for f in fs])
        print(f"  {g:14s} " + " ".join(f"{m:+6.1f}" for m in meds) + f"  {dg}")
    print("guichet  contexte  deguisement            gain_ctx gain_dec | " + " ".join(s[2:] for s in show))
    for e in eps:
        o = st[e]
        print(f"{e}  {o['ctx']:8s} {o.get('disg', '-'):22s} {o['gain_ctx']:+.4f} {o.get('gain_dec', 0):+.4f} | " + " ".join(f"{o.get(s, float('nan')):+6.1f}" for s in show)
              + ("  <- copie confirmee" if e in COP else "  <- variante confirmee" if e in VAR else ""))
    ALLROWS.update({e: st[e] for e in eps})

# lecteur existant : z fp8, M des directions
for p in [x for x in a.glan.split(",") if x]:
    G = json.load(open(p))["out"]
    for e, o in G.items():
        if e in ALLROWS:
            for k in ("z_fp8", "M"):
                if k in o:
                    ALLROWS[e]["Z_gl_" + k] = o[k]


def auc(stat, sign=1):
    xs = [sign * ALLROWS[e][stat] for e in VAR if e in ALLROWS and stat in ALLROWS[e]]
    ys = [sign * ALLROWS[e][stat] for e in COP if e in ALLROWS and stat in ALLROWS[e]]
    if len(xs) < 3 or len(ys) < 3:
        return None
    A = sum((x > y) + 0.5 * (x == y) for x in xs for y in ys) / (len(xs) * len(ys))
    rng = random.Random(0)
    bs = []
    for _ in range(2000):
        xb = [rng.choice(xs) for _ in xs]
        yb = [rng.choice(ys) for _ in ys]
        bs.append(sum((x > y) + 0.5 * (x == y) for x in xb for y in yb) / (len(xb) * len(yb)))
    bs.sort()
    return A, bs[50], bs[1949]


print("\n== Separation des 6 copies confirmees et des 5 variantes confirmees (AUC ; IC 95 % par bootstrap)")
stats = sorted({k for o in ALLROWS.values() for k in o if k.startswith("Z_")})
res = []
for s in stats:
    for sign, lab in ((1, "+"), (-1, "-")):
        r = auc(s, sign)
        if r:
            res.append((r[0], r[1], r[2], s, lab))
    # valeur absolue
    xs = {e: abs(ALLROWS[e][s]) for e in ALLROWS if s in ALLROWS[e]}
    for e in xs:
        ALLROWS[e]["A" + s] = xs[e]
    r = auc("A" + s, 1)
    if r:
        res.append((r[0], r[1], r[2], s, "|.|"))
for A, lo, hi, s, lab in sorted(res, reverse=True)[:25]:
    print(f"  {s:22s} {lab:3s} AUC {A:.2f}  [{lo:.2f} ; {hi:.2f}]  variantes : " + " ".join(f"{ALLROWS.get(e, {}).get(s, float("nan")):+.1f}" for e in VAR)
          + "  copies : " + " ".join(f"{ALLROWS.get(e, {}).get(s, float("nan")):+.1f}" for e in COP))
json.dump(ALLROWS, open(a.out, "w"), indent=1)
