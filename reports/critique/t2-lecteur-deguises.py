"""Classement des guichets deguises (session t2-lecteur, aucune requete) : le deguisement explique-t-il tout l'ecart a la base ?
Entrees : sorties de gl_analyse.py (directions, z fp8 contre copies simples) et de fitdisg.py --stages C sur les copies
DEGUISEES (dq_F*.json : LLR fp8 par tenseur, int8 par ligne, fp16, contexte sans prompt, comme pour les guichets reels).
Critere fixe avant lecture des copies deguisees (22/11, 6 h 25) :
  residu_q = max des z de quantification d'un guichet calcules contre les copies deguisees (loi nulle elargie) ;
  M = z de la meilleure direction LoRA apprise sur un autre guichet (loi nulle simple, gl_analyse) ;
  variante probable si residu_q >= 3 ou (M >= 3 et p_M <= 0,05) ; sinon copie deguisee. Classement par max(residu_q, M).
Comparaison par type de deguisement : meme statistique pour le guichet de meme deguisement dans les deux autres familles.
Usage : deguises.py --gl g2_F1_snap_analyse.json,... --dq dq_F1.json,dq_F2.json,dq_F3.json"""
import json, statistics, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--gl", required=True)
ap.add_argument("--dq", required=True)
a = ap.parse_args()
TYPE = {  # deguisement visible (sondes P000/P001, T lue, troncature mesuree)
    "T=1,0": ["ep-33", "ep-30", "ep-31"],
    "troncature": ["ep-14", "ep-07", "ep-27"],
    "persona": ["ep-23", "ep-08", "ep-16"],
    "prompt fixe": ["ep-02", "ep-20", "ep-29"],
    "sans signe (filigrane ?)": ["ep-25", "ep-06", "ep-19"],
}
G = {}
for p in a.gl.split(","):
    G.update(json.load(open(p))["out"])


def pmax(v, TS):
    i = max(range(len(v)), key=lambda k: v[k])
    if 0 < i < len(v) - 1:
        x, y, z = v[i - 1], v[i], v[i + 1]
        d = x - 2 * y + z
        if d < 0:
            s = 0.5 * (x - z) / d
            return y - 0.25 * (x - z) * s
    return v[i]


Q = {}
for p in a.dq.split(","):
    R = json.load(open(p))
    TA, n = R["TA"], R["ntok"]
    ends = list(R["base"])
    fk = [e for e in ends if e.startswith("fk-")]
    for sc, P in R["quant"].items():
        llr = {e: (pmax(P[e], TA) - pmax(R["base"][e], TA)) / n[e] for e in ends}
        vs = [llr[f] for f in fk]
        med = statistics.median(vs)
        mad = statistics.median(abs(v - med) for v in vs) * 1.4826 or 1e-9
        for e in R["eps"]:
            Q.setdefault(e, {})[sc] = (llr[e] - med) / mad
        # pire copie deguisee, par deguisement : jusqu'ou un deguisement seul fait monter le z
        byd = {}
        for f in fk:
            byd.setdefault(f.rsplit("-", 1)[0][3:], []).append((llr[f] - med) / mad)
        Q.setdefault("_max_" + sc, {}).update({d: max(v) for d, v in byd.items()})
print("z maximal atteint par une copie deguisee, par deguisement (loi nulle elargie) :")
for sc in ("fp8_t", "int8_ch", "fp16"):
    m = Q.get("_max_" + sc, {})
    print(f"  {sc:8s} " + " ".join(f"{d}:{v:+.1f}" for d, v in sorted(m.items())))
rows = []
for t, es in TYPE.items():
    print(f"\n{t}")
    for e in es:
        q = Q.get(e, {})
        g = G.get(e, {})
        res = max(q.values()) if q else float("nan")
        M, pM = g.get("M", float("nan")), g.get("p_M", 1.0)
        verdict = "variante probable" if (res >= 3 or (M >= 3 and pM <= 0.05)) else "copie deguisee"
        rows.append((max(res, M), e, t, verdict))
        print(f"  {e}  z fp8 (copies simples) {g.get('z_fp8', float('nan')):+5.1f} | contre copies deguisees : fp8 {q.get('fp8_t', float('nan')):+5.1f} "
              f"int8 {q.get('int8_ch', float('nan')):+5.1f} fp16 {q.get('fp16', float('nan')):+5.1f} | direction M {M:+5.1f} (p {pM:.3f}, {g.get('M_dir', '-')}) -> {verdict}")
print("\nclassement (du plus au moins suspect) :")
for s, e, t, v in sorted(rows, reverse=True):
    print(f"  {e}  {s:+5.1f}  {t:24s} {v}")
