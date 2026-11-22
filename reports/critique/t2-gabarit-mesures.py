"""T2 gabarit : traits de surface par guichet, sur le texte brut, sans GPU ni requete.
Usage : python t2-gabarit-mesures.py [--ref-tirage 0|1|2|3|tous]
Lit E:/EAC-lab/t2/reel : ep bloc 0 + ep_bloc1.jsonl ; ref bloc 0 (tirages 0 a 3) + ref_bloc1.jsonl ; T* dans ep_v2_diag.json.
Pour chaque trait : z apparie (question par question, ecart a la mediane de la famille / erreur type), par bloc puis total."""
import json, re, sys, collections, statistics as st, math
R = "E:/EAC-lab/t2/reel/"
FAM = {"F1": "01 02 04 12 14 17 18 23 24 25 33", "F2": "03 06 07 08 20 21 22 26 28 30 32", "F3": "05 09 10 11 13 15 16 19 27 29 31"}
FAM = {k: ["ep-" + x for x in v.split()] for k, v in FAM.items()}
FAM["REF"] = ["ref-%02d" % i for i in range(1, 10)]
REFV = {"ref-01": "clair", "ref-02": "fp8", "ref-03": "prompt", "ref-04": "LoRA0.7", "ref-05": "persona", "ref-06": "filigrane", "ref-07": "LoRA1.5", "ref-08": "T1.0", "ref-09": "LoRA0.3"}
W = re.compile(r"[a-z']+")

def load(files, draw=None):
    D = collections.defaultdict(dict)
    for f in files:
        for l in open(R + f, encoding="utf8"):
            r = json.loads(l)
            if r.get("type") not in ("answer", None) or "response" not in r: continue
            d = r.get("draw", 0) if draw is None else draw
            D[r["endpoint"]][(r.get("block", 0), d, r.get("probe_id", r.get("id")), r.get("repeat", 0))] = r
    return D

def ep_data():
    D = load(["collecte_v4_ep_bloc0_20261121T193826Z.jsonl"]); D1 = load(["ep_bloc1.jsonl"])
    for e in D1: D[e].update(D1[e])
    return D

def ref_data():
    D = load(["collecte_v4_ref_bloc0_20261121T191611Z.jsonl"], draw=0)
    for i in (1, 2, 3):
        Di = load(["ref_frais_%d.jsonl" % i])
        for e in Di: D[e].update(Di[e])
    D1 = load(["ref_bloc1.jsonl"], draw=0)
    for e in D1: D[e].update(D1[e])
    return D

def words(t): return W.findall(t.lower())

def ngram_rep(w, n=3):
    g = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return (len(g) - len(set(g))) / len(g) if g else 0.0

def ttr(w, n=60):
    w = w[:n]; return len(set(w)) / len(w) if w else 0.0

def natural_end(t):
    t = t.rstrip(); return 1.0 if t and (t[-1] in '.!?)"\'*`:' or t.endswith("```")) else 0.0

def reading_items(D, e, draw):
    """reponses de lecture (kind reading ou repeat rep 0), cle (bloc, question)"""
    out = {}
    for (b, d, p, rep), r in D[e].items():
        if d != draw and b == 0: continue
        if r.get("kind") in ("extract", "identity") or p in ("P000", "P001"): continue
        if rep != 0: continue
        out[(b, p)] = r["response"]
    return out

def features(D, fam, draw=0):
    eps = [e for e in FAM[fam] if e in D]
    items = {e: reading_items(D, e, draw) for e in eps}
    qs = sorted(set.intersection(*[set(v) for v in items.values()]))
    W_ = {e: {q: words(items[e][q]) for q in qs} for e in eps}
    F = {e: collections.defaultdict(dict) for e in eps}
    for q in qs:
        # mots rares : absents des reponses des AUTRES guichets de la famille a la meme question
        cnt = collections.Counter();
        for e in eps: cnt.update(set(W_[e][q]))
        open3 = collections.Counter(" ".join(W_[e][q][:3]) for e in eps)
        for e in eps:
            w = W_[e][q]; s = set(w)
            rare = sum(1 for x in s if cnt[x] == 1) / max(1, len(s))
            o = " ".join(w[:3])
            F[e]["rare"][q] = rare
            F[e]["ttr60"][q] = ttr(w)
            F[e]["rep3"][q] = ngram_rep(w)
            F[e]["len"][q] = len(w)
            F[e]["fin"][q] = natural_end(items[e][q])
            F[e]["ouv_seule"][q] = 1.0 if open3[o] == 1 else 0.0   # ouverture (3 mots) que personne d'autre n'a
            F[e]["je"][q] = sum(1 for x in w if x in ("i", "i'm", "my", "me", "i've", "i'd")) / max(1, len(w)) * 100
    return eps, qs, F

def zpair(F, eps, feat, qs):
    res = {}
    for e in eps:
        d = []
        for q in qs:
            med = st.median(F[x][feat][q] for x in eps if x != e)
            d.append(F[e][feat][q] - med)
        m = st.mean(d); s = st.pstdev(d) or 1e-9
        res[e] = (m, m / (s / math.sqrt(len(d))), st.mean(F[e][feat][q] for q in qs))
    return res

def sondes(D, e, draw=0):
    out = {}
    for (b, d, p, rep), r in D[e].items():
        if b == 0 and d == draw and p in ("P000", "P001"): out[p] = r["response"]
    return out

def p002(D, e, draw=0):
    a = [r["response"] for (b, d, p, rep), r in D[e].items() if b == 0 and d == draw and p == "P002"]
    if len(a) < 2: return None
    w1, w2 = words(a[0]), words(a[1]); k = 0
    while k < min(len(w1), len(w2)) and w1[k] == w2[k]: k += 1
    return k

KEYS = ["qwen", "alibaba", "claude", "anthropic", "llama", "meta", "openai", "gpt", "smollm", "hugging", "assistant", "you are"]
def ident(t):
    t = t.lower(); return ",".join(k for k in KEYS if k in t) or "-"

def table(D, fam, draw=0, label=""):
    eps, qs, F = features(D, fam, draw)
    feats = ["rare", "ttr60", "rep3", "len", "fin", "ouv_seule", "je"]
    Z = {f: zpair(F, eps, f, qs) for f in feats}
    # stabilite bloc 0 / bloc 1
    q0 = [q for q in qs if q[0] == 0]; q1 = [q for q in qs if q[0] == 1]
    Zb = {f: (zpair(F, eps, f, q0), zpair(F, eps, f, q1)) if q0 and q1 else None for f in feats}
    print(f"\n=== {fam} {label} : {len(eps)} guichets, {len(qs)} questions (bloc0 {len(q0)}, bloc1 {len(q1)}) ===")
    print("guichet  T*   | " + " | ".join(f"{f:>15}" for f in feats) + " | P002 | P001 ident / P000 debut")
    for e in eps:
        row = []
        for f in feats:
            m, z, v = Z[f][e]
            st_ = ""
            if Zb[f]:
                z0, z1 = Zb[f][0][e][1], Zb[f][1][e][1]; st_ = "*" if (abs(z0) > 2 and abs(z1) > 2 and z0 * z1 > 0) else " "
            row.append(f"{v:6.3f} z{z:+5.1f}{st_}")
        s = sondes(D, e, draw); pp = p002(D, e, draw)
        T = TSTAR.get(e, "")
        print(f"{e:7} {T!s:4} | " + " | ".join(f"{x:>15}" for x in row) + f" | {pp!s:>4} | {ident(s.get('P001',''))} / {s.get('P000','')[:70]!r}")
    return Z

TSTAR = {}
try:
    dg = json.load(open(R + "ep_v2_diag.json"))["diag"]; TSTAR = {e: v["T"] for e, v in dg.items()}
except Exception: pass

if __name__ == "__main__":
    arg = sys.argv[sys.argv.index("--ref-tirage") + 1] if "--ref-tirage" in sys.argv else "0"
    RD = ref_data()
    for d in ([0, 1, 2, 3] if arg == "tous" else [int(arg)]):
        table(RD, "REF", d, f"tirage {d} du bloc 0 + bloc 1  (verite : " + ", ".join(f"{k[-2:]}={v}" for k, v in REFV.items()) + ")")
    # filigrane : repetitions sur tous les tirages ref
    print("\nREF, toutes paires de tirages, meme question : textes identiques / prefixe >= 30 mots")
    for e in FAM["REF"]:
        by = collections.defaultdict(list)
        for (b, d, p, rep), r in RD[e].items():
            if b == 0: by[p].append(words(r["response"]))
        ident_, long_, n = 0, 0, 0
        for p, L in by.items():
            for i in range(len(L)):
                for j in range(i + 1, len(L)):
                    n += 1; k = 0
                    while k < min(len(L[i]), len(L[j])) and L[i][k] == L[j][k]: k += 1
                    ident_ += L[i] == L[j]; long_ += k >= 30
        print(f"  {e} {REFV[e]:10} paires {n:4}  identiques {ident_:3}  prefixe>=30 {long_:3}")
    ED = ep_data()
    for fam in ("F1", "F2", "F3"): table(ED, fam, 0)
