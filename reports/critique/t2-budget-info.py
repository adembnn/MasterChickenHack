"""t2-budget : quelles questions ont porté le plus d'information entre variantes et copies ? (0 requête, 0 GPU)
Critères simples et rejouables, calculés sur les réponses déjà payées :
  len  = nombre de mots de la réponse
  div  = diversité lexicale = mots distincts / mots, sur les 60 premiers mots
  voc  = distance de vocabulaire = 1 - Jaccard(mots de la réponse, mots de l'ensemble des copies de la famille)
Pour chaque question q, chaque famille F, chaque critère m :
  d(q) = (moyenne du critère sur les variantes - moyenne sur les copies) / écart type intra-copies de ce critère dans F
  (voc : d = distance(variante, copies) - distance(copie, autres copies), même normalisation)
Un d au hasard n'est pas nul en valeur absolue : on retranche la valeur attendue sous 200 permutations des étiquettes.
Usage : python t2-budget-info.py [--ref]   (données : E:\\EAC-lab\\t2\\reel)
"""
import json, re, sys, random, statistics as st, collections, os

D = r"E:\EAC-lab\t2\reel"
random.seed(7)

FAM = {"qwen3b": "01 02 04 12 14 17 18 23 24 25 33", "qwen15b": "03 06 07 08 20 21 22 26 28 30 32",
       "llama": "05 09 10 11 13 15 16 19 27 29 31"}
FAM = {f: ["ep-" + x for x in v.split()] for f, v in FAM.items()}
VAR = {"qwen3b": ["ep-17", "ep-18"], "qwen15b": ["ep-22"], "llama": ["ep-05", "ep-15"]}
COP = {"qwen3b": ["ep-04", "ep-14", "ep-23"], "qwen15b": ["ep-21", "ep-32", "ep-07"],
       "llama": ["ep-10", "ep-13", "ep-19"]}  # Llama : pas de copie confirmée par le public, on prend les 3 « simples » (T* = 0,70)
REF_VAR, REF_COP = ["ref-02", "ref-04", "ref-07", "ref-09"], ["ref-01", "ref-03", "ref-05", "ref-06", "ref-08"]


def qtype(p):
    if p.startswith("Repeat all the text"): return "extract"
    if p.startswith("Which language model"): return "identity"
    if p.startswith("Write a short paragraph of about 80"): return "repeat80"
    if p.startswith("Explain"): return "explain"
    if p.startswith("Write a short Python"): return "python"
    if p.startswith("Describe, step by step"): return "steps"
    if p.startswith("Compare"): return "compare"
    if p.startswith("Write the opening paragraph"): return "story"
    if p.startswith("Rewrite this politely"): return "rewrite"
    if "numbered list" in p: return "list"
    return "autre"


def load(files, field_draw=False):
    """-> {(endpoint, prompt): [réponses]}"""
    out = collections.defaultdict(list)
    for f in files:
        for l in open(os.path.join(D, f), encoding="utf-8"):
            r = json.loads(l)
            if r.get("type") == "answer" and r.get("response") is not None:
                out[(r["endpoint"], r["prompt"])].append(r["response"])
    return out


def words(t): return re.findall(r"[a-z0-9']+", t.lower())


def m_len(t): return len(words(t))


def m_div(t):
    w = words(t)[:60]
    return len(set(w)) / len(w) if w else 0.0


def jac(a, b):
    a, b = set(a), set(b)
    return 1 - len(a & b) / max(1, len(a | b))


def analyse(data, eps_by_group, label):
    """eps_by_group : liste de (nom, variantes, copies). Renvoie {type: {critère: (d signé moyen, excès de |d| sur le hasard, n questions)}}"""
    prompts = sorted({p for (_, p) in data})
    res = collections.defaultdict(lambda: collections.defaultdict(list))  # type -> critère -> liste de d (signés)
    nul = collections.defaultdict(lambda: collections.defaultdict(list))
    for name, V, C, pool in eps_by_group:
        def val(e, p, m):
            xs = data.get((e, p))
            if not xs: return None
            if m == "len": return st.mean(m_len(x) for x in xs)
            if m == "div": return st.mean(m_div(x) for x in xs)
        for m in ("len", "div", "voc", "open"):
            # écart type intra-copies, pour normaliser
            def score(Vs, Cs, p):
                if m == "voc":
                    ref_words = lambda es: [w for e in es for x in data.get((e, p), []) for w in words(x)]
                    dv, dc = [], []
                    for e in Vs:
                        for x in data.get((e, p), []):
                            dv.append(jac(words(x), ref_words(Cs)))
                    for e in Cs:
                        others = [c for c in Cs if c != e]
                        for x in data.get((e, p), []):
                            dc.append(jac(words(x), ref_words(others)))
                    if not dv or not dc: return None
                    return st.mean(dv) - st.mean(dc)
                if m == "open":
                    def pre(x, y):
                        n = 0
                        for u, v in zip(words(x)[:8], words(y)[:8]):
                            if u != v: break
                            n += 1
                        return n
                    def mp(e, others):
                        return [st.mean(pre(x, y) for o in others for y in data.get((o, p), []))
                                for x in data.get((e, p), []) if any(data.get((o, p)) for o in others)]
                    dv = [v for e in Vs for v in mp(e, Cs)]
                    dc = [v for e in Cs for v in mp(e, [c for c in Cs if c != e])]
                    if not dv or not dc: return None
                    return -(st.mean(dv) - st.mean(dc))
                a = [val(e, p, m) for e in Vs]; b = [val(e, p, m) for e in Cs]
                a = [x for x in a if x is not None]; b = [x for x in b if x is not None]
                if not a or not b: return None
                return st.mean(a) - st.mean(b)
            # normalisation : sd des copies autour de leur moyenne, poolé sur toutes les questions
            dev = []
            for p in prompts:
                b = [val(e, p, m) if m != "voc" else None for e in C]
                b = [x for x in b if x is not None]
                if len(b) > 1:
                    mu = st.mean(b); dev += [x - mu for x in b]
            if m == "open":
                s = 1.5
            elif m == "voc":
                s = 0.06  # échelle fixe pour voc (écart type typique d'une distance de Jaccard entre deux réponses)
                devv = []
                for p in prompts:
                    for e in C:
                        others = [c for c in C if c != e]
                        for x in data.get((e, p), []):
                            devv.append(jac(words(x), [w for o in others for y in data.get((o, p), []) for w in words(y)]))
                s = st.pstdev(devv) if len(devv) > 3 else 0.06
            else:
                s = st.pstdev(dev) if len(dev) > 3 else 1.0
            s = s or 1.0
            for p in prompts:
                sc = score(V, C, p)
                if sc is None: continue
                res[qtype(p)][m].append(sc / s)
                # hasard : mêmes tailles, tirées dans le pool (guichets hors variantes)
                for _ in range(20):
                    smp = random.sample(pool, len(V) + len(C))
                    sn = score(smp[:len(V)], smp[len(V):], p)
                    if sn is not None: nul[qtype(p)][m].append(abs(sn / s))
    out = {}
    for t in res:
        out[t] = {}
        for m in res[t]:
            xs = res[t][m]
            out[t][m] = (st.mean(xs), st.mean(abs(x) for x in xs) - st.mean(nul[t][m]), len(xs))
    return out


def show(out, label):
    print(f"\n== {label} ==  (d signé moyen | excès de |d| sur le hasard | nb questions·familles)")
    print(f"{'type':9s} " + " | ".join(f"{m:>22s}" for m in ("len", "div", "voc", "open")))
    for t in sorted(out, key=lambda t: -sum(out[t][m][1] for m in out[t])):
        print(f"{t:9s} " + " | ".join(f"{out[t][m][0]:+6.2f} {out[t][m][1]:+6.2f}  n={out[t][m][2]:<4d}" if m in out[t] else " " * 22 for m in ("len", "div", "voc", "open")))


if __name__ == "__main__":
    ep = load(["collecte_v4_ep_bloc0_20261121T193826Z.jsonl", "collecte_v4_ep_bloc1_20261122T011020Z.jsonl",
               "collecte_v4_ep_bloc1_20261122T021208Z.jsonl"])
    groups = [(f, VAR[f], COP[f], [e for e in FAM[f] if e not in VAR[f]]) for f in FAM]
    # une question n'est comptée que si tous les guichets du groupe l'ont (le bloc 1 est complet à 825 / 825)
    o_ep = analyse(ep, groups, "ep")
    show(o_ep, "ep : variantes confirmées vs copies confirmées (3 familles)")
    ref = load(["collecte_v4_ref_bloc0_20261121T191611Z.jsonl", "collecte_v4_ref_bloc0_redraw3_20261121T225045Z.jsonl",
                "collecte_v4_ref_bloc1_20261122T004456Z.jsonl"])
    o_ref = analyse(ref, [("ref", REF_VAR, REF_COP, REF_COP + REF_VAR)], "ref")
    show(o_ref, "ref (corrigé connu) : 02 fp8, 04 LoRA 0,7, 07 LoRA 1,5, 09 LoRA 0,3 vs 5 copies")
    # accord entre les deux classements (somme des excès sur les trois critères)
    ts = [t for t in o_ep if t in o_ref]
    a = {t: sum(o_ep[t][m][1] for m in o_ep[t]) for t in ts}
    b = {t: sum(o_ref[t][m][1] for m in o_ref[t]) for t in ts}
    ra = {t: i for i, t in enumerate(sorted(ts, key=lambda t: -a[t]))}
    rb = {t: i for i, t in enumerate(sorted(ts, key=lambda t: -b[t]))}
    n = len(ts); d2 = sum((ra[t] - rb[t]) ** 2 for t in ts)
    print(f"\nCorrélation de rang (Spearman) ep vs ref sur {n} types : {1 - 6 * d2 / (n * (n * n - 1)):+.2f}")
    print("classement ep :", sorted(ts, key=lambda t: -a[t]))
    print("classement ref:", sorted(ts, key=lambda t: -b[t]))
    json.dump({"ep": {t: {m: list(v) for m, v in o_ep[t].items()} for t in o_ep},
               "ref": {t: {m: list(v) for m, v in o_ref[t].items()} for t in o_ref}},
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "t2-budget-info.json"), "w"), indent=1)
