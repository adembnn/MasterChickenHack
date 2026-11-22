"""T2 gabarit : reponses « hors sujet » (persona tiree a chaque requete) et mise en forme, par guichet.
Pour chaque question : similarite de Jaccard (ensembles de mots) de la reponse aux reponses des autres guichets de la famille.
queue = nombre de reponses sous le 5e centile de la famille ; md = part de reponses avec puces/titres/gras ; court = part sous 60 mots.
Usage : python t2-gabarit-horsujet.py   (ref : 4 tirages du bloc 0 + bloc 1, chacun separement)"""
import importlib.util, statistics as st, re, numpy as np
s = importlib.util.spec_from_file_location("m", __file__.replace("horsujet", "mesures")); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
MD = re.compile(r"(^\s*([-*•]|\d+\.)\s)|(^#)|(\*\*)", re.M)
def run(D, fam, draw, lab):
    eps = [e for e in m.FAM[fam] if e in D]; it = {e: m.reading_items(D, e, draw) for e in eps}
    qs = sorted(set.intersection(*[set(v) for v in it.values()])); S = {e: [] for e in eps}
    for q in qs:
        ws = {e: set(m.words(it[e][q])) for e in eps}
        for e in eps:
            S[e].append(st.mean(len(ws[e] & ws[x]) / max(1, len(ws[e] | ws[x])) for x in eps if x != e))
    allv = np.concatenate([S[e] for e in eps]); p5 = np.percentile(allv, 5)
    out = []
    for e in eps:
        md = st.mean(1.0 if MD.search(it[e][q]) else 0.0 for q in qs); ct = st.mean(1.0 if len(m.words(it[e][q])) < 60 else 0.0 for q in qs)
        b0 = sum(1 for q, v in zip(qs, S[e]) if v < p5 and q[0] == 0); b1 = sum(1 for q, v in zip(qs, S[e]) if v < p5 and q[0] == 1)
        out.append((e, round(st.mean(S[e]), 3), b0 + b1, b0, b1, round(md, 2), round(ct, 2)))
    print(f"== {fam} {lab} (p5 = {p5:.3f}, {len(qs)} questions)  guichet jaccard_moy queue(b0,b1) md court")
    for r in out: print(f"   {r[0]} {m.REFV.get(r[0], m.TSTAR.get(r[0], '')):>9}  J {r[1]:.3f}  queue {r[2]:2} ({r[3]},{r[4]})  md {r[5]:.2f}  court {r[6]:.2f}")
RD = m.ref_data()
for d in range(4): run(RD, "REF", d, f"tirage {d}")
ED = m.ep_data()
for f in ("F1", "F2", "F3"): run(ED, f, 0, "")
