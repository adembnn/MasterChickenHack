"""T2 gabarit : filigrane sans repetition. Idee : un filigrane a cle (tirage pilote par le contexte) rend le mot suivant plus previsible
sachant le precedent, d'une question a l'autre. Mesure : pour chaque mot precedent vu >= 4 fois chez le guichet, part du successeur le plus frequent ;
moyenne ponderee, reponses coupees a 80 mots pour egaliser. z = ecart a la mediane de la famille / dispersion de la famille (pas de z apparie ici).
Puis stabilite : meme mesure sur bloc 0 et bloc 1 separement. Usage : python t2-gabarit-filigrane.py"""
import importlib.util, collections, statistics as st
s = importlib.util.spec_from_file_location("m", __file__.replace("filigrane", "mesures")); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
def succ(texts):
    nxt = collections.defaultdict(collections.Counter)
    for t in texts:
        w = m.words(t)[:80]
        for a, b in zip(w, w[1:]): nxt[a][b] += 1
    num = den = 0
    for a, c in nxt.items():
        n = sum(c.values())
        if n >= 4: num += c.most_common(1)[0][1]; den += n
    return num / den if den else 0
def run(D, fam, draw, lab):
    eps = [e for e in m.FAM[fam] if e in D]; it = {e: m.reading_items(D, e, draw) for e in eps}
    qs = sorted(set.intersection(*[set(v) for v in it.values()]))
    v = {e: [succ([it[e][q] for q in qs if q[0] == b]) for b in (0, 1)] + [succ([it[e][q] for q in qs])] for e in eps}
    out = []
    for k in range(3):
        med = st.median(v[e][k] for e in eps); sd = st.pstdev([v[e][k] for e in eps]) or 1e-9
        for e in eps: v[e].append((v[e][k] - med) / sd)
    print(f"== {fam} {lab}   guichet  z_bloc0  z_bloc1  z_tout")
    for e in sorted(eps, key=lambda e: -v[e][5]):
        print(f"   {e} {m.REFV.get(e, m.TSTAR.get(e, '')):>9}  {v[e][3]:+5.2f}  {v[e][4]:+5.2f}  {v[e][5]:+5.2f}")
RD = m.ref_data()
for d in range(4): run(RD, "REF", d, f"tirage {d}")
ED = m.ep_data()
for f in ("F1", "F2", "F3"): run(ED, f, 0, "")
