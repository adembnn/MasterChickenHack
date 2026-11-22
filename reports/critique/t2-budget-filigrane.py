"""t2-budget : sur les ref, une même question posée 4 fois (bloc 0 + 3 tirages frais) : préfixe commun moyen par guichet. Valide la détection du filigrane par répétition. 0 requête."""
import json, re, collections, itertools, statistics as st
D = r"E:\EAC-lab\t2\reel"
def words(t): return re.findall(r"[a-z0-9']+", t.lower())
def pre(x, y):
    n = 0
    for u, v in zip(words(x), words(y)):
        if u != v: break
        n += 1
    return n
a = collections.defaultdict(list)
for f in ["collecte_v4_ref_bloc0_20261121T191611Z.jsonl", "collecte_v4_ref_bloc0_redraw3_20261121T225045Z.jsonl"]:
    for l in open(D + "\\" + f, encoding="utf-8"):
        r = json.loads(l)
        if r.get("type") == "answer" and r["kind"] == "reading":
            a[(r["endpoint"], r["prompt"])].append(r["response"])
by = collections.defaultdict(list)
for (e, p), xs in a.items():
    for x, y in itertools.combinations(xs, 2): by[e].append(pre(x, y))
for e in sorted(by): print(f"{e}: préfixe commun moyen {st.mean(by[e]):5.1f} mots, médiane {st.median(by[e]):4.1f}, max {max(by[e])}, {len(by[e])} paires")
print()
for e in ("ref-06", "ref-02", "ref-08", "ref-01"):
    xs = by[e]
    print(e, "part des paires avec préfixe >= 20 mots :", round(sum(x >= 20 for x in xs) / len(xs), 3), "; >= 30 :", round(sum(x >= 30 for x in xs) / len(xs), 3))
# par question : combien de questions (sur 24) montrent au moins une paire >= 20 pour ref-06
q = collections.defaultdict(int)
for (e, p), xs in a.items():
    if e == "ref-06":
        q[p] = max(pre(x, y) for x, y in itertools.combinations(xs, 2))
print("ref-06 : questions avec une paire >= 20 mots :", sum(v >= 20 for v in q.values()), "sur", len(q))
