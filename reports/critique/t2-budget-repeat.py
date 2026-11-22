"""t2-budget : la même question posée 2 fois (P002, bloc 0) : textes identiques ? (filigrane / déterminisme). 0 requête."""
import json, re, collections, statistics as st
D = r"E:\EAC-lab\t2\reel"
def words(t): return re.findall(r"[a-z0-9']+", t.lower())
def run(files):
    a = collections.defaultdict(dict)
    for f in files:
        for l in open(D + "\\" + f, encoding="utf-8"):
            r = json.loads(l)
            if r.get("type") == "answer" and r.get("probe_id") == "P002":
                a[r["endpoint"]].setdefault(str(r.get("draw", "0")), {})[int(r["repeat"])] = r["response"]
    return a
def sim(x, y):
    n = 0
    for u, v in zip(words(x), words(y)):
        if u != v: break
        n += 1
    sx, sy = set(words(x)), set(words(y))
    return n, len(sx & sy) / max(1, len(sx | sy)), x == y
for lab, files in (("ep", ["collecte_v4_ep_bloc0_20261121T193826Z.jsonl"]), ("ref", ["collecte_v4_ref_bloc0_20261121T191611Z.jsonl"])):
    a = run(files)
    print(lab)
    for e in sorted(a):
        d = a[e].get("0", {})
        if 0 in d and 1 in d:
            n, j, eq = sim(d[0], d[1])
            print(f"  {e}: préfixe commun {n:3d} mots, Jaccard {j:.2f}, identiques={eq}")
