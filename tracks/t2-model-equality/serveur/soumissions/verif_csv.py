# controle avant envoi T2 : python verif_csv.py REFERENCE.csv CANDIDAT.csv
import csv, sys
def lire(p):
    rows = [r for r in csv.reader(open(p, encoding="utf-8")) if r and r[0].strip().startswith("ep-")]
    return {r[0].strip(): (r[1].strip(), r[2].strip()) for r in rows}, len(rows)
ref, nr = lire(sys.argv[1]); cand, nc = lire(sys.argv[2])
attendu = {"ep-%02d" % i for i in range(1, 34)}
ok = True
def faute(m):
    global ok; ok = False; print("ECHEC :", m)
if nc != 33 or set(cand) != attendu: faute("lignes %d, endpoints manquants %s, en trop %s" % (nc, sorted(attendu - set(cand)), sorted(set(cand) - attendu)))
fam = {}
for e, (f, m) in cand.items():
    if m in fam and fam[m] != f: faute("le modele %s est dans deux familles (%s, %s)" % (m, fam[m], f))
    fam[m] = f
def part(d, i):
    g = {}
    for e, v in d.items(): g.setdefault(v[i], set()).add(e)
    return {frozenset(s) for s in g.values()}
if set(ref) == set(cand):
    if part(ref, 0) != part(cand, 0): faute("la partition en familles change")
    def paires(d):
        es = sorted(d); return {(a, b) for i, a in enumerate(es) for b in es[i+1:] if d[a][1] == d[b][1]}
    pr, pc = paires(ref), paires(cand)
    print("paires meme modele : reference %d, candidat %d ; ajoutees %d, retirees %d" % (len(pr), len(pc), len(pc - pr), len(pr - pc)))
    bouge = sorted({e for p in (pr ^ pc) for e in p})
    print("endpoints dont au moins une paire change :", bouge)
    print("groupes de modeles du candidat :", sorted([sorted(s) for s in part(cand, 1) if len(s) > 1], key=len, reverse=True))
    print("isoles du candidat :", sorted(e for s in part(cand, 1) if len(s) == 1 for e in s))
else: faute("la reference n a pas les 33 endpoints")
print("VERDICT :", "CONFORME" if ok else "NON CONFORME")
