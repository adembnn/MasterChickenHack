# Pistes 1 et 2 : prefixe commun entre endpoints, textes identiques, determinisme.
import sys, itertools, statistics as st, collections, importlib.util
spec = importlib.util.spec_from_file_location('c', 'E:/EAC/reports/critique/t2-regard-neuf-charge.py')
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)

def cpl(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y: break
        n += 1
    return n
def cplw(a, b):
    return cpl(a.split(), b.split())

def analyse(R, truth=None, label=''):
    # cle (draw, pid, rep) -> {ep: text}
    by = collections.defaultdict(dict)
    for r in R:
        by[(r['draw'], r['pid'], r['rep'])][r['ep']] = r['text']
    eps = sorted({r['ep'] for r in R})
    pair = collections.defaultdict(list)
    for k, d in by.items():
        for a, b in itertools.combinations(sorted(d), 2):
            pair[(a, b)].append((cpl(d[a], d[b]), cplw(d[a], d[b]), d[a] == d[b]))
    print(f'== {label}: {len(eps)} endpoints, {len(by)} cles question')
    rows = []
    for (a, b), v in pair.items():
        ch = st.mean(x[0] for x in v); w = st.mean(x[1] for x in v)
        eq = sum(x[2] for x in v); g20 = sum(x[1] >= 20 for x in v)
        same = None if truth is None else truth[a] == truth[b]
        rows.append((a, b, same, ch, w, eq, g20, len(v)))
    if truth:
        for s in (True, False):
            sub = [r for r in rows if r[2] == s]
            print(f"  meme modele={s}: n={len(sub)} prefixe moyen car {st.mean(r[3] for r in sub):.1f} mots {st.mean(r[4] for r in sub):.2f}"
                  f" | textes identiques total {sum(r[5] for r in sub)} | questions avec >=20 mots communs {sum(r[6] for r in sub)}")
        # AUC du prefixe moyen en mots
        P = [r[4] for r in rows if r[2]]; N = [r[4] for r in rows if not r[2]]
        auc = sum((p > q) + 0.5 * (p == q) for p in P for q in N) / (len(P) * len(N))
        print(f'  AUC prefixe moyen (mots) meme/different = {auc:.3f}')
    rows.sort(key=lambda r: -r[4])
    print('  top 12 paires par prefixe moyen en mots:')
    for r in rows[:12]:
        print('   ', r[0], r[1], 'meme' if r[2] else ('diff' if r[2] is False else '?'), f'car {r[3]:.1f} mots {r[4]:.2f} ident {r[5]} >=20mots {r[6]}/{r[7]}')
    print('  distribution prefixe moyen (mots) toutes paires: min %.2f med %.2f max %.2f' % (
        min(r[4] for r in rows), st.median(r[4] for r in rows), max(r[4] for r in rows)))
    return rows

def determinisme(R, label):
    # meme endpoint, meme question : repetitions (P002 x2) et tirages differents
    by = collections.defaultdict(list)
    for r in R:
        by[(r['ep'], r['pid'])].append(r['text'])
    print(f'== determinisme {label}')
    res = {}
    for e in sorted({r['ep'] for r in R}):
        n = eq = 0; w = []
        for (ee, pid), T in by.items():
            if ee != e or len(T) < 2: continue
            for a, b in itertools.combinations(T, 2):
                n += 1; eq += a == b; w.append(cplw(a, b))
        res[e] = (eq, n, st.mean(w) if w else 0)
        print(f'  {e}: paires repetees {n}, identiques {eq}, prefixe commun moyen {res[e][2]:.2f} mots')
    return res

if __name__ == '__main__':
    R = c.ref(); E = c.ep()
    analyse(R, c.REF_TRUTH, 'REF (4 tirages x 25 questions)')
    determinisme(R, 'REF')
    analyse(E, None, 'EP bloc 0')
    determinisme(E, 'EP (P002 x2)')
