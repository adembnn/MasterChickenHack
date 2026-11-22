# Piste 1 bis : matrice du prefixe commun moyen (en mots) entre endpoints d'une famille, par bloc,
# et stabilite d'un bloc a l'autre (questions disjointes) : correlation des deux matrices et permutation.
import csv, itertools, collections, statistics as st, random, importlib.util
spec = importlib.util.spec_from_file_location('c', 'E:/EAC/reports/critique/t2-regard-neuf-charge.py')
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)

def cplw(a, b):
    n = 0
    for x, y in zip(a.split(), b.split()):
        if x != y: break
        n += 1
    return n

def mat(R, eps, block):
    by = collections.defaultdict(dict)
    for r in R:
        if r['ep'] in eps and r['block'] == block and r['kind'] == 'reading':
            by[(r['draw'], r['pid'], r['rep'])][r['ep']] = r['text']
    M = {}
    for a, b in itertools.combinations(eps, 2):
        v = [min(cplw(d[a], d[b]), 30) for d in by.values() if a in d and b in d]
        M[(a, b)] = st.mean(v) if v else 0
    return M

def corr(x, y):
    mx, my = st.mean(x), st.mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return num / (sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)) ** .5

def show(eps, M, truth=None):
    print('        ' + ' '.join(e[-2:].rjust(5) for e in eps))
    for a in eps:
        row = [('  -  ' if a == b else f'{M[tuple(sorted((a, b)))]:5.1f}') for b in eps]
        print(f'  {a}{("[" + truth[a] + "]") if truth else ""}'.ljust(10) + ' '.join(row))

def stab(R, eps, label, truth=None, b0=0, b1=1):
    M0, M1 = mat(R, eps, b0), mat(R, eps, b1)
    print(f'== {label} : bloc {b0}'); show(eps, M0, truth)
    print(f'== {label} : bloc {b1}'); show(eps, M1, truth)
    keys = list(M0); x = [M0[k] for k in keys]; y = [M1[k] for k in keys]
    r = corr(x, y)
    # permutation des etiquettes d'endpoints dans le bloc 1
    rnd = random.Random(0); cnt = 0; N = 2000
    for _ in range(N):
        p = eps[:]; rnd.shuffle(p); mp = dict(zip(eps, p))
        y2 = [M1[tuple(sorted((mp[a], mp[b])))] for a, b in keys]
        cnt += corr(x, y2) >= r
    print(f'  correlation bloc {b0} / bloc {b1} des {len(keys)} paires : r = {r:.2f} (p permutation = {cnt / N:.3f})')
    if truth:
        for s in (True, False):
            v = [(M0[k] + M1[k]) / 2 for k in keys if (truth[k[0]] == truth[k[1]]) == s]
            print(f'  meme modele={s}: prefixe moyen {st.mean(v):.2f} mots (n={len(v)})')

if __name__ == '__main__':
    R = c.ref(); eps = sorted(c.REF_TRUTH)
    stab(R, eps, 'REF', c.REF_TRUTH)
    fam = {r['endpoint']: r['family'] for r in csv.DictReader(open('E:/EAC-lab/t2/reel/ep_v2.csv'))}
    E = c.ep()
    for F in ('F1', 'F2', 'F3'):
        stab(E, sorted(e for e in fam if fam[e] == F), f'EP {F}')
