# Piste 3 : signaux de surface par endpoint (longueur appariee par question, coupure, mise en forme).
import re, math, collections, csv, statistics as st, importlib.util, itertools
spec = importlib.util.spec_from_file_location('c', 'E:/EAC/reports/critique/t2-regard-neuf-charge.py')
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)

FEATS = {
    'logmots': lambda t: math.log(1 + len(t.split())),
    'newlines': lambda t: t.count('\n') / max(1, len(t)) * 1000,
    'gras': lambda t: t.count('**') / max(1, len(t)) * 1000,
    'puces': lambda t: len(re.findall(r'(?m)^\s*(?:[-*]|\d+\.)\s', t)),
    'motlong': lambda t: st.mean(len(w) for w in t.split()) if t.split() else 0,
    'phrases': lambda t: len(re.findall(r'[.!?](\s|$)', t)),
    'virgules': lambda t: t.count(',') / max(1, len(t.split())) * 100,
    'nonascii': lambda t: sum(ord(ch) > 127 for ch in t) / max(1, len(t)) * 1000,
}

def paired(R, eps, label, truth=None):
    # valeur de l'endpoint moins la mediane des endpoints de la famille, meme question et meme tirage
    by = collections.defaultdict(dict)
    for r in R:
        if r['kind'] == 'reading' and r['ep'] in eps:
            by[(r['draw'], r['pid'], r['rep'])][r['ep']] = r['text']
    print(f'== {label} (z = moyenne des ecarts apparies / erreur type ; n questions)')
    hdr = 'endpoint'.ljust(12) + ''.join(k[:9].rjust(10) for k in FEATS) + '   coupe'
    print(hdr)
    fin = collections.Counter(); tot = collections.Counter()
    for r in R:
        if r['ep'] in eps: tot[r['ep']] += 1; fin[r['ep']] += r['fin'] == 'length'
    out = {}
    for e in eps:
        row = []
        for k, f in FEATS.items():
            d = []
            for key, dd in by.items():
                if e in dd and len(dd) >= 3:
                    others = [f(t) for ee, t in dd.items() if ee != e]
                    d.append(f(dd[e]) - st.median(others))
            z = st.mean(d) / (st.stdev(d) / math.sqrt(len(d))) if len(d) > 2 and st.stdev(d) > 0 else 0
            row.append(z)
        out[e] = row
        tag = f' [{truth[e]}]' if truth else ''
        print((e + tag).ljust(12) + ''.join(f'{z:10.1f}' for z in row) + f'   {fin[e]}/{tot[e]}')
    return out

if __name__ == '__main__':
    R = c.ref()
    eps = sorted(c.REF_TRUTH)
    paired([r for r in R if r['block'] == 0], eps, 'REF bloc 0, 4 tirages', c.REF_TRUTH)
    for d in range(4):
        o = paired([r for r in R if r['draw'] == d and r['block'] == 0], eps, f'REF tirage {d} seul (25 reponses, comme un ep)', c.REF_TRUTH)
    fam = {r['endpoint']: r['family'] for r in csv.DictReader(open('E:/EAC-lab/t2/reel/ep_v2.csv'))}
    E = c.ep()
    if any(r['block'] == 1 for r in R):
        paired([r for r in R if r['block'] == 1], eps, 'REF bloc 1 seul (25 questions neuves)', c.REF_TRUTH)
    for F in ('F1', 'F2', 'F3'):
        for b in sorted({r['block'] for r in E}):
            paired([r for r in E if r['block'] == b], sorted(e for e in fam if fam[e] == F), f'EP {F} bloc {b}')
