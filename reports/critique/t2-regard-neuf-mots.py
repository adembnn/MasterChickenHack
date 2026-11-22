# Piste 5 : mots sur-representes par endpoint (empreinte lexicale d'une LoRA).
# Pour chaque endpoint : log-odds (prior de Dirichlet) de chaque mot contre les autres endpoints de la meme famille.
import re, math, collections, csv, sys, importlib.util
spec = importlib.util.spec_from_file_location('c', 'E:/EAC/reports/critique/t2-regard-neuf-charge.py')
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
tok = lambda s: re.findall(r"[A-Za-z']+|[0-9]+|[^\sA-Za-z0-9]", s.lower())

def logodds(groups, a=0.5):
    tot = collections.Counter()
    for g in groups.values(): tot.update(g)
    N0 = sum(tot.values()); V = len(tot)
    res = {}
    for e, g in groups.items():
        rest = tot - g; n1 = sum(g.values()); n2 = sum(rest.values())
        out = []
        for w in g | rest:
            y1, y2 = g[w], rest[w]
            l1 = math.log((y1 + a) / (n1 + a * V - y1 - a)); l2 = math.log((y2 + a) / (n2 + a * V - y2 - a))
            var = 1 / (y1 + a) + 1 / (y2 + a)
            out.append(((l1 - l2) / math.sqrt(var), w, y1, y2))
        out.sort(reverse=True)
        res[e] = out
    return res

def run(R, fams, label, top=8):
    print('==', label)
    for F, eps in fams.items():
        groups = {e: collections.Counter() for e in eps}
        for r in R:
            if r['ep'] in groups and r['kind'] == 'reading':
                groups[r['ep']].update(tok(r['text']))
        lo = logodds(groups)
        for e in eps:
            s = ', '.join(f'{w}({y1}/{y2}, z{z:.1f})' for z, w, y1, y2 in lo[e][:top])
            print(f'  {F} {e}: {s}')

if __name__ == '__main__':
    R = c.ref()
    run(R, {'ref': sorted(c.REF_TRUTH)}, 'REF (S=01,03,05,06,08 ; fp8=02 ; LoRA 0.7=04, 1.5=07, 0.3=09)')
    fam = {r['endpoint']: r['family'] for r in csv.DictReader(open('E:/EAC-lab/t2/reel/ep_v2.csv'))}
    fams = collections.defaultdict(list)
    for e, f in sorted(fam.items()): fams[f].append(e)
    run(c.ep(), fams, 'EP bloc 0')
