# Piste 4 bis : signature de style par endpoint, mesuree sur des QUESTIONS DISJOINTES.
# Moitie A = questions de numero pair, moitie B = numero impair (et, si le bloc 1 est fourni, bloc 0 contre bloc 1).
# Profil = frequences des 300 tokens les plus courants de la famille, centrees sur la moyenne de la famille.
# Mesures : (1) l'endpoint se reconnait-il lui-meme d'une moitie a l'autre (rang de soi parmi la famille) ;
#           (2) sur les ref, AUC de la similarite croisee pour les paires meme modele contre modeles differents.
import re, math, collections, csv, sys, statistics as st, importlib.util, itertools
spec = importlib.util.spec_from_file_location('c', 'E:/EAC/reports/critique/t2-regard-neuf-charge.py')
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
tok = lambda s: re.findall(r"[A-Za-z']+|[^\sA-Za-z0-9]", s.lower())

def half(r, mode):
    if mode == 'bloc': return 'A' if r['block'] == 0 else 'B'
    return 'A' if int(r['pid'][1:]) % 2 == 0 else 'B'

def profiles(R, eps, mode, K=300):
    cnt = {(e, h): collections.Counter() for e in eps for h in 'AB'}
    for r in R:
        if r['ep'] in eps and r['kind'] == 'reading':
            cnt[(r['ep'], half(r, mode))].update(tok(r['text']))
    tot = collections.Counter(); [tot.update(v) for v in cnt.values()]
    vocab = [w for w, _ in tot.most_common(K)]
    P = {}
    for k, v in cnt.items():
        n = sum(v.values()) or 1
        P[k] = [math.log((v[w] + 0.5) / n) for w in vocab]
    for h in 'AB':  # centrage par moitie (retire l'effet des questions)
        mu = [st.mean(P[(e, h)][i] for e in eps) for i in range(K)]
        for e in eps: P[(e, h)] = [x - m for x, m in zip(P[(e, h)], mu)]
    return P

def cos(a, b):
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb)

def run(R, eps, label, mode='parite', truth=None):
    P = profiles(R, eps, mode)
    S = {(a, b): (cos(P[(a, 'A')], P[(b, 'B')]) + cos(P[(b, 'A')], P[(a, 'B')])) / 2 for a in eps for b in eps}
    print(f'== {label} (moities : {mode})')
    ranks = []
    for e in eps:
        order = sorted(eps, key=lambda f: -S[(e, f)])
        rk = order.index(e) + 1; ranks.append(rk)
        near = [f for f in order if f != e][:3]
        tag = f'[{truth[e]}]' if truth else ''
        print(f'  {e}{tag}: soi {S[(e, e)]:+.3f} rang {rk}/{len(eps)} ; plus proches : ' +
              ', '.join(f'{f}{"[" + truth[f] + "]" if truth else ""} {S[(e, f)]:+.3f}' for f in near))
    print(f'  rang moyen de soi {st.mean(ranks):.2f} (hasard {(len(eps) + 1) / 2:.1f}) ; reconnu 1er : {sum(r == 1 for r in ranks)}/{len(eps)}')
    if truth:
        pos = [S[(a, b)] for a, b in itertools.combinations(eps, 2) if truth[a] == truth[b]]
        neg = [S[(a, b)] for a, b in itertools.combinations(eps, 2) if truth[a] != truth[b]]
        auc = sum((p > q) + .5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))
        print(f'  AUC similarite croisee meme/different (36 paires) = {auc:.3f}')
    return S

if __name__ == '__main__':
    R = c.ref(); eps = sorted(c.REF_TRUTH)
    run(R, eps, 'REF, 100 reponses', truth=c.REF_TRUTH)
    for d in range(4):
        run([r for r in R if r['draw'] == d and r['block'] == 0], eps, f'REF tirage {d} (25 reponses)', truth=c.REF_TRUTH)
    if any(r['block'] == 1 for r in R):
        for d in range(4):
            run([r for r in R if (r['block'] == 0 and r['draw'] == d) or r['block'] == 1], eps,
                f'REF bloc 0 tirage {d} contre bloc 1 (25 + 25, questions disjointes, comme les ep)', 'bloc', truth=c.REF_TRUTH)
    fam = {r['endpoint']: r['family'] for r in csv.DictReader(open('E:/EAC-lab/t2/reel/ep_v2.csv'))}
    E = c.ep()
    mode = 'bloc' if len({r['block'] for r in E}) > 1 else 'parite'
    for F in ('F1', 'F2', 'F3'):
        run(E, sorted(e for e in fam if fam[e] == F), f'EP {F}', mode)
