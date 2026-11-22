# Piste 5 : taux de refus / d'excuse par endpoint, sur chaque bloc (questions disjointes).
import re, csv, collections, importlib.util
spec = importlib.util.spec_from_file_location('c', 'E:/EAC/reports/critique/t2-regard-neuf-charge.py')
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
PAT = re.compile(r"\b(I cannot|I can't|I can’t|I'm sorry|I’m sorry|I am sorry|I apologi[sz]e|as an AI|as a language model|I'm not able|I am not able|I don't have the (capability|ability))", re.I)
def run(R, groups, label):
    print('==', label)
    for g, eps in groups.items():
        for e in eps:
            row = []
            for b in sorted({r['block'] for r in R}):
                rs = [r for r in R if r['ep'] == e and r['block'] == b and r['kind'] == 'reading']
                k = sum(bool(PAT.search(r['text'][:300])) for r in rs)
                row.append(f'bloc{b} {k}/{len(rs)}')
            print(f'  {g} {e}: ' + '  '.join(row))
R = c.ref()
run(R, {'ref': [f'{e}[{c.REF_TRUTH[e]}]' and e for e in sorted(c.REF_TRUTH)]}, 'REF (S=01,03,05,06,08)')
fam = {r['endpoint']: r['family'] for r in csv.DictReader(open('E:/EAC-lab/t2/reel/ep_v2.csv'))}
G = collections.defaultdict(list)
for e, f in sorted(fam.items()): G[f].append(e)
run(c.ep(), G, 'EP')
