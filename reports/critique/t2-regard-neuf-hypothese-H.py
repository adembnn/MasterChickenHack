# Hypothese H (structure des ref transposee) : chaque endpoint porte UNE seule modification.
# Donc tout endpoint deguise (prompt systeme, persona, filigrane, temperature/decodage) est une COPIE,
# et parmi les endpoints non deguises d'une famille, un seul est la base en clair, les autres sont des variantes.
# Ce script (1) verifie H sur le corrige des ref, (2) calcule le score exact (0,1 MCC famille + 0,9 MCC modele)
# de plusieurs fichiers sous des structures cachees compatibles avec H, sans aucune requete.
import itertools, math

def mcc(tp, tn, fp, fn):
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return max(0.0, (tp * tn - fp * fn) / d) if d else 0.0

def score(truth, pred):
    """truth, pred : dict ep -> (famille, modele)"""
    eps = sorted(truth); cf = [0] * 4; cm = [0] * 4
    for a, b in itertools.combinations(eps, 2):
        sf = truth[a][0] == truth[b][0]; pf = pred[a][0] == pred[b][0]
        cf[(not sf) * 2 + (sf != pf)] += 1  # 0 tp,1 fn,2 tn,3 fp
        if sf:
            sm = truth[a] == truth[b]; pm = pred[a] == pred[b]
            cm[(not sm) * 2 + (sm != pm)] += 1
    f = mcc(cf[0], cf[2], cf[3], cf[1]); m = mcc(cm[0], cm[2], cm[3], cm[1])
    return 0.1 * f + 0.9 * m

def build(groups):
    """groups : {famille: [liste de groupes (chaque groupe = liste d'ep)]}"""
    d = {}
    for F, gs in groups.items():
        for i, g in enumerate(gs):
            for e in g: d[e] = (F, f'{F}m{i}')
    return d

# 1) H sur les ref : deguises = 03 (prompt), 05 (persona), 06 (filigrane), 08 (T 1,0) -> tous S dans le corrige
ref_truth = {'ref-01':'S','ref-02':'T1','ref-03':'S','ref-04':'T3','ref-05':'S','ref-06':'S','ref-07':'T4','ref-08':'S','ref-09':'T2'}
degu = ['ref-03', 'ref-05', 'ref-06', 'ref-08']; clair = ['ref-01', 'ref-02', 'ref-04', 'ref-07', 'ref-09']
print('H sur les ref : deguises tous copies ?', all(ref_truth[e] == 'S' for e in degu),
      '| non deguises : copies =', [e for e in clair if ref_truth[e] == 'S'], 'variantes =', [e for e in clair if ref_truth[e] != 'S'])

# 2) Structures ep compatibles avec H (familles du fichier 1967).
F1 = ['ep-01','ep-02','ep-04','ep-12','ep-14','ep-17','ep-18','ep-23','ep-24','ep-25','ep-33']
F2 = ['ep-03','ep-06','ep-07','ep-08','ep-20','ep-21','ep-22','ep-26','ep-28','ep-30','ep-32']
F3 = ['ep-05','ep-09','ep-10','ep-11','ep-13','ep-15','ep-16','ep-19','ep-27','ep-29','ep-31']
CLAIR = {'F1': ['ep-01','ep-04','ep-12','ep-17','ep-18'],
         'F2': ['ep-03','ep-21','ep-22','ep-26','ep-32'],
         'F3': ['ep-05','ep-09','ep-10','ep-11','ep-13','ep-15','ep-19','ep-27']}
FAM = {'F1': F1, 'F2': F2, 'F3': F3}
SURS = {'F1': ['ep-17','ep-18'], 'F2': ['ep-22'], 'F3': ['ep-05','ep-15']}

def truth_H(plain, var):
    g = {}
    for F in FAM:
        base = [e for e in FAM[F] if e not in var[F]]
        g[F] = [base] + [[v] for v in var[F]]
    return build(g)

def file_1967():
    return build({'F1': [[e for e in F1 if e not in ('ep-17','ep-18')], ['ep-17','ep-18']],
                  'F2': [[e for e in F2 if e != 'ep-22'], ['ep-22']],
                  'F3': [[e for e in F3 if e not in ('ep-05','ep-15')], ['ep-05'], ['ep-15']]})

def file_H(var):
    return build({F: [[e for e in FAM[F] if e not in var[F]]] + [[v] for v in var[F]] for F in FAM})

# Enumerer toutes les structures H : pour F1 et F2, la base en clair est l'un des non-deguises qui n'est pas une variante sure ;
# pour F3, 4 variantes parmi les 8 non deguises dont 05 et 15.
res = {'1967': [], 'H_bonne_base': [], 'H_base_fausse': [], 'H_sans_F3': []}
F3_opts = [c for c in itertools.combinations(CLAIR['F3'], 4) if 'ep-05' in c and 'ep-15' in c]
for p1 in [e for e in CLAIR['F1'] if e not in SURS['F1']]:
    for p2 in [e for e in CLAIR['F2'] if e not in SURS['F2']]:
        for v3 in F3_opts:
            var = {'F1': [e for e in CLAIR['F1'] if e != p1], 'F2': [e for e in CLAIR['F2'] if e != p2], 'F3': list(v3)}
            T = truth_H(None, var)
            res['1967'].append(score(T, file_1967()))
            res['H_bonne_base'].append(score(T, file_H(var)))
            # base mal choisie dans F1 et F2 (on isole la base, on rattache une variante)
            q1 = [e for e in CLAIR['F1'] if e not in SURS['F1'] and e != p1][0]
            q2 = [e for e in CLAIR['F2'] if e not in SURS['F2'] and e != p2][0]
            wrong = {'F1': [e for e in CLAIR['F1'] if e != q1], 'F2': [e for e in CLAIR['F2'] if e != q2], 'F3': list(v3)}
            res['H_base_fausse'].append(score(T, file_H(wrong)))
            # H applique a F1 et F2 seulement, F3 garde 05 et 15
            part = {'F1': var['F1'], 'F2': var['F2'], 'F3': ['ep-05', 'ep-15']}
            res['H_sans_F3'].append(score(T, file_H(part)))
for k, v in res.items():
    print(f'{k:15s} : score moyen {sum(v)/len(v):.3f}  min {min(v):.3f}  max {max(v):.3f}  ({len(v)} structures)')

# 3) Et si H est fausse (deguises qui portent aussi une variante) : vraie structure = 1967 + 2 variantes cachees
# parmi les deguises de F1 ; que vaut le fichier H ?
var = {'F1': ['ep-01','ep-04','ep-17','ep-18'], 'F2': ['ep-03','ep-21','ep-22','ep-26'], 'F3': ['ep-05','ep-15','ep-09','ep-11']}
T = truth_H(None, var); print('controle : H exacte, fichier H =', round(score(T, file_H(var)), 3), '; fichier 1967 =', round(score(T, file_1967()), 3))
alt = {'F1': ['ep-17','ep-18','ep-14','ep-24'], 'F2': ['ep-22','ep-07','ep-28','ep-21'], 'F3': ['ep-05','ep-15','ep-27','ep-11']}
T2 = truth_H(None, alt)
print('H fausse (variantes parmi les "deguises" T 0,8-0,9) : fichier H =', round(score(T2, file_H(var)), 3), '; 1967 =', round(score(T2, file_1967()), 3))

# 4) Les envois passes, rejoues sous H (toutes paires ; le public n'en voit que 30 %).
def file_from(isolated_groups):
    g = {}
    for F in FAM:
        iso = [grp for grp in isolated_groups if grp[0] in FAM[F]]
        flat = [e for grp in iso for e in grp]
        g[F] = [[e for e in FAM[F] if e not in flat]] + iso
    return build(g)
HIST = {'1803 (public 0,10)': [['ep-14'], ['ep-23'], ['ep-07'], ['ep-28'], ['ep-16']],
        '1882 (public 0,40)': [['ep-05'], ['ep-15'], ['ep-22']],
        '1967 (public 0,60)': [['ep-05'], ['ep-15'], ['ep-22'], ['ep-17', 'ep-18']],
        '2176 = 1967+11 (0,60)': [['ep-05'], ['ep-15'], ['ep-22'], ['ep-17', 'ep-18'], ['ep-11']],
        '2383 = 1967+19 (0,60)': [['ep-05'], ['ep-15'], ['ep-22'], ['ep-17', 'ep-18'], ['ep-19']],
        '2491 = 1967+{28,03} (0,60)': [['ep-05'], ['ep-15'], ['ep-22'], ['ep-17', 'ep-18'], ['ep-28', 'ep-03']],
        '2595 = 1967+{09,11} (0,60)': [['ep-05'], ['ep-15'], ['ep-22'], ['ep-17', 'ep-18'], ['ep-09', 'ep-11']]}
print('Envois passes sous H (moyenne sur les 180 structures H) :')
for name, iso in HIST.items():
    v = []
    for p1 in [e for e in CLAIR['F1'] if e not in SURS['F1']]:
        for p2 in [e for e in CLAIR['F2'] if e not in SURS['F2']]:
            for v3 in F3_opts:
                var = {'F1': [e for e in CLAIR['F1'] if e != p1], 'F2': [e for e in CLAIR['F2'] if e != p2], 'F3': list(v3)}
                v.append(score(truth_H(None, var), file_from(iso)))
    print(f'  {name:28s} : {sum(v)/len(v):.3f} (min {min(v):.3f}, max {max(v):.3f})')

# 5) Fichier candidat sous H. Usage :
#    python t2-regard-neuf-hypothese-H.py --csv sortie.csv --base F1=ep-xx F2=ep-yy F3=ep-zz [--clair F1=ep-a,ep-b,...]
#    --base : la base en clair de chaque famille (le non-deguise dont le lecteur jumeau dit le plus "copie") ;
#    --clair : remplace la liste des non-deguises si le tri des deguisements change.
import sys
if '--csv' in sys.argv:
    a = sys.argv; out = a[a.index('--csv') + 1]
    def kv(flag):
        if flag not in a: return {}
        i = a.index(flag) + 1; d = {}
        while i < len(a) and not a[i].startswith('--'):
            k, v = a[i].split('='); d[k] = v.split(','); i += 1
        return d
    clair = dict(CLAIR); clair.update(kv('--clair')); base = kv('--base')
    var = {F: [e for e in clair[F] if e not in base.get(F, [])] for F in FAM}
    pred = file_H(var)
    with open(out, 'w', encoding='utf8') as f:
        f.write('endpoint,family,model\n')
        for e in sorted(pred): f.write(f'{e},{pred[e][0]},{pred[e][1]}\n')
    print('ecrit', out, '; variantes isolees :', var)
