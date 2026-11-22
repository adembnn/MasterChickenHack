"""T2 gabarit : la lecture G7 (7 copies + 4 variantes par famille) rejoue-t-elle les 11 scores si l'on ajoute 1 ou 2 variantes ?
Base G7 : variantes 17 18 12 24 (F1, X = 01 copie) | 03 22 26 28 (F2) | 05 09 11 15 (F3). Option : 17 et 18 meme modele.
Public : comme t2-gabarit-public-enum.py. Usage : python t2-gabarit-ajout-variantes.py"""
import itertools, importlib.util
s = importlib.util.spec_from_file_location("e", __file__.replace("ajout-variantes", "public-enum")); e = importlib.util.module_from_spec(s); s.loader.exec_module(e); p = e.p
BASE = "17 18 12 24 03 22 26 28 05 09 11 15".split()
COP_SURES = set("04 21 32 07 14 23".split())
CAND = sorted({x for f in p.FAM for x in f} - set(BASE) - COP_SURES)
def ncompat(tr):
    n = 0
    for k in range(len(e.IND) + 1):
        for S in itertools.combinations(e.IND, k):
            pub = e.SURS | set(S)
            if all(abs(0.1 + 0.9 * max(0, p.mcc(tr, p.lab_sub(i, g), pub)) - o) < 0.05 for _, i, g, o in p.SUBS): n += 1
    return n
for same in (False, True):
    for k in (0, 1, 2):
        for add in itertools.combinations(CAND, k):
            tr = p.truth_from(set(BASE) | set(add))
            if same: tr["18"] = tr["17"]
            n = ncompat(tr)
            if n: print(f"17=18 {same}  variantes ajoutees {add} : {n} ensembles publics compatibles")
print("fin")
