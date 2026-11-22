"""T2 gabarit : z apparie par bloc (bloc 0 | bloc 1) des traits de t2-gabarit-mesures.py, pour tous les guichets ep.
Un trait « stable » garde le meme signe et |z| >= 2 dans les deux blocs. Usage : python t2-gabarit-stabilite.py"""
import importlib.util
s = importlib.util.spec_from_file_location("m", __file__.replace("stabilite", "mesures")); m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
FE = ["rare", "ttr60", "rep3", "len", "fin", "ouv_seule"]
ED = m.ep_data()
for fam in ("F1", "F2", "F3"):
    eps, qs, F = m.features(ED, fam, 0)
    q0 = [q for q in qs if q[0] == 0]; q1 = [q for q in qs if q[0] == 1]
    Z = {f: (m.zpair(F, eps, f, q0), m.zpair(F, eps, f, q1)) for f in FE}
    print(f"== {fam}  (z bloc0 | bloc1 ; * = stable)")
    print("        " + "".join(f"{f:>16}" for f in FE))
    for e in eps:
        cells = []
        for f in FE:
            a, b = Z[f][0][e][1], Z[f][1][e][1]; star = "*" if abs(a) >= 2 and abs(b) >= 2 and a * b > 0 else " "
            cells.append(f"{a:+5.1f}|{b:+5.1f}{star}")
        print(f"{e:7} " + "".join(f"{c:>16}" for c in cells))
