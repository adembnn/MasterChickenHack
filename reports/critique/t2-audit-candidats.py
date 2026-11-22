"""Audit T2 : pour chaque envoi candidat (1967 + un changement), loi du score public prevu sous les etats (verite, S)
echantillonnes par t2-audit-mcmc.py. python t2-audit-candidats.py etats.npz [etats2.npz ...]"""
import sys, numpy as np, importlib.util
spec = importlib.util.spec_from_file_location("a", __file__.replace("t2-audit-candidats.py", "t2-audit-score.py"))
a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
FP8, G = a.FP8, a.G
C = {f"+{e}": a.lab(FP8 + [e], G) for e in a.E if e not in FP8 + ["17", "18"]}
C.update({"+10+13": a.lab(FP8 + ["10", "13"], G), "+33+30+31": a.lab(FP8 + ["33", "30", "31"], G),
          "+02+08+29": a.lab(FP8 + ["02", "08", "29"], G), "+23+20+16": a.lab(FP8 + ["23", "20", "16"], G),
          "+14+07+27": a.lab(FP8 + ["14", "07", "27"], G), "+25+26+10": a.lab(FP8 + ["25", "26", "10"], G),
          "+25+26+13": a.lab(FP8 + ["25", "26", "13"], G)})
for f in sys.argv[1:]:
    d = np.load(f); L, S = d["L"], d["S"]; M = (S[:, a.A] & S[:, a.B]).astype(np.float32)
    T = (L[:, a.A] == L[:, a.B]).astype(np.float32)
    print(f"== {f} : {len(L)} etats")
    rows = []
    for nm, l in C.items():
        p = (l[a.A] == l[a.B]).astype(np.float32)
        tp = (M * T) @ p; fp = (M * (1 - T)) @ p; fn = (M * T).sum(1) - tp; tn = (M * (1 - T)).sum(1) - fp
        sc = np.round((0.1 + 0.9 * a.mcc(tp, fp, fn, tn)) / 0.05) * 0.05
        up, dn = (sc > 0.62).mean(), (sc < 0.58).mean(); eq = 1 - up - dn
        ent = -sum(x * np.log2(x) for x in (up, dn, eq) if x > 0)
        rows.append((ent, nm, up, dn, eq, sc[sc > 0.62].mean() if up else 0))
    for ent, nm, up, dn, eq, m in sorted(rows, reverse=True)[:14]:
        print(f"  {nm:12s} hausse {up:.2f} (moy. {m:.2f})  baisse {dn:.2f}  inchange {eq:.2f}  info {ent:.2f} bit")
