"""T2 gabarit : pour une verite donnee, enumere les ensembles publics compatibles avec les 11 scores.
Public = 11 guichets surs (copies 04 21 32 07 14 23 ; variantes 17 18 22 05 15) + un sous-ensemble S des 14 indecis ; hors public fixe : 01 03 09 11 12 19 24 28.
Compatible = chaque score rejoue a moins de 0,05 (scores publies arrondis a 0,1). Usage : python t2-gabarit-public-enum.py"""
import itertools, importlib.util
s = importlib.util.spec_from_file_location("p", __file__.replace("public-enum", "score-public")); p = importlib.util.module_from_spec(s); s.loader.exec_module(p)
SURS = set("04 21 32 07 14 23 17 18 22 05 15".split()); IND = sorted({e for f in p.FAM for e in f} - SURS - p.HORS)
PROP = [("1967+26", p.FP8 + ["26"], [["17", "18"]]), ("1967+{10,13}", p.FP8 + ["10", "13"], [["17", "18"]]), ("1967+25", p.FP8 + ["25"], [["17", "18"]]),
        ("1967+{33,30,31}", p.FP8 + ["33", "30", "31"], [["17", "18"]]), ("1967+{02,08,29}", p.FP8 + ["02", "08", "29"], [["17", "18"]]), ("1967+06", p.FP8 + ["06"], [["17", "18"]]),
        ("1967+{25,06,19}", p.FP8 + ["25", "06", "19"], [["17", "18"]]), ("1967+{17|18}+26", p.FP8 + ["26", "17", "18"], [])]
def enum(name, variants):
    tr = p.truth_from(set(variants)); ok = []
    for k in range(len(IND) + 1):
        for S in itertools.combinations(IND, k):
            pub = SURS | set(S); good = True
            for sid, iso, gr, obs in p.SUBS:
                sc = 0.1 + 0.9 * max(0, p.mcc(tr, p.lab_sub(iso, gr), pub))
                if abs(sc - obs) >= 0.05: good = False; break
            if good: ok.append(S)
    print(f"{name:40} ensembles compatibles : {len(ok)}")
    if ok:
        freq = {e: sum(e in S for S in ok) / len(ok) for e in IND}
        print("   P(public) : " + " ".join(f"{e}:{v:.2f}" for e, v in freq.items()))
        for pn, iso, gr in PROP:
            vals = []
            for S in ok:
                pub = SURS | set(S); base = 0.1 + 0.9 * max(0, p.mcc(tr, p.lab_sub(p.FP8, [["17", "18"]]), pub))
                vals.append(0.1 + 0.9 * max(0, p.mcc(tr, p.lab_sub(iso, gr), pub)) - base)
            up = sum(v > 0.05 for v in vals) / len(vals); dn = sum(v < -0.05 for v in vals) / len(vals)
            print(f"   {pn:20} hausse {up:.2f}  baisse {dn:.2f}  gain moyen {sum(vals)/len(vals):+.3f}")
    return ok
if __name__ == "__main__":
    for x in ("01", "12", "24"):
        enum(f"G7 F1 X={x} ; F2 V=03 22 26 28", ["17", "18"] + [e for e in ("01", "12", "24") if e != x] + ["03", "22", "26", "28", "05", "09", "11", "15"])
    for v2 in (["03", "22", "28", "21"], ["03", "22", "28", "32"], ["03", "22", "28", "06"], ["03","22","26"], ["03","22","28"], ["22","26","28"]):
        enum(f"G F1 V=17 18 12 24 ; F2 V={' '.join(v2)}", ["17", "18", "12", "24"] + v2 + ["05", "09", "11", "15"])
    enum("W4 t2-critique", ["12","24","01","18","22","28","03","21","15","05","11","09"])
