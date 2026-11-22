"""T2 gabarit : rejoue les 11 scores publics sous une verite donnee par le gabarit.
Verite « G7 » : par famille 7 copies (clair, prompt, persona, filigrane, T 1,0, tronque, classe X) + 4 variantes, chacune son modele.
Public suppose = paires entre les 25 guichets hors {01,03,09,11,12,19,24,28}. Score = 0,1 + 0,9 * max(0, MCC modele) (familles justes).
Usage : python t2-gabarit-score-public.py"""
import itertools, math
FAM = [["01","02","04","12","14","17","18","23","24","25","33"], ["03","06","07","08","20","21","22","26","28","30","32"], ["05","09","10","11","13","15","16","19","27","29","31"]]
HORS = set("01 03 09 11 12 19 24 28".split())
FP8 = ["05", "15", "22"]
SUBS = [("1803", ["14","23","07","28","16"], [], .10), ("1882", FP8, [], .40), ("1967", FP8, [["17","18"]], .60), ("2052", FP8 + ["17","18"], [], .60),
        ("2176", FP8 + ["11"], [["17","18"]], .60), ("2383", FP8 + ["11","19"], [["17","18"]], .60), ("2491", FP8 + ["28","03"], [["17","18"]], .60),
        ("2595", FP8 + ["09","11"], [["17","18"]], .60), ("2707", FP8 + ["12","24"], [["17","18"]], .60),
        ("2795", FP8 + ["17","18","04","12","03","21","32","09","11"], [], .30), ("2899", FP8 + ["01"], [["17","18"]], .60)]
def lab_sub(iso, groups):
    lab = {}
    for f in FAM:
        for e in f: lab[e] = "base" + f[0]
    for e in iso: lab[e] = "iso" + e
    for k, g in enumerate(groups):
        for e in g: lab[e] = "g%d" % k
    return lab
def mcc(truth, pred, eps):
    tp = fp = fn = tn = 0
    for f in FAM:
        for a, b in itertools.combinations([e for e in f if e in eps], 2):
            t = truth[a] == truth[b]; p = pred[a] == pred[b]
            tp += t and p; fp += (not t) and p; fn += t and not p; tn += (not t) and not p
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / d if d else 0.0
def truth_from(variants):
    t = {}
    for f in FAM:
        for e in f: t[e] = ("V" + e) if e in variants else ("C" + f[0])
    return t
def run(name, variants, extra=()):
    pub = {e for f in FAM for e in f} - HORS
    tr = truth_from(set(variants)); out = []; bad = 0
    for sid, iso, gr, obs in list(SUBS) + list(extra):
        s = 0.1 + 0.9 * max(0, mcc(tr, lab_sub(iso, gr), pub)); out.append(f"{sid}:{s:.3f}/{obs}")
        if obs is not None and round(s, 1) != obs: bad += 1
    print(f"{name:34} ecarts(arrondi 0,1)={bad}  " + " ".join(out))
    return bad
PROPOSES = [("1967+26", FP8 + ["26"], [["17","18"]], None), ("1967+26 17|18", FP8 + ["26","17","18"], [], None),
            ("tout G7 (X=01)", FP8 + ["17","18","12","24","03","26","28","09","11"], [], None)]
if __name__ == "__main__":
    # G7 : F2 variantes {03,22,26,28}, F3 {05,09,11,15}, F1 {17,18} + 2 parmi {01,12,24}
    for x in ("01", "12", "24"):
        v1 = ["17", "18"] + [e for e in ("01", "12", "24") if e != x]
        run(f"G7 X(F1)={x}", v1 + ["03","22","26","28","05","09","11","15"], PROPOSES)
    print("-- autres lectures (pour comparaison)")
    run("W4 t2-critique", ["12","24","01","18","22","28","03","21","15","05","11","09"], PROPOSES)
    run("G7 mais 26 copie, 21 variante", ["17","18","12","24","03","22","21","28","05","09","11","15"], PROPOSES)
    run("G7 mais 26 copie, 32 variante", ["17","18","12","24","03","22","32","28","05","09","11","15"], PROPOSES)
