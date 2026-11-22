#!/usr/bin/env python
"""Mesure en régime « tout échantillonné » (monde local, T = 0,7 par défaut, endpoint t1 à T = 1,0), au budget 100.

Plan par endpoint : 12 sondes meta x 1 + N sondes fermées x R répétitions (N x R = 88).
Sorties : A. z, D et accord des modes contre la base en clair ; B. détection de l'endpoint chaud par self ;
C. AUC « mêmes poids » par type de paire ; D. MCC sur des mondes de structure corrigé, par décodeur.
Usage : python eval_S.py --probes ../probes/probes_S.jsonl --world ../../worldS --fams olmo,qwen
"""
import argparse, glob, itertools, json, os, random, statistics
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
from parse import load
from features import build
from learn import mcc
from learn2 import ref_world
from rules import linkage
import sampled


def decode_S(eps, C, kinds, selfr, zcut=3.0, hot_gap=0.06, attach=True, hub="z"):
    """P' (échantillonnés sans prompt système ni excès de température) regroupés si z < zcut (lien complet sur z).
    Groupe central = l'endpoint P' de z moyen le plus bas avec les autres. S et chauds rattachés au centre."""
    ps = [e for e in eps if kinds[e] == "P"]
    lab, nxt = {}, 0
    if len(ps) >= 2:
        S = {frozenset((a, b)): -C[frozenset((a, b))]["z"] for a, b in itertools.combinations(ps, 2)}
        cl = [[e] for e in ps]
        while len(cl) > 1:  # lien complet : on fusionne tant que le pire z entre deux groupes reste sous zcut
            best, bi = None, None
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    w = max(-S[frozenset((a, b))] for a in cl[i] for b in cl[j])
                    if best is None or w < best:
                        best, bi = w, (i, j)
            if best >= zcut:
                break
            i, j = bi
            cl[i] += cl[j]
            del cl[j]
        lab = {e: k for k, c in enumerate(cl) for e in c}
    elif ps:
        lab = {ps[0]: 0}
    nxt = (max(lab.values()) + 1) if lab else 0
    hub_ep = None
    if ps:
        if hub == "z":
            hub_ep = min(ps, key=lambda e: statistics.mean([C[frozenset((e, o))]["z"] for o in eps if o != e]))
        else:
            hub_ep = max(ps, key=lambda e: statistics.mean([C[frozenset((e, o))]["mode"] for o in eps if o != e]))
    for e in eps:
        if kinds[e] == "P":
            continue
        if attach and hub_ep is not None:
            lab[e] = lab[hub_ep]
        else:
            lab[e] = nxt
            nxt += 1
    return lab, hub_ep


def decode_star(eps, Z, sys_flag, zcut=3.0, attach=True):
    """Décodeur du régime échantillonné. Z[paire] = plus grand z parmi les tests (réponses fermées, textes libres,
    longueur). Les endpoints sans prompt système détecté (P') sont comparés au centre (l'endpoint P' de z médian le plus
    bas) : z < zcut -> même modèle que le centre, sinon modèle à part. Les endpoints S vont au centre."""
    ps = [e for e in eps if not sys_flag[e]]
    lab = {}
    if not ps:
        return {e: i for i, e in enumerate(eps)}, None
    hub = min(ps, key=lambda e: statistics.median([Z[frozenset((e, o))] for o in ps if o != e] or [0]))
    nxt = 1
    for e in eps:
        if e == hub or (sys_flag[e] and attach) or (not sys_flag[e] and Z[frozenset((e, hub))] < zcut):
            lab[e] = 0
        else:
            lab[e] = nxt
            nxt += 1
    return lab, hub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", required=True)
    ap.add_argument("--world", required=True)
    ap.add_argument("--fams", default="olmo,qwen")
    ap.add_argument("--worlds", type=int, default=200)
    a = ap.parse_args()
    probes = [json.loads(l) for l in open(a.probes, encoding="utf-8") if l.strip()]
    prompts = {p["id"]: p["prompt"] for p in probes}
    closed = [p["id"] for p in probes if p["kind"] in ("ab5", "nb5", "lp", "word", "q1")]
    meta = [p["id"] for p in probes if p["kind"] == "meta"]
    by = defaultdict(list)
    for p in probes:
        by[p["kind"]].append(p["id"])
    plans = {
        "17 fermées x 4 (6 ab5, 5 nb5, 6 lp) + 16 libres x 2": ({**{p: 4 for p in by["ab5"][:6] + by["nb5"][:5] + by["lp"][:6]}, **{p: 2 for p in by["open"][:4]}, "__meta2__": 2}),
        "22 sondes x 4 (8 ab5, 6 nb5, 8 lp)": ({**{p: 4 for p in by["ab5"][:8] + by["nb5"][:6] + by["lp"][:8]}}),
        "11 sondes x 8 (4 ab5, 3 nb5, 4 lp)": ({**{p: 8 for p in by["ab5"][:4] + by["nb5"][:3] + by["lp"][:4]}}),
        "14 sondes x 6 (6 ab5, 4 nb5, 4 lp) + 4 q1 x 1": ({**{p: 6 for p in by["ab5"][:6] + by["nb5"][:4] + by["lp"][:4]}, **{p: 1 for p in by["q1"][:4]}}),
    }
    dis = lambda e: "plain" if e.endswith("plain2") else e.split("-")[-1].rstrip("2b")
    wt = lambda e: e.split("-")[1]
    for fam in a.fams.split(","):
        files = sorted(glob.glob(os.path.join(a.world, f"{fam}_S_*.jsonl")))
        kinds, data = load(files, a.probes)
        truth = {}
        for f in glob.glob(os.path.join(a.world, f"{fam}_S_*_truth.csv")):
            for l in open(f).read().splitlines():
                if l and not l.startswith("endpoint"):
                    ep, _, model, _ = l.split(",")
                    truth[ep] = model
        for pname, plan in plans.items():
            only = dict(plan)
            mrep = only.pop("__meta2__", 1)
            only.update({p: mrep for p in meta})
            assert sum(only.values()) <= 100, sum(only.values())
            eps = build(data, kinds, only=only, prompts=prompts)
            names = sorted(eps)
            A = {e: sampled.answers(eps[e]) for e in names}
            selfr = {e: sampled.self_rate(A[e]) for e in names}
            C = {frozenset((x, y)): sampled.compare(A[x], A[y]) for x, y in itertools.combinations(names, 2)}
            med = statistics.median(selfr.values())
            print(f"\n======== famille {fam}, régime échantillonné, plan : {pname} ({sum(only.values())} réponses) ========")
            base = f"{fam}-base-plain"
            if base in eps:
                print("A. contre la base en clair (z ; accord des modes) : " + "  ".join(
                    f"{e[len(fam) + 1:]}={C[frozenset((base, e))]['z']:.1f};{C[frozenset((base, e))]['mode']:.2f}"
                    for e in names if e != base and (wt(e) == "base" or dis(e) == "plain")))
            hot = [selfr[e] for e in names if dis(e) == "t1"]
            cold = [selfr[e] for e in names if dis(e) != "t1"]
            print(f"B. self : médiane famille {med:.3f} ; endpoints T=1,0 {np.mean(hot):.3f} [{min(hot):.3f} ; {max(hot):.3f}] ; "
                  f"autres {np.mean(cold):.3f} [{min(cold):.3f} ; {max(cold):.3f}] ; AUC chaud "
                  f"{roc_auc_score([1] * len(hot) + [0] * len(cold), [-x for x in hot] + [-x for x in cold]):.2f}")
            sysok = sum(eps[e].sys == (dis(e) in ("sys", "persona")) for e in names)
            print(f"   prompt système détecté correctement : {sysok}/{len(names)}")
            rows = defaultdict(lambda: ([], [], []))
            for pr, c in C.items():
                x, y = sorted(pr)
                k = "/".join(sorted((dis(x), dis(y))))
                rows[k][0].append(-c["z"]); rows[k][1].append(c["mode"]); rows[k][2].append(int(truth[x] == truth[y]))
            print("C. AUC « mêmes poids » par type de paire (z ; modes) : " + "  ".join(
                f"{k}={roc_auc_score(v[2], v[0]):.2f};{roc_auc_score(v[2], v[1]):.2f}" for k, v in sorted(rows.items())
                if 0 < sum(v[2]) < len(v[2])))
            # types : S par fuite ; chaud si self < médiane - 0,06 ; sinon P
            def kd_fn(world, gap=0.06):
                m = statistics.median(selfr[e] for e in world)
                return {e: ("S" if eps[e].sys else ("H" if selfr[e] < m - gap else "P")) for e in world}
            wr = random.Random(1)
            worlds = [[e for e in ref_world(fam, wr) if e in eps] for _ in range(a.worlds)]
            T = {pr: sampled.compare_text(eps[min(pr)].texts, eps[max(pr)].texts, kinds) for pr in C}
            nz = lambda v: -9.0 if v != v else v
            Zc = {pr: nz(C[pr]["z"]) for pr in C}
            Za = {pr: max(nz(C[pr]["z"]), nz(T[pr]["zt"]), nz(T[pr]["zlen"])) for pr in C}
            sysf = {e: eps[e].sys for e in names}
            print(f"E. {'décodeur en étoile':44s} {'MCC ref':>8s} {'[p10 ; p90]':>14s}   centre = base")
            for vn, (Z, zc, att) in {"fermées seules, z < 3": (Zc, 3.0, True), "fermées + textes + longueur, z < 3": (Za, 3.0, True),
                                     "fermées + textes + longueur, z < 2,5": (Za, 2.5, True), "fermées + textes + longueur, z < 4": (Za, 4.0, True),
                                     "idem z < 3, S isolés": (Za, 3.0, False)}.items():
                ms, hubs = [], []
                for w in worlds:
                    lab, h = decode_star(w, Z, sysf, zc, att)
                    ms.append(mcc(truth, lab, w)); hubs.append(h == base)
                print(f"   {vn:44s} {np.mean(ms):8.3f} [{np.percentile(ms, 10):.2f} ; {np.percentile(ms, 90):.2f}]   {sum(hubs)}/{len(hubs)}")
            # Décodeur « déguisés ensemble » : sous l'hypothèse du corrigé, tout endpoint déguisé sert la base. On
            # fusionne donc les endpoints à prompt système détecté (et, en option, les endpoints « chauds » : taux de
            # collision interne inférieur à la médiane du monde moins `gap`), et on isole tous les autres.
            print(f"F. {'décodeur « déguisés ensemble »':44s} {'MCC ref':>8s} {'[p10 ; p90]':>14s}")
            for vn, gap in {"S fusionnés, reste isolé": None, "S + chauds (self < médiane - 0,06)": 0.06,
                            "S + chauds (self < médiane - 0,10)": 0.10, "S + chauds (self < médiane - 0,03)": 0.03}.items():
                ms = []
                for w in worlds:
                    m = statistics.median(selfr[e] for e in w)
                    lab, nxt = {}, 1
                    for e in w:
                        if sysf[e] or (gap is not None and selfr[e] < m - gap):
                            lab[e] = 0
                        else:
                            lab[e] = nxt
                            nxt += 1
                    ms.append(mcc(truth, lab, w))
                print(f"   {vn:44s} {np.mean(ms):8.3f} [{np.percentile(ms, 10):.2f} ; {np.percentile(ms, 90):.2f}]")
            # Règle « typé -> base » (proposée par t2-critique) : le centre est l'endpoint non typé de z médian le plus
            # bas ; un endpoint typé (prompt système extrait ; option : filigrane par biais d'étiquettes slp >= 0,25 ;
            # option : chaud par taux de collision) prend l'étiquette du centre ; un endpoint non typé rejoint le centre
            # si son z (réponses fermées) est sous zcut ; sinon il est seul.
            def slp(x, y):
                common = [L for L in eps[x].lp if L in eps[y].lp]
                return statistics.pstdev([eps[x].lp[L] - eps[y].lp[L] for L in common]) if len(common) >= 4 else 0.0
            print(f"G. {'règle « typé -> base »':44s} {'MCC ref':>8s} {'[p10 ; p90]':>14s}")
            for vn, (use_w, gap, zc, ZZ) in {"S typé ; fermées z < 3": (False, None, 3.0, Zc), "S + filigrane typés ; fermées z < 3": (True, None, 3.0, Zc),
                                             "S + filigrane typés ; fermées z < 2": (True, None, 2.0, Zc),
                                             "S + filigrane typés ; fermées+textes z < 3": (True, None, 3.0, Za),
                                             "S + filigrane typés ; fermées+textes z < 3,5": (True, None, 3.5, Za),
                                             "S + filigrane + chaud ; fermées+textes z < 3,5": (True, 0.10, 3.5, Za),
                                             "S + filigrane typés ; fermées+textes z < 4,5": (True, None, 4.5, Za)}.items():
                ms = []
                for w in worlds:
                    m = statistics.median(selfr[e] for e in w)
                    typed = {e: sysf[e] or (gap is not None and selfr[e] < m - gap) for e in w}
                    un = [e for e in w if not typed[e]]
                    if not un:
                        ms.append(0.0)
                        continue
                    hub = min(un, key=lambda e: statistics.median([ZZ[frozenset((e, o))] for o in un if o != e] or [0]))
                    lab, nxt = {}, 1
                    for e in w:
                        if e == hub or typed[e] or (use_w and slp(e, hub) >= 0.25) or ZZ[frozenset((e, hub))] < zc:
                            lab[e] = 0
                        else:
                            lab[e] = nxt
                            nxt += 1
                    ms.append(mcc(truth, lab, w))
                print(f"   {vn:44s} {np.mean(ms):8.3f} [{np.percentile(ms, 10):.2f} ; {np.percentile(ms, 90):.2f}]")
            if base in eps:
                print("   z max contre la base en clair : " + "  ".join(f"{e[len(fam) + 1:]}={Za[frozenset((base, e))]:.1f}"
                      for e in names if e != base and (wt(e) == "base" or dis(e) == "plain")))
            variants = {"tout isolé": None,
                        "P' par z < 3, S et chauds isolés": dict(zcut=3.0, attach=False),
                        "P' par z < 3, S et chauds au centre": dict(zcut=3.0),
                        "P' par z < 2, S et chauds au centre": dict(zcut=2.0),
                        "P' par z < 4, S et chauds au centre": dict(zcut=4.0),
                        "idem z < 3, centre par accord des modes": dict(zcut=3.0, hub="mode")}
            print(f"D. {'décodeur':44s} {'MCC ref':>8s} {'[p10 ; p90]':>14s}   centre = base")
            for vn, kw in variants.items():
                ms, hubs = [], []
                for w in worlds:
                    if kw is None:
                        lab, h = {e: i for i, e in enumerate(w)}, None
                    else:
                        lab, h = decode_S(w, C, kd_fn(w), selfr, **kw)
                    ms.append(mcc(truth, lab, w)); hubs.append(h == base)
                print(f"   {vn:44s} {np.mean(ms):8.3f} [{np.percentile(ms, 10):.2f} ; {np.percentile(ms, 90):.2f}]   {sum(hubs)}/{len(hubs)}")


if __name__ == "__main__":
    main()
