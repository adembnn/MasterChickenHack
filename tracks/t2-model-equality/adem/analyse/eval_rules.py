#!/usr/bin/env python
"""Mesure le décodeur à règles (rules.py) au plan gelé de 100 réponses, sur chaque famille locale. Sans apprentissage.

Sorties :
  A. détection du type (P, S, T) par déguisement vrai
  B. « quel poids se cache derrière un endpoint déguisé ? » : pour chaque endpoint déguisé, l'ancre en clair la plus
     proche parmi les 7 poids ; exactitude stricte et exactitude « grossière » (base, int8 et fp8 confondus)
  C. détection du filigrane par l'écart de biais d'étiquettes (dlp) : AUC copie filigranée de la base vs variante en clair
  D. MCC modèle sur des mondes tirés (ref = structure du corrigé ; mix = 14 endpoints au hasard), par variante de règles
Usage : python eval_rules.py --probes ../probes/probes_gelees.jsonl --plan ../probes/plan.json --world ../../world3 --fams olmo,qwen
"""
import argparse, glob, itertools, json, os, random, statistics
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
from parse import load
from features import build, pair_features
from learn import mcc
from learn2 import kind, ref_world
from rules import decode, SIMS

GROS = {"base": "b", "int8": "b", "fp8": "b"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--world", required=True)
    ap.add_argument("--regime", default="G")
    ap.add_argument("--fams", default="olmo,qwen")
    ap.add_argument("--worlds", type=int, default=200)
    a = ap.parse_args()
    probes = [json.loads(l) for l in open(a.probes, encoding="utf-8") if l.strip()]
    prompts = {p["id"]: p["prompt"] for p in probes}
    plan = json.load(open(a.plan))
    det = {p: 1 for p in plan["deterministe"]}
    det.update({p: 2 for p in plan["ronde0"]})
    samp = dict(plan["echantillonne"])

    def plan_fn(name, d):
        nondet = any(len(set(t.strip() for t in d[p][:2])) > 1 for p in plan["ronde0"] if p in d)
        return (samp if nondet else det), 4

    for fam in a.fams.split(","):
        files = sorted(glob.glob(os.path.join(a.world, f"{fam}_{a.regime}_*.jsonl")))
        kinds, data = load(files, a.probes)
        kinds.update({p["id"]: p["kind"] for p in probes})
        truth = {}
        for f in glob.glob(os.path.join(a.world, f"{fam}_{a.regime}_*_truth.csv")):
            for l in open(f).read().splitlines():
                if l and not l.startswith("endpoint"):
                    ep, _, model, _ = l.split(",")
                    truth[ep] = model
        eps = build(data, kinds, plan=plan_fn, prompts=prompts)
        names = sorted(eps)
        kd = {e: kind(eps[e]) for e in names}
        F = {frozenset((x, y)): pair_features(eps[x], eps[y]) for x, y in itertools.combinations(names, 2)}
        dis = lambda e: "plain" if e.endswith("plain2") else e.split("-")[-1].rstrip("2b")
        wt = lambda e: e.split("-")[1]
        print(f"\n================ famille {fam} : {len(names)} endpoints, plan gelé (100 réponses) ================")
        # A
        ok = defaultdict(lambda: [0, 0])
        for e in names:
            want = "T" if dis(e) == "t1" else ("S" if dis(e) in ("sys", "persona") else "P")
            ok[dis(e)][0] += kd[e] == want
            ok[dis(e)][1] += 1
        print("A. type détecté correct : " + "  ".join(f"{k} {v[0]}/{v[1]}" for k, v in sorted(ok.items())))
        # B
        anchors = [e for e in names if e.endswith("-plain")]
        print("B. ancre en clair la plus proche d'un endpoint déguisé (strict / grossier), par statistique :")
        print(f"   {'statistique':12s}" + "".join(f"{d:>16s}" for d in ("sys", "persona", "wm", "t1")))
        for sn, sf in SIMS.items():
            row = ""
            for d in ("sys", "persona", "wm", "t1"):
                s_ok = g_ok = n = 0
                for e in names:
                    if dis(e) != d:
                        continue
                    best = max(anchors, key=lambda o: sf(F[frozenset((e, o))]))
                    n += 1
                    s_ok += wt(best) == wt(e)
                    g_ok += GROS.get(wt(best), wt(best)) == GROS.get(wt(e), wt(e))
                row += f"{s_ok:>6d}/{n:<2d} {g_ok:>3d}/{n:<2d} "
            print(f"   {sn:12s}{row}")
        # C
        base = f"{fam}-base-plain"
        if base in eps:
            for st in ("dlp", "slp"):
                pos = [F[frozenset((base, e))][st] for e in names if wt(e) == "base" and dis(e) == "wm"]
                neg = [F[frozenset((base, e))][st] for e in names if wt(e) != "base" and dis(e) == "plain"]
                if pos and neg:
                    print(f"C. filigrane, {st} contre la base en clair : copies filigranées {[round(x, 2) for x in pos]} ; "
                          f"variantes en clair {[round(x, 2) for x in neg]} ; AUC {roc_auc_score([1] * len(pos) + [0] * len(neg), pos + neg):.2f}")
            print("   identité contre la base en clair : " + "  ".join(
                f"{e[len(fam) + 1:]}={F[frozenset((base, e))]['ident']:.2f}" for e in names if e != base and (dis(e) == "plain" or wt(e) == "base")))
        # D
        wr = random.Random(1)
        variants = {
            "tout isolé": None, "tout fusionné": "all",
            "v3 : identité, seuil 0,57 (divergence 0,43)": dict(tau_pp=0.57, attach_s=False, attach_t=False, detect_w=False, force_p=True),
            "règles : P seuls, S/T/filigrane isolés": dict(attach_s=False, attach_t=False, detect_w=False),
            "règles : + T rattaché au centre": dict(attach_s=False, attach_t=True, detect_w=False),
            "règles : + S rattaché (ident+pref)": dict(attach_s=True, attach_t=True, detect_w=False),
            "règles : + filigrane (slp >= 0,25)": dict(attach_s=True, attach_t=True, detect_w=True),
            "règles complètes, filigrane slp >= 0,15": dict(tau_w=0.15),
            "règles complètes, filigrane dlp >= 0,25": dict(wstat="dlp"),
            "règles complètes, sim fermées": dict(sim="fermées"),
            "règles complètes, sim mixte": dict(sim="mixte"),
            "règles complètes, marge 0,15": dict(margin=0.15),
            "règles complètes, S toujours au centre": dict(margin=9.0),
            "règles complètes, centre voté sur 1 statistique": dict(hub_votes="nn"),
            "règles complètes, sim mixte, centre 1 statistique": dict(hub_votes="nn", sim="mixte"),
            "règles complètes, centre = moyenne sur P et S": dict(hub_votes="PS"),
            "règles complètes, centre = moyenne sur P": dict(hub_votes="P"),
            "règles complètes, sim ident": dict(sim="ident"),
            "plus proche ancre, sans a priori (marge 0, T aussi)": dict(margin=0.0, attach_t="nearest"),
            "règles complètes, tau_pp 0,7": dict(tau_pp=0.7),
            "règles complètes, tau_pp 0,9": dict(tau_pp=0.9),
        }
        print(f"D. {'décodeur':52s} {'MCC ref':>8s} {'[p10 ; p90]':>14s} {'MCC mix':>8s} {'[p10 ; p90]':>14s} {'MCC tout':>9s}")
        worlds = [(ref_world(fam, wr), wr.sample(names, min(14, len(names)))) for _ in range(a.worlds)]
        hub_ok = []
        for ref, _ in worlds:
            world = [e for e in ref if e in eps]
            _, info = decode(world, {pr: f for pr, f in F.items() if all(x in world for x in pr)}, kd)
            hub_ok.append(info["hub"] == f"{fam}-base-plain")
        print(f"   (groupe central = base en clair dans {sum(hub_ok)}/{len(hub_ok)} mondes ref)")
        for vn, kw in variants.items():
            mr, mm = [], []
            for ref, mix in worlds:
                for world, acc in ((ref, mr), (mix, mm)):
                    world = [e for e in world if e in eps]
                    if kw is None:
                        lab = {e: i for i, e in enumerate(world)}
                    elif kw == "all":
                        lab = {e: 0 for e in world}
                    else:
                        kw2 = dict(kw)
                        kk = {e: "P" for e in world} if kw2.pop("force_p", False) else kd
                        Fw = {pr: f for pr, f in F.items() if all(x in world for x in pr)}
                        lab, _ = decode(world, Fw, kk, **kw2)
                    acc.append(mcc(truth, lab, world))
            if kw is None:
                lab = {e: i for i, e in enumerate(names)}
            elif kw == "all":
                lab = {e: 0 for e in names}
            else:
                kw2 = dict(kw)
                lab, _ = decode(names, F, {e: "P" for e in names} if kw2.pop("force_p", False) else kd, **kw2)
            mt = mcc(truth, lab, names)
            q = lambda v: f"[{np.percentile(v, 10):.2f} ; {np.percentile(v, 90):.2f}]"
            print(f"   {vn:52s} {np.mean(mr):8.3f} {q(mr):>14s} {np.mean(mm):8.3f} {q(mm):>14s} {mt:9.3f}")


if __name__ == "__main__":
    main()
