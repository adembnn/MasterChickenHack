#!/usr/bin/env python
"""Calage du banc sur les deux scores publics reels (regles soumises 1803 = 0,10 et 1882 = 0,40), puis leviers.
Hors ligne (numpy) : lit les sommes par reponse de feats.py (relecture des guichets jumeaux sous la base et sous des
lecteurs), assemble au hasard des familles de 11 guichets selon une structure donnee, applique les regles et calcule
le score 0,1*MCC_famille + 0,9*MCC_modele (familles supposees justes : les deux regles reelles commencent par elles).

Monde verifie (session principale) : 3 familles x 11, tirage pur a T~0,7, copies chaudes 0,8-0,9, une copie T=1,0
par famille, fp8 per-tensor, 25 sondes du bloc 0 (probes_v4.plan, 256 tokens).

Regles reproduites :
  1803 : isoler les guichets dont T* >= 0,9 (T* : temperature ou E[-log p_T] = H_T sous la base) ;
  1882 : isoler les guichets fp8 per-tensor, les deux lecteurs concordants (e4m3 poids seuls ET W8A8), z robuste
         (mediane par prompt des autres guichets de la famille, retrait iteratif des |z|>3) et z/sqrt(n) >= 0,51.
Leviers : seuils fp8, un seul lecteur, lecteurs LoRA (temperature de lecture la plus proche de T*), indice de
longueur, budget 25/50/75/100.

Usage : python calage.py --feats feats --worlds llama1b_train_s2,llama1b_val_s101,qwen1.5_val_s101 --draws 300
"""
import argparse
import itertools
import json
import math
import os
import random
import statistics

import numpy as np

TEMPS = [0.6, 0.7, 0.8, 0.9, 1.0, 1.2]


def mcc(tp, fp, fn, tn):
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / d if d else 0.0


class Guichet:
    def __init__(self, meta, X, ri, ci):
        self.meta, self.X, self.ri, self.ci = meta, X, ri, ci
        v = meta["variant"]
        self.kind = "base" if v == "base" else ("fp8" if v == "fp8" else v.split(":")[0] if not v.startswith("lora:") else
                                               "lora@" + v.split("@")[1])
        self.lora_name = v[5:].split("@")[0] if v.startswith("lora:") else None
        self.T = meta["disguise"]["T"]
        self.dkind = meta["disguise"]["kind"]

    def rows(self, b):
        return self.ri < b

    def tstar(self, b):
        m = self.rows(b)
        n = self.X[m, self.ci["n"]].sum()
        g = [(self.X[m, self.ci[f"lp_{T}"]].sum() + self.X[m, self.ci[f"H_{T}"]].sum()) / max(n, 1) for T in TEMPS]
        # g(T) = E[log p_T + H_T] ; nul a la bonne temperature, decroissant en T
        for (t0, g0), (t1, g1) in zip(zip(TEMPS, g), zip(TEMPS[1:], g[1:])):
            if (g0 >= 0 >= g1) or (g0 <= 0 <= g1):
                return t0 + (t1 - t0) * g0 / (g0 - g1) if g0 != g1 else t0
        return TEMPS[0] if g[0] < 0 else TEMPS[-1]

    def per_slot(self, col, b):
        m = self.rows(b)
        return {int(s): float(v) for s, v in zip(self.ri[m], self.X[m, self.ci[col]])}

    def mean_len(self, b):
        m = self.rows(b)
        return float(self.X[m, self.ci["n"]].mean())


def load_world(feats_dir, name):
    z = np.load(os.path.join(feats_dir, f"{name}.npz"))
    meta = json.load(open(os.path.join(feats_dir, f"{name}.meta.json")))
    ci = {c: i for i, c in enumerate(z["cols"].tolist())}
    out = []
    for gi, m in enumerate(meta):
        sel = z["gi"] == gi
        if sel.sum() == 0:
            continue
        out.append(Guichet(m, z["X"][sel], z["ri"][sel], ci))
    return out


def robust_z(S, eps, z_out=3.0, iters=3):
    ref = list(eps)
    z = {}
    for _ in range(iters):
        z = {}
        for ep in eps:
            pool = [q for q in ref if q != ep]
            common = [p for p in S[ep] if all(p in S[q] for q in pool)]
            if len(pool) < 2 or len(common) < 3:
                z[ep] = (None, 0)
                continue
            d = [S[ep][p] - statistics.median(S[q][p] for q in pool) for p in common]
            mu = sum(d) / len(d)
            sd = math.sqrt(sum((x - mu) ** 2 for x in d) / (len(d) - 1))
            z[ep] = (mu / (sd / math.sqrt(len(d))) if sd > 0 else None, len(d))
        new_ref = [e for e in eps if z[e][0] is None or abs(z[e][0]) <= z_out]
        if new_ref == ref or len(new_ref) < 3:
            break
        ref = new_ref
    return {e: ((v / math.sqrt(n)) if (v is not None and n) else 0.0) for e, (v, n) in z.items()}


# ----------------------------------------------------------------------------------------- assemblage
def assemble(pool, structure, rng):
    """structure : (n_copies_hors_T1, variantes). Toujours 1 copie claire + 1 copie T=1,0 ; copies chaudes 0,9 permises."""
    n_copies, variants = structure
    base = [g for g in pool if g.kind == "base"]
    plain = [g for g in base if g.dkind == "plain"]
    t1 = [g for g in base if g.dkind == "temp1.0"]
    others = [g for g in base if g.dkind not in ("plain", "temp1.0") and g.T < 0.95]
    lnames = sorted({g.lora_name for g in pool if g.lora_name})
    lname = rng.choice(lnames) if lnames else None
    fam = [rng.choice(plain), rng.choice(t1)]
    fam += [rng.choice(others) for _ in range(n_copies - 2)]
    for v in variants:
        if v.startswith("lora@"):
            cand = [g for g in pool if g.kind == v and g.lora_name == lname]
        else:
            cand = [g for g in pool if g.kind == v]
        if not cand:
            return None
        fam.append(rng.choice(cand))
    return fam


# ----------------------------------------------------------------------------------------- regles
def rule(fam, b, name, p):
    """Renvoie la liste des indices isoles (chacun devient un modele a part ; le reste = 1 modele)."""
    idx = list(range(len(fam)))
    iso = set()
    if name == "1803" or p.get("tstar"):
        iso |= {i for i in idx if fam[i].tstar(b) >= p.get("t_hot", 0.9)}
    if name in ("1882", "lever"):
        z = {}
        for rd in (["fp8", "fp8wa"] if p.get("both", True) else ["fp8"]):
            S = {i: fam[i].per_slot(f"{rd}_d_1.0", b) for i in idx}
            z[rd] = robust_z(S, idx)
        t = p.get("t_fp8", 0.51)
        iso |= {i for i in idx if all(z[rd][i] >= t for rd in z)}
    if p.get("lora"):
        best = {}
        for rd in ("alpaca_r16", "stories_r32", "gsm_r16"):
            S = {}
            for i in idx:
                T = 0.8 if fam[i].tstar(b) < 0.9 else 1.0
                S[i] = fam[i].per_slot(f"{rd}_d_{T}", b)
            zr = robust_z(S, idx)
            for i in idx:
                best[i] = max(best.get(i, -9), zr[i])
        iso |= {i for i in idx if best[i] >= p.get("t_lora", 0.7)}
    if p.get("length"):
        L = [fam[i].mean_len(b) for i in idx]
        med = statistics.median(L)
        top = max(idx, key=lambda i: L[i])
        if L[top] > p.get("len_ratio", 1.3) * med:
            iso.add(top)
    return iso


def score(families, isos):
    T = [0, 0, 0, 0]
    detail = {}
    for fam, iso in zip(families, isos):
        truth = [("S" if g.kind == "base" else g.kind) for g in fam]
        pred = [("I%d" % i if i in iso else "S") for i in range(len(fam))]
        for i, j in itertools.combinations(range(len(fam)), 2):
            t, p = truth[i] == truth[j], pred[i] == pred[j]
            T[0 if t and p else 3 if (not t and not p) else 1 if p else 2] += 1
        for i, g in enumerate(fam):
            key = g.kind if g.kind != "base" else ("copie T1" if g.dkind == "temp1.0" else "copie")
            d = detail.setdefault(key, [0, 0])
            d[0] += i in iso
            d[1] += 1
    return 0.1 + 0.9 * max(0.0, mcc(*T)), detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--worlds", required=True, help="un monde par famille, repetes si moins de 3")
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--budgets", default="25,50")
    ap.add_argument("--out", default="calage_results.json")
    a = ap.parse_args()
    worlds = [load_world(a.feats, w) for w in a.worlds.split(",")]
    while len(worlds) < 3:
        worlds.append(worlds[len(worlds) % len(worlds)])
    budgets = [int(x) for x in a.budgets.split(",")]
    structures = {   # compatibles avec les scores publics reels (etape 1)
        "7 copies + fp8, LoRA 0,3/0,7/1,5": (7, ["fp8", "lora@0.3", "lora@0.7", "lora@1.5"]),
        "7 copies + fp8, int8, LoRA 0,3/0,7": (7, ["fp8", "int8", "lora@0.3", "lora@0.7"]),
        "6 copies + fp8, int8, LoRA 0,3/0,7/1,5": (6, ["fp8", "int8", "lora@0.3", "lora@0.7", "lora@1.5"]),
    }
    rules = {
        "regle actuelle (1882 : fp8, 2 lecteurs, 0,51)": ("1882", {}),
        "fp8 seuil 0,35": ("lever", {"t_fp8": 0.35}),
        "fp8 seuil 0,7": ("lever", {"t_fp8": 0.7}),
        "fp8 1 seul lecteur": ("lever", {"both": False}),
        "+ LoRA (lecture a T proche de T*) seuil 0,7": ("lever", {"lora": True, "t_lora": 0.7}),
        "+ LoRA seuil 1,0": ("lever", {"lora": True, "t_lora": 1.0}),
        "+ LoRA seuil 1,5": ("lever", {"lora": True, "t_lora": 1.5}),
        "+ longueur (le plus long > 1,3 x mediane)": ("lever", {"length": True}),
        "1803 (T* >= 0,9)": ("1803", {}),
    }
    base_key = ("regle actuelle (1882 : fp8, 2 lecteurs, 0,51)", 25)
    out = {}
    for sname, st in structures.items():
        r2 = random.Random(12345)
        scores = {}
        det = {}
        for _ in range(a.draws):
            fams = [assemble(w, st, r2) for w in worlds]
            if any(f is None for f in fams):
                continue
            for rname, (rn, p) in rules.items():
                for b in budgets:
                    isos = [rule(f, b, rn, p) for f in fams]
                    sc, d = score(fams, isos)
                    scores.setdefault((rname, b), []).append(sc)
                    dd = det.setdefault((rname, b), {})
                    for k, (x, n) in d.items():
                        dd.setdefault(k, [0, 0])
                        dd[k][0] += x
                        dd[k][1] += n
        base = scores[base_key]
        print(f"\n=== {sname} ({len(base)} tirages de 3 familles) ; regle actuelle a 25 : {statistics.mean(base):.3f}")
        for (rname, b), v in scores.items():
            gain = [x - y for x, y in zip(v, base)]
            rates = {k: round(x / n, 2) for k, (x, n) in det[(rname, b)].items()}
            out[f"{sname} | {rname} | b={b}"] = {"score": statistics.mean(v), "gain": statistics.mean(gain),
                                                 "gain_sd": statistics.pstdev(gain), "isolation_rate": rates}
            print(f"  b={b:3d} {rname:46s} score {statistics.mean(v):.3f}  gain {statistics.mean(gain):+.3f} ±{statistics.pstdev(gain):.3f}"
                  f"  | isole : " + ", ".join(f"{k} {x:.2f}" for k, x in sorted(rates.items())), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
