#!/usr/bin/env python
"""Banc de jugement : tableau précision / rappel / AUC par famille, seuils fixés sur les AUTRES familles (leave-one-family-out).
python evalb.py --n 25 [--ctx auto]
Positifs = LoRA (toutes forces) ; négatifs = copies de base (tous déguisements) ; quantifiés rapportés à part.
Précision « ep » = précision attendue à la composition d'une famille ep : 3 LoRA (0,3/0,7/1,5) pour 7 copies.
"""
import argparse, glob, json, os
import numpy as np
from detect import world_table

FAM_OF = lambda w: ("ref" if w.startswith("ref") else w.split("_")[0])


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    pos, neg = np.array(pos), np.array(neg)
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


DETECTORS = {
    "LR": lambda r: r["LR"],
    "relLR": lambda r: r["rel_LR"],
    "-z0": lambda r: -r["z0"],
    "dT": lambda r: r["dT"],
    "LRnw": lambda r: r["LR"] if r["wm_gain"] <= 2 else 0.0,   # LR, annulé si le modèle filigrane explique l'écart
    "LRw": lambda r: r["LRw"],
    "LR+len": lambda r: r["LR"] / max(1.0, r["ntok"] / 3000),   # LR croît avec le nombre de tokens
}


def logistic_fit(X, y, l2=1.0, it=500):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(it):
        p = 1 / (1 + np.exp(-(Xs @ w + b)))
        g = Xs.T @ (p - y) / len(y) + l2 * w / len(y)
        w -= 0.5 * g; b -= 0.5 * (p - y).mean()
    return lambda Z: ((Z - mu) / sd) @ w + b


SUPF = ["LR", "LRw", "wm_gain", "z0", "z1", "z2", "dT", "resp_var", "top1", "rank10", "rel_LR", "rel_z0", "rel_dT"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--ctx", default="auto")
    ap.add_argument("--feats", default="feats")
    ap.add_argument("--fp_q", type=float, default=1.0, help="quantile des copies d'entraînement pour le seuil (1,0 = max)")
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rows = []
    for p in sorted(glob.glob(os.path.join(a.feats, "*.npz"))):
        if os.path.basename(p).startswith("ep_"):
            continue
        rows += world_table(p, a.n, a.ctx)
    for r in rows:
        r["fam"] = FAM_OF(r["world"])
    fams = sorted({r["fam"] for r in rows})
    # détecteur supervisé (d) : appris sur les autres familles
    X = np.array([[r[k] for k in SUPF] for r in rows]); y = np.array([r["lab"].startswith("lora") for r in rows], float)
    lab_ok = np.array([r["lab"] in ("copy",) or r["lab"].startswith("lora") for r in rows])
    for f in fams:
        tr = np.array([r["fam"] != f for r in rows]) & lab_ok
        sc = logistic_fit(X[tr], y[tr])
        for i, r in enumerate(rows):
            if r["fam"] == f:
                r["sup"] = float(sc(X[i:i + 1])[0])
    dets = dict(DETECTORS, sup=lambda r: r["sup"])
    report = {}
    for dn, fn in dets.items():
        print(f"\n=== détecteur {dn}  (N={a.n}, ctx={a.ctx})")
        print("famille   AUC   seuil  rappel0.3 rappel0.7 rappel1.5  FP/copies  FPchaudes/déguisées  quant_isolés  préc_ep")
        report[dn] = {}
        for f in fams:
            te = [r for r in rows if r["fam"] == f]
            trc = [fn(r) for r in rows if r["fam"] != f and r["lab"] == "copy"]
            thr = float(np.quantile(trc, a.fp_q)) * a.margin if trc else float("inf")
            if dn == "sup":
                thr = max(thr, 0.0)
            pos = [fn(r) for r in te if r["lab"].startswith("lora")]
            neg = [fn(r) for r in te if r["lab"] == "copy"]
            rec = {}
            for s in ("0.3", "0.7", "1.5"):
                v = [fn(r) > thr for r in te if r["lab"] == "lora" + s]
                rec[s] = (sum(v), len(v))
            fps = [r for r in te if r["lab"] == "copy" and fn(r) > thr]
            q = [fn(r) > thr for r in te if r["lab"] == "quant"]
            tpr = {s: (rec[s][0] / rec[s][1] if rec[s][1] else 0) for s in rec}
            fpr = len(fps) / max(1, len(neg))
            tp_ep = sum(tpr.values()); prec = tp_ep / (tp_ep + 7 * fpr) if tp_ep + fpr > 0 else float("nan")
            fpk = ",".join(sorted(f"{r['kind']}@{r['Tgen']}" for r in fps))
            print(f"{f:9s} {auc(pos, neg):.3f} {thr:7.2f}  " + "  ".join(f"{rec[s][0]:2d}/{rec[s][1]:<5d}" for s in rec)
                  + f"   {len(fps)}/{len(neg):<5d}  {fpk[:60]:60s} {sum(q)}/{len(q)}  {prec:.2f}")
            report[dn][f] = dict(auc=auc(pos, neg), thr=thr, rec=rec, fp=len(fps), ncopy=len(neg), fp_kinds=fpk,
                                 quant=(sum(q), len(q)), prec_ep=prec)
    if a.out:
        json.dump({"args": vars(a), "report": report,
                   "rows": [{k: v for k, v in r.items()} for r in rows]}, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
