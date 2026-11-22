"""À lancer sur la machine d'entraînement depuis ~/eac/t1 (lecture seule) : pour une carte t1-reel, le MÊME modèle de pli prédit ses images hors-pli
(oof.npz) et la réserve (res.npz). Compare score au seuil 0, confiance et score par tranche de difficulté : la réserve est-elle seulement plus dure ?"""
import json, sys
import numpy as np
from pathlib import Path
tag, ME = sys.argv[1], sys.argv[2]; R = Path("t1-reel/runs") / tag / ME; trm = json.load(open("release/train/metadata.json")); z = np.load("release/train/masks_128.npz"); g = {str(n): m.astype(bool) for n, m in zip(z["names"], z["masks"])}
def miou(P, G):
    P = P.reshape(len(P), -1); G = G.reshape(len(G), -1); out = np.zeros(len(P))
    for c in (True, False):
        p = P == c; t = G == c; u = (p | t).sum(1); out += np.where(u > 0, (p & t).sum(1) / np.maximum(u, 1), 1.0)
    return out / 2
def feats(ii):
    G = np.stack([g[trm[i]["image_name"]] for i in ii]); full = np.array([trm[i]["full_watermark"] for i in ii]); rare = np.where(full[:, None, None], ~G, G).reshape(len(ii), -1).mean(1); return G, full, rare
S = {"hors-pli": [], "réserve": []}
for f in range(5):
    for lab, fn in (("hors-pli", "oof.npz"), ("réserve", "res.npz")):
        q = np.load(R / f"fold{f}" / fn); ii = [int(i) for i in q["idx"]]; L = q["logits"].astype(np.float32); G, full, rare = feats(ii); S[lab].append((miou(L > 0, G), np.abs(L).reshape(len(L), -1).mean(1), (np.abs(L) < 1).reshape(len(L), -1).mean(1), full, rare, np.array(ii)))
cat = lambda lab, k: np.concatenate([s[k] for s in S[lab]])
print(f"{tag} {ME} : modèle de pli unique, seuil 0")
for lab in S:
    sc, conf, unc, full, rare = (cat(lab, k) for k in range(5)); print(f"  {lab:9s} n={len(sc):3d} (prédictions) : mIoU {sc.mean():.4f} ; pleines {sc[full].mean():.4f} ({full.mean():.0%}) ; partielles {sc[~full].mean():.4f} ; |logit| moyen {conf.mean():.2f} ; cases incertaines (|logit| < 1) {unc.mean():.1%} ; zone rare < 3 % : {np.mean(rare < .03):.0%} des images")
# réserve : moyenne par image sur les 5 modèles, pour comparer image par image
o = [cat("hors-pli", k) for k in range(5)]; r = [cat("réserve", k) for k in range(5)]; ridx = cat("réserve", 5); uni = np.unique(ridx); rs = np.array([r[0][ridx == i].mean() for i in uni]); rfull = np.array([r[3][ridx == i][0] for i in uni]); rrare = np.array([r[4][ridx == i][0] for i in uni])
print("  par tranche (pleine / partielle x taille de la zone rare) : hors-pli contre réserve (moyenne des 5 modèles uniques)")
tot_o = tot_r = 0.0; wsum = 0
for pop in (True, False):
    for lo, hi in ((0, .03), (.03, .15), (.15, 1.01)):
        ko = (o[3] == pop) & (o[4] >= lo) & (o[4] < hi); kr = (rfull == pop) & (rrare >= lo) & (rrare < hi)
        if ko.sum() and kr.sum():
            print(f"     {'pleines' if pop else 'partielles':10s} zone {lo:.2f}-{min(hi, 1):.2f} : hors-pli {o[0][ko].mean():.3f} (n={ko.sum()}, {ko.mean():.0%}) | réserve {rs[kr].mean():.3f} (n={kr.sum()}, {kr.mean():.0%}) | écart {o[0][ko].mean() - rs[kr].mean():+.3f}")
            w = ko.sum() + kr.sum(); tot_o += w * o[0][ko].mean(); tot_r += w * rs[kr].mean(); wsum += w
rng = np.random.default_rng(0); d = o[0].mean() - rs.mean(); b = np.array([o[0][rng.integers(0, len(o[0]), len(o[0]))].mean() - rs[rng.integers(0, len(rs), len(rs))].mean() for _ in range(10000)])
print(f"  écart brut hors-pli − réserve : {d:+.4f} IC95 [{np.percentile(b, 2.5):+.3f} ; {np.percentile(b, 97.5):+.3f}] ; à composition égale (mêmes poids par tranche) : {(tot_o - tot_r) / wsum:+.4f}")
