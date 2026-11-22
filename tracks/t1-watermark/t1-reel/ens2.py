"""Moyenne par population : un jeu de cartes pour les images prédites « pleines » (plus de la moitié des cases filigranées selon la moyenne du jeu « pleines »), un autre pour les « partielles ».
La population est décidée par la prédiction, jamais par la vérité : la même règle s'applique au test.
Usage : python ens2.py <methode> --pleines d1 d2 ... --partielles d3 ... [--out DOSSIER]"""
import sys, numpy as np
from pathlib import Path
import ens
m = sys.argv[1]; a = sys.argv[2:]; out = a[a.index("--out") + 1] if "--out" in a else None; a = a[:a.index("--out")] if "--out" in a else a
ip, iq = a.index("--pleines"), a.index("--partielles"); DP = a[ip + 1:iq] if ip < iq else a[ip + 1:]; DQ = a[iq + 1:ip] if iq < ip else a[iq + 1:]
def build(s):
    names = None; LP, LQ = [], []
    for d in DP: n, l = ens.load(d, m, s); assert names is None or n == names; names = n; LP.append(l)
    for d in DQ: n, l = ens.load(d, m, s); assert n == names; LQ.append(l)
    AP, AQ = np.mean(LP, 0), np.mean(LQ, 0); g = (AP > 0).reshape(len(AP), -1).mean(1) > 0.5; return names, np.where(g[:, None, None], AP, AQ), g
names, A, g = build("train"); res = np.array([ens.pos[n] % 5 == 0 for n in names]); fu = np.array([ens.full[n] for n in names])
sc = np.array([ens.miou(x > 0, ens.Y[n]) for x, n in zip(A, names)]); h = np.array([ens.iou0(x > 0, ens.Y[n]) for x, n in zip(A, names)])
print(f"pleines = {' + '.join(Path(d).parent.name if Path(d).name == m else Path(d).name for d in DP)} ; partielles = {' + '.join(Path(d).parent.name if Path(d).name == m else Path(d).name for d in DQ)}")
print(f"   population bien devinée : {np.mean(g == fu):.3f} | CV {sc[~res].mean():.3f} (pleines {sc[~res & fu].mean():.3f}, partielles {sc[~res & ~fu].mean():.3f}) | réserve {sc[res].mean():.3f} (pleines {sc[res & fu].mean():.3f}, partielles {sc[res & ~fu].mean():.3f}) | trou des pleines {h[~res & fu].mean():.3f} / {h[res & fu].mean():.3f}")
if out:
    Path(out).mkdir(parents=True, exist_ok=True)
    for s in ("train", "test"): n, A, g = build(s); np.savez_compressed(Path(out) / f"pred_{s}_{m}.npz", names=np.array(n), logits=A.astype(np.float16)); print("écrit", out, s, "| part d'images prédites pleines :", round(float(g.mean()), 3))
