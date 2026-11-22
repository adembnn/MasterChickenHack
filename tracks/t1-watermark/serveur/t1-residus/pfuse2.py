"""Fusion apprise « réseau + cartes de re-tamponnage » sur les PLEINES TrustMark : régression logistique par case (numpy seul).
Variables : logit du réseau, z-scores de re-tamponnage à 5 lissages (bruts et recentrés sur la médiane de l'image), log du rapport à 5 lissages, R3 global.
Apprise sur les pleines HORS réserve (et en validation croisée à 5 plis dedans), notée sur la réserve. Images sans toile fiable : réseau seul.
Usage : python pfuse2.py <pred_train.npz> <maps_pz.npz> <maps_pleine1.npz> [<maps_pleine2.npz>]"""
import sys, json, numpy as np
from pathlib import Path
R = Path("E:/EAC-lab/t1/release/train"); m128 = np.load(R / "masks_128.npz"); G = dict(zip(m128["names"].tolist(), m128["masks"]))
def miou(p, g):
    s = 0.0
    for c in (0, 1):
        u = ((p == c) | (g == c)).sum(); s += 1.0 if u == 0 else ((p == c) & (g == c)).sum() / u
    return s / 2
z = np.load(sys.argv[1], allow_pickle=True); NET = dict(zip(z["names"].tolist(), z["logits"].astype(np.float32)))
pz = np.load(sys.argv[2], allow_pickle=True); names = pz["names"].tolist(); Z = pz["z"].astype(np.float32); r3 = pz["r3"]; gidx = pz["gidx"]; RAT = {}
for f in sys.argv[3:]:
    m = np.load(f, allow_pickle=True); RAT.update(dict(zip(m["names"].tolist(), m["ratio"].astype(np.float32))))
N = len(names); g = np.stack([G[n] > 0 for n in names]); L = np.stack([NET[n] for n in names]); res = gidx % 5 == 0; ok = r3 < 0.6
lr = np.log(np.maximum(np.stack([RAT[n] for n in names]), 1e-3)); lr = lr - np.median(lr, axis=(2, 3), keepdims=True)
import os
RMF = os.environ.get("RM", "")
if RMF:
    rm = np.load(RMF, allow_pickle=True); assert rm["names"].tolist() == names; RM = [np.clip(rm[k].astype(np.float32), -20, 20) for k in ("g", "zg", "e")]
feats = lambda i, use_net=True: np.concatenate(([np.clip(L[i], -12, 12)[None]] if use_net else []) + [Z[i], Z[i] - np.median(Z[i], axis=(1, 2), keepdims=True), lr[i], np.full((1, 128, 128), r3[i], np.float32)] + ([q[i] for q in RM] + [RM[1][i] - np.median(RM[1][i], axis=(1, 2), keepdims=True)] if RMF else [])).reshape(-1, 128 * 128).T
def fit(idx, use_net=True, it=300, l2=1e-3):
    X = np.concatenate([feats(i, use_net) for i in idx]); y = np.concatenate([g[i].ravel() for i in idx]).astype(np.float32)
    mu = X.mean(0); sd = X.std(0) + 1e-6; Xs = (X - mu) / sd; w = np.zeros(X.shape[1], np.float32); b = 0.0
    wt = np.where(y > 0, 0.5 / max(y.mean(), 1e-6), 0.5 / max(1 - y.mean(), 1e-6)).astype(np.float32)      # classes équilibrées : les trous comptent autant que le filigrane
    for _ in range(it):
        p = 1 / (1 + np.exp(-(Xs @ w + b))); gr = (p - y) * wt; w -= 1.0 * (Xs.T @ gr / len(y) + l2 * w); b -= 1.0 * gr.mean()
    return mu, sd, w, b
def score(i, mdl, use_net=True):
    mu, sd, w, b = mdl; return (((feats(i, use_net) - mu) / sd) @ w + b).reshape(128, 128)
def best_thr(S, idx):
    ts = np.linspace(-3, 3, 25); v = [np.mean([miou(S[i] > t, g[i]) for i in idx]) for t in ts]; return ts[int(np.argmax(v))]
base = np.array([miou(L[i] > 0, g[i]) for i in range(N)]); tr = np.where(~res & ok)[0]; te = np.where(res)[0]
print("PFUSE2 n=%d pleines (%d en réserve ; toile fiable %d) | réseau seul : hors réserve %.4f, RÉSERVE %.4f" % (N, res.sum(), ok.sum(), base[~res].mean(), base[res].mean()))
for nm, use_net in (("réseau + cartes", True), ("cartes seules", False)):
    mdl = fit(tr, use_net); S = {i: score(i, mdl, use_net) for i in range(N)}; t = best_thr(S, tr)
    s = np.array([miou(S[i] > t, g[i]) if ok[i] else base[i] for i in range(N)]); d = s - base
    # validation croisée à 5 plis sur le hors réserve
    fold = gidx % 5; cv = base.copy()
    for f in (1, 2, 3, 4):
        a_ = np.where(~res & ok & (fold != f))[0]; b_ = np.where(~res & ok & (fold == f))[0]
        if len(b_) == 0: continue
        m2 = fit(a_, use_net); S2 = {i: score(i, m2, use_net) for i in list(a_) + list(b_)}; t2 = best_thr(S2, a_)
        for i in b_: cv[i] = miou(S2[i] > t2, g[i])
    dc = (cv - base)[~res]
    print("  %-16s | hors réserve, validation croisée : %.4f (%+.4f ± %.4f ; améliorées %d, dégradées %d) | RÉSERVE : %.4f (%+.4f ± %.4f ; n=%d ; améliorées %d, dégradées %d) | seuil %.2f" % (
        nm, cv[~res].mean(), dc.mean(), dc.std() / np.sqrt(len(dc)), (dc > 1e-6).sum(), (dc < -1e-6).sum(), s[res].mean(), d[res].mean(), d[res].std() / np.sqrt(res.sum()), res.sum(), (d[res] > 1e-6).sum(), (d[res] < -1e-6).sum(), t))
    if use_net and not RMF: print("     poids (variables centrées-réduites) : réseau %.2f | z %s | z recentré %s | log rapport %s | R3 %.2f" % (mdl[2][0], np.round(mdl[2][1:6], 2), np.round(mdl[2][6:11], 2), np.round(mdl[2][11:16], 2), mdl[2][16]))
