"""Calage SAM sur les partielles : comparaison de règles de choix des masques (prédictions hors-pli)."""
import os, sys, json, itertools, numpy as np
from pathlib import Path
HERE = Path(os.environ["WMLOC_ROOT"]); D = HERE / "adem" / "t1-pixelseal"; sys.path.insert(0, str(HERE / "kit"))
from wmloc.utils import miou_image
pred = sys.argv[1] if len(sys.argv) > 1 else "pred_train_G.npz"; T_PART = float(sys.argv[2]) if len(sys.argv) > 2 else -2.75
z = np.load(D / pred); names = z["names"].tolist(); L = z["logits"].astype(np.float32); S = np.load(D / "sam_train.npz"); role = z["role"] if "role" in z.files else np.zeros(len(names), int)
meta = json.loads((HERE / "release/train/metadata.json").read_text()); idx = {m["image_name"]: i for i, m in enumerate(meta)}
g = np.load(HERE / "release/train/masks_128.npz"); G = dict(zip(g["names"].tolist(), g["masks"])); G = np.stack([G[n] for n in names]).astype(bool)
pop = json.loads((D / "pop_train.json").read_text()); part = np.array([pop[n] <= 0.75 for n in names]); res = np.array([idx[n] % 5 == 0 for n in names])
bx = {r["name"] for r in json.loads((D / "box_train.json").read_text()) if "box" in r and r["bits"] >= 0.85 and r["bits"] - r["glob"] >= 0.35}
def masks(n): k = int(S[n + "__n"]); return np.unpackbits(S[n])[:k * 16384].reshape(k, 128, 128).astype(bool) if k else np.zeros((0, 128, 128), bool)
sel = np.array([part[j] and names[j] not in bx and role[j] != 2 for j in range(len(names))]); J = np.where(sel)[0]; MK = {j: masks(names[j]) for j in J}
base = {j: miou_image(L[j] - T_PART > 0, G[j]) for j in J}
def eiou(m, p): i = (p * m).sum(); return i / (m.sum() + p.sum() - i + 1e-6)
def greedy(p, M, start=None, steps=6):
    cur = np.zeros(16384, bool) if start is None else start.ravel().copy(); A = M.reshape(len(M), -1); w = p.ravel(); sw = w.sum(); best = eiou(cur, w) if cur.any() else 0.0
    for _ in range(steps):
        cand = A | cur[None]; inter = cand @ w; sc = inter / (cand.sum(1) + sw - inter + 1e-6); k = int(sc.argmax())
        if sc[k] <= best + 1e-4: break
        best = float(sc[k]); cur = cand[k]
    return cur.reshape(128, 128), best
def vote(p, M, c):                                    # tous les masques dont la probabilité moyenne dépasse c
    A = M.reshape(len(M), -1); mp = (A @ p.ravel()) / A.sum(1); k = mp >= c
    u = A[k].any(0).reshape(128, 128) if k.any() else np.zeros((128, 128), bool); return u, (eiou(u.ravel(), p.ravel()) if u.any() else 0.0)
def refine(p, M, c):                                  # prédiction du réseau, corrigée masque par masque : un masque majoritairement prédit est pris en entier, sinon retiré
    b = p > 0.5; A = M.reshape(len(M), -1); order = np.argsort(-A.sum(1)); out = b.ravel().copy()
    for k in order:
        f = out[A[k]].mean()
        if f >= c: out[A[k]] = True
        elif f <= 1 - c and f > 0: out[A[k]] = False
    out = out.reshape(128, 128); return out, eiou(out.ravel(), p.ravel())
R = {}
for j in J:
    for k in (0.5, 1.0, 2.0):
        p = 1 / (1 + np.exp(-k * (L[j] - T_PART))); M = MK[j]
        if len(M) == 0: continue
        R[(j, "glouton", k)] = greedy(p, M); R[(j, "glouton+réseau", k)] = greedy(p, M, start=(p > 0.5))
        for c in (0.4, 0.5, 0.6, 0.7): R[(j, f"vote{c}", k)] = vote(p, M, c); R[(j, f"corrige{c}", k)] = refine(p, M, c)
fr = G.reshape(len(G), -1).mean(1); tune = np.array([j for j in J if not res[j]]); rs = np.array([j for j in J if res[j]])
print(f"partielles mesurées : {len(J)} (réglage {len(tune)}, réserve {len(rs)}) ; réseau seul : réglage {np.mean([base[j] for j in tune]):.4f} RÉSERVE {np.mean([base[j] for j in rs]):.4f}")
def run(method, k, tau, js): return np.array([(miou_image(R[(j, method, k)][0], G[j]) if (j, method, k) in R and R[(j, method, k)][1] >= tau else base[j]) for j in js])
methods = sorted({m for (_, m, _) in R}); TAUS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
for m in methods:
    v = max((run(m, k, t, tune).mean(), k, t) for k in (0.5, 1.0, 2.0) for t in TAUS); h = tune[::2], tune[1::2]; gains = []
    for A_, B_ in ((h[0], h[1]), (h[1], h[0])):
        b = max((run(m, k, t, A_).mean(), k, t) for k in (0.5, 1.0, 2.0) for t in TAUS); gains.append(run(m, b[1], b[2], B_).mean() - np.mean([base[j] for j in B_]))
    sc = run(m, v[1], v[2], rs); line = f"{m:16s} k={v[1]} tau={v[2]} : réglage {v[0]:.4f} | RÉSERVE {sc.mean():.4f} | moitiés croisées {gains[0]:+.3f} {gains[1]:+.3f} |"
    for lo, hi in ((0, .03), (.03, .06), (.06, 1)):
        s = np.array([j for j in J if lo <= fr[j] < hi]); line += f" zone {lo:.2f}-{hi:.2f} {run(m, v[1], v[2], s).mean():.3f} (réseau {np.mean([base[j] for j in s]):.3f})"
    print(line)
