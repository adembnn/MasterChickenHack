"""Calage sur contours d'objets : la zone rare prédite (floue) est remplacée par l'union de masques SAM qui maximise le recouvrement
ATTENDU d'après la carte de probabilité du réseau. Mesuré sur les prédictions hors-pli ; réglage hors réserve."""
import os, sys, json, numpy as np
from pathlib import Path
HERE = Path(os.environ["WMLOC_ROOT"]); D = HERE / "adem" / "t1-pixelseal"; sys.path.insert(0, str(HERE / "kit"))
from wmloc.utils import miou_image
pred = sys.argv[1] if len(sys.argv) > 1 else "pred_train_G.npz"
z = np.load(D / pred); names = z["names"].tolist(); L = z["logits"].astype(np.float32); S = np.load(D / "sam_train.npz")
meta = json.loads((HERE / "release/train/metadata.json").read_text()); idx = {m["image_name"]: i for i, m in enumerate(meta)}
g = np.load(HERE / "release/train/masks_128.npz"); G = dict(zip(g["names"].tolist(), g["masks"])); G = np.stack([G[n] for n in names]).astype(bool)
pop = json.loads((D / "pop_train.json").read_text()); full = np.array([pop[n] > 0.75 for n in names]); res = np.array([idx[n] % 5 == 0 for n in names])
bx = {r["name"]: r for r in json.loads((D / "box_train.json").read_text()) if "box" in r and r["bits"] >= 0.85 and r["bits"] - r["glob"] >= 0.35}
def masks(n): k = int(S[n + "__n"]); return np.unpackbits(S[n])[:k * 16384].reshape(k, 128, 128).astype(bool) if k else np.zeros((0, 128, 128), bool)
MK = [masks(n) for n in names]; T = {True: 0.0, False: -2.75}                # seuils par population déjà réglés (post.py)
def rare_score(j): return -(L[j] - T[True]) if full[j] else (L[j] - T[False])     # > 0 = classe rare
def greedy(p, M, tgt=None):                           # union gloutonne de masques qui maximise l'IoU (attendue si tgt None, vraie sinon)
    cur = np.zeros((128, 128), bool); best = 0.0; A = M.reshape(len(M), -1); w = (p if tgt is None else tgt.astype(np.float32)).ravel(); sw = w.sum()
    for _ in range(6):
        cand = A | cur.ravel()[None]; inter = cand @ w; sc = inter / (cand.sum(1) + sw - inter + 1e-6); k = int(sc.argmax())
        if sc[k] <= best + 1e-4: break
        best = float(sc[k]); cur = cand[k].reshape(128, 128)
    return cur, best
def final(j, rare): return ~rare if full[j] else rare  # masque filigrané
base = np.array([miou_image(final(j, rare_score(j) > 0), G[j]) for j in range(len(names))])
rows = []
for j in range(len(names)):
    if names[j] in bx or len(MK[j]) == 0: rows.append(None); continue
    s = rare_score(j); tg = (~G[j]) if full[j] else G[j]; o, oi = greedy(None, MK[j], tg); r = {"oracle": miou_image(final(j, o), G[j]), "oi": oi}
    for k in (0.5, 1.0, 2.0):
        m, e = greedy(1 / (1 + np.exp(-k * s)), MK[j]); r[k] = (miou_image(final(j, m), G[j]), e, m.mean())
    rows.append(r)
ok = np.array([r is not None for r in rows]); fr = np.where(full, 1 - G.reshape(len(G), -1).mean(1), G.reshape(len(G), -1).mean(1))
print(f"{len(names)} images, {ok.sum()} avec masques SAM (hors rectangles déjà rendus) ; médiane {int(np.median([len(m) for m in MK]))} masques par image")
for nm, sel in (("PLEINES", full & ok), ("PARTIELLES", ~full & ok)):
    print(f"\n{nm} n={sel.sum()} | réseau seul : hors réserve {base[sel & ~res].mean():.4f} RÉSERVE {base[sel & res].mean():.4f} | plafond SAM (meilleure union connaissant la vérité) : {np.mean([rows[j]['oracle'] for j in np.where(sel)[0]]):.4f}")
    for lo, hi in ((0, .03), (.03, .06), (.06, .15), (.15, 1)):
        s = np.where(sel & (fr >= lo) & (fr < hi))[0]
        if len(s): print(f"   zone rare {lo:.2f}-{hi:.2f} n={len(s):2d} : réseau {base[s].mean():.3f} | plafond SAM {np.mean([rows[j]['oracle'] for j in s]):.3f} | SAM choisi par la carte (k=1) {np.mean([rows[j][1.0][0] for j in s]):.3f}")
    bestcfg = (-1, None)
    for k in (0.5, 1.0, 2.0):
        for tau in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 2.0):
            sc = np.array([(rows[j][k][0] if rows[j][k][1] >= tau else base[j]) if rows[j] is not None else base[j] for j in range(len(names))])
            v = sc[sel & ~res].mean()
            if v > bestcfg[0]: bestcfg = (v, (k, tau), sc)
    k, tau = bestcfg[1]; sc = bestcfg[2]; used = np.mean([rows[j][k][1] >= tau for j in np.where(sel)[0]])
    print(f"   règle : prendre l'union SAM si IoU attendue >= {tau} (k={k}), sinon le réseau ; appliquée à {used:.0%} des images")
    print(f"   -> hors réserve {sc[sel & ~res].mean():.4f} (réseau {base[sel & ~res].mean():.4f}) | RÉSERVE {sc[sel & res].mean():.4f} (réseau {base[sel & res].mean():.4f})")
    h = np.where(sel & ~res)[0]; A_, B_ = h[::2], h[1::2]; gains = []
    for X_, Y_ in ((A_, B_), (B_, A_)):
        bc = max(((np.mean([(rows[j][kk][0] if rows[j][kk][1] >= tt else base[j]) for j in X_]), kk, tt) for kk in (0.5, 1.0, 2.0) for tt in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 2.0)))
        gains.append(np.mean([(rows[j][bc[1]][0] if rows[j][bc[1]][1] >= bc[2] else base[j]) for j in Y_]) - base[Y_].mean())
    print(f"   gain réglé sur une moitié, mesuré sur l'autre : {gains[0]:+.4f}, {gains[1]:+.4f}")
