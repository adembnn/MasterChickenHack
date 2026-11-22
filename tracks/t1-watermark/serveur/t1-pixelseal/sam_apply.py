"""Calage des images PARTIELLES sur les contours SAM, pour n'importe quel filigrane.
Pour chaque image classée partielle (bits justes de l'image entière <= 0,75) : p = sigmoïde(k x (logit - t)) ; on prend l'union gloutonne de
masques SAM qui maximise l'IoU attendue sous p ; si cette IoU attendue >= tau on rend l'union, sinon on garde la carte.
Réglage de (t, k, tau) hors réserve ; mesure sur la réserve (index % 5 == 0) et en moitiés croisées.
Usage : python sam_apply.py <methode> <pred_train.npz> <pred_test.npz> <sortie_train.npz> <sortie_test.npz>"""
import os, sys, json, numpy as np
from pathlib import Path
HERE = Path(os.environ["WMLOC_ROOT"]); D = Path(os.environ.get("SAM_DIR", HERE / "adem" / "t1-pixelseal")); sys.path.insert(0, str(HERE / "kit"))
from wmloc.utils import miou_image
method, ptr, pte, otr, ote = sys.argv[1:6]
def load(p): z = np.load(p); return z["names"].tolist(), z["logits"].astype(np.float32)
def acc(split): f = np.load(D / f"feat_{split}.npz"); return dict(zip(f["names"].tolist(), f["acc"].tolist()))
def sam(split):
    S = np.load(D / f"sam_{split}_{method}.npz"); files = set(S.files)
    def get(n):
        if n + "__n" not in files: return np.zeros((0, 128, 128), bool)
        k = int(S[n + "__n"]); return np.unpackbits(S[n])[:k * 16384].reshape(k, 128, 128).astype(bool) if k else np.zeros((0, 128, 128), bool)
    return get
def snap(l, M):
    p = (1 / (1 + np.exp(-l))).ravel(); sp = p.sum(); A = M.reshape(len(M), -1); cur = np.zeros(16384, bool); best = 0.0
    for _ in range(6):
        cand = A | cur[None]; inter = cand @ p; sc = inter / (cand.sum(1) + sp - inter + 1e-6); k = int(sc.argmax())
        if sc[k] <= best + 1e-4: break
        best = float(sc[k]); cur = cand[k]
    return cur.reshape(128, 128), best
names, L = load(ptr); A = acc("train"); part = np.array([A[n] <= 0.75 for n in names]); get = sam("train")
meta = json.loads((HERE / "release/train/metadata.json").read_text()); idx = {m["image_name"]: i for i, m in enumerate(meta)}; tf = {m["image_name"]: bool(m.get("full_watermark")) for m in meta}
g = np.load(HERE / "release/train/masks_128.npz"); G = dict(zip(g["names"].tolist(), g["masks"])); G = np.stack([G[n] for n in names]).astype(bool)
res = np.array([idx[n] % 5 == 0 for n in names]); truefull = np.array([tf[n] for n in names]); J = np.where(part)[0]; MK = {j: get(names[j]) for j in J}
TS = (-3.0, -2.0, -1.0, 0.0, 1.0); KS = (0.5, 1.0, 2.0); TAUS = (0.2, 0.3, 0.4, 0.5, 0.6)
base = {(j, t): miou_image(L[j] > t, G[j]) for j in J for t in np.linspace(-4, 3, 15)}
tb = max((np.mean([base[(j, t)] for j in J if not res[j]]), t) for t in np.linspace(-4, 3, 15))[1]       # meilleur seuil seul, hors réserve
R = {}
for j in J:
    if len(MK[j]) == 0: continue
    for t in TS:
        for k in KS: m, e = snap(k * (L[j] - t), MK[j]); R[(j, t, k)] = (miou_image(m, G[j]), e, m)
def run(js, t, k, tau): return np.array([R[(j, t, k)][0] if (j, t, k) in R and R[(j, t, k)][1] >= tau else base[(j, tb)] for j in js])
tune = np.array([j for j in J if not res[j]]); rs = np.array([j for j in J if res[j]])
cfg = max((run(tune, t, k, tau).mean(), t, k, tau) for t in TS for k in KS for tau in TAUS)
h = tune[::2], tune[1::2]; gains = []
for X, Y in ((h[0], h[1]), (h[1], h[0])):
    c = max((run(X, t, k, tau).mean(), t, k, tau) for t in TS for k in KS for tau in TAUS); gains.append(run(Y, *c[1:]).mean() - np.mean([base[(j, tb)] for j in Y]))
b_t = np.mean([base[(j, tb)] for j in tune]); b_r = np.mean([base[(j, tb)] for j in rs]); s_r = run(rs, *cfg[1:])
allb = np.array([miou_image(L[j] > (tb if part[j] else 0), G[j]) for j in range(len(names))]); alls = allb.copy(); alls[J] = run(J, *cfg[1:])
fr = G.reshape(len(G), -1).mean(1)
print(f"== {method} : {len(J)} images classées partielles (hors réserve {len(tune)}, réserve {len(rs)}), {sum(len(MK[j]) > 0 for j in J)} avec masques SAM")
print(f"partielles, carte seule (seuil {tb:+.1f} réglé hors réserve) : hors réserve {b_t:.4f} | RÉSERVE {b_r:.4f}")
print(f"partielles, + calage SAM (t={cfg[1]}, k={cfg[2]}, tau={cfg[3]}) : hors réserve {cfg[0]:.4f} | RÉSERVE {s_r.mean():.4f} | moitiés croisées {gains[0]:+.4f} {gains[1]:+.4f}")
for lo, hi in ((0, .03), (.03, .06), (.06, .15), (.15, 1)):
    s = np.array([j for j in J if lo <= fr[j] < hi and not truefull[j]])
    if len(s): print(f"   zone filigranée {lo:.2f}-{hi:.2f} n={len(s):2d} : carte {np.mean([base[(j, tb)] for j in s]):.3f} -> SAM {run(s, *cfg[1:]).mean():.3f}")
print(f"toutes les images du filigrane ({len(names)}) : hors réserve {allb[~res].mean():.4f} -> {alls[~res].mean():.4f} | RÉSERVE {allb[res].mean():.4f} -> {alls[res].mean():.4f}")
def write(names, L, part, get, out):
    O = L.copy(); n = 0
    for j in np.where(part)[0]:
        M = get(names[j]); O[j] = L[j] - tb
        if len(M):
            m, e = snap(cfg[2] * (L[j] - cfg[1]), M)
            if e >= cfg[3]: O[j] = np.where(m, 6.0, -6.0); n += 1
    np.savez_compressed(out, names=np.array(names), logits=O.astype(np.float16)); print("écrit", out, "| images calées sur SAM :", n, "sur", int(part.sum()), "partielles")
write(names, L, part, get, otr)
tn, TL = load(pte); At = acc("test"); write(tn, TL, np.array([At[n] <= 0.75 for n in tn]), sam("test"), ote)
