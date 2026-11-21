"""Critique T1 tour 1 (agent de Titouan) : deux post-traitements sans entrainement, mesures en ecart apparie.
E1 = regle de decision a IoU esperee maximale (Nowozin 2014) sur probabilites calibrees, par image.
E2 = a priori de position (carte moyenne de la verite hors reserve, par filigrane x population) ajoute aux logits.
Reserve = indice global % 5 == 0 dans release/train/metadata.json. Reglages : hors reserve seulement.
Usage : python crit1.py <methode> <pred_train.npz> [cle_logits]"""
import json, sys
import numpy as np
from scipy.ndimage import gaussian_filter

KIT = "/home/hermes/eac/kit/release/train"
meth, path = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(0)

meta = json.load(open(f"{KIT}/metadata.json"))
z = np.load(f"{KIT}/masks_128.npz")
GT = dict(zip(z["names"].tolist(), z["masks"]))
gidx = {m["image_name"]: i for i, m in enumerate(meta)}
full = {m["image_name"]: bool(m["full_watermark"]) for m in meta}

d = np.load(path, allow_pickle=True)
names = [str(n) for n in d["names"]]
key = sys.argv[3] if len(sys.argv) > 3 else [k for k in d.files if k != "names"][0]
L = d[key].astype(np.float32)
assert L.shape[1:] == (128, 128), L.shape
sel = [i for i, n in enumerate(names) if n in GT]
names = [names[i] for i in sel]; L = L[sel]
G = np.stack([GT[n] for n in names]).astype(np.uint8)
F = np.array([full[n] for n in names])
fold = np.array([gidx[n] % 5 for n in names])
RES = fold == 0
print(f"{meth}: {len(names)} images, reserve {RES.sum()}, pleines {F.sum()}, cle {key}")


def miou(p, g):
    s = 0.0
    for c in (0, 1):
        u = np.logical_or(p == c, g == c).sum()
        s += 1.0 if u == 0 else np.logical_and(p == c, g == c).sum() / u
    return s / 2


def scores(P, idx):
    return np.array([miou(P[i], G[i]) for i in idx])


# ---------- briques ----------
def thr_masks(Lx, t):
    return (Lx > t).astype(np.uint8)


def tune_thr(Lx, idx):
    ts = np.quantile(Lx[idx], np.linspace(0.02, 0.98, 49))
    ts = np.unique(np.concatenate([ts, [0.0]]))
    best = max(ts, key=lambda t: scores(thr_masks(Lx[idx], t), range(len(idx))).mean() if False else
               np.mean([miou(Lx[i] > t, G[i]) for i in idx]))
    return best


def calib(Lx, idx):
    """regression logistique 1D (a, b) sur pixels hors reserve, par Newton."""
    x = Lx[idx].reshape(-1); y = G[idx].reshape(-1).astype(np.float64)
    s = rng.choice(len(x), min(len(x), 400000), replace=False); x = x[s].astype(np.float64); y = y[s]
    a, b = 0.5, 0.0
    for _ in range(30):
        p = 1 / (1 + np.exp(-(a * x + b))); w = p * (1 - p) + 1e-9
        g = np.array([((p - y) * x).sum(), (p - y).sum()])
        H = np.array([[(w * x * x).sum(), (w * x).sum()], [(w * x).sum(), w.sum()]])
        a, b = np.array([a, b]) - np.linalg.solve(H, g)
    return a, b


def nowozin(p):
    """p : probas 128x128 -> masque qui maximise 0,5*(E-IoU1 + E-IoU0) (approximation ratio des esperances)."""
    f = p.reshape(-1); N = f.size; o = np.argsort(-f); fs = f[o]
    P = fs.sum(); k = np.arange(1, N)
    S = np.cumsum(fs)[:-1]
    iou1 = S / (k + P - S)
    C = (N - k) - (P - S)
    iou0 = C / ((N - k) + (N - P) - C)
    kb = k[np.argmax(iou1 + iou0)]
    m = np.zeros(N, np.uint8); m[o[:kb]] = 1
    return m.reshape(128, 128)


def smooth(Lx, s):
    return Lx if s == 0 else np.stack([gaussian_filter(l, s) for l in Lx])


# ---------- evaluation : CV 4 plis hors reserve + reserve ----------
def run(method_fn, label):
    """method_fn(train_idx, test_idx) -> masques pour test_idx. Renvoie scores par image (CV hors reserve, reserve)."""
    out = np.zeros(len(names))
    for f in range(1, 5):
        tr = np.where((fold != 0) & (fold != f))[0]; te = np.where(fold == f)[0]
        out[te] = scores(method_fn(tr, te), range(len(te))) if False else [miou(m, G[i]) for m, i in zip(method_fn(tr, te), te)]
    tr = np.where(fold != 0)[0]; te = np.where(RES)[0]
    out[te] = [miou(m, G[i]) for m, i in zip(method_fn(tr, te), te)]
    return out


def per_pop(fn):
    """applique fn separement par population (pleine / partielle), reglages sur la meme population."""
    def g(tr, te):
        res = [None] * len(te)
        for pop in (True, False):
            trp = tr[F[tr] == pop]; tep_pos = [j for j, i in enumerate(te) if F[i] == pop]
            tep = te[tep_pos]
            if len(tep) == 0: continue
            ms = fn(trp, tep)
            for j, m in zip(tep_pos, ms): res[j] = m
        return res
    return g


def base_thr(tr, te):
    t = tune_thr(L, tr); return [L[i] > t for i in te]


def make_sm_thr(sig):
    def f(tr, te):
        Ls = smooth(L[np.concatenate([tr, te])], sig)
        Ltr, Lte = Ls[:len(tr)], Ls[len(tr):]
        ts = np.unique(np.concatenate([np.quantile(Ltr, np.linspace(0.02, 0.98, 49)), [0.0]]))
        t = max(ts, key=lambda t: np.mean([miou(Ltr[j] > t, G[i]) for j, i in enumerate(tr)]))
        return [Lte[j] > t for j in range(len(te))]
    return f


def e1_now(tr, te):
    a, b = calib(L, tr)
    return [nowozin(1 / (1 + np.exp(-(a * L[i] + b)))) for i in te]


def prior_logit(tr):
    pi = gaussian_filter(G[tr].mean(0).astype(np.float64), 3)
    pi = np.clip(pi, 1e-3, 1 - 1e-3)
    return np.log(pi / (1 - pi)).astype(np.float32)


def e2_prior_thr(tr, te):
    lp = prior_logit(tr); lp = lp - lp.mean()
    best = None
    for w in (0.0, 0.25, 0.5, 1.0, 2.0):
        t = tune_thr(L + w * lp, tr)
        s = np.mean([miou((L[i] + w * lp) > t, G[i]) for i in tr])
        if best is None or s > best[0]: best = (s, w, t)
    _, w, t = best
    print(f"   prior w={w}", end="")
    return [(L[i] + w * lp) > t for i in te]


def e12(tr, te):
    """a priori + calibration + Nowozin"""
    lp = prior_logit(tr); lp = lp - lp.mean()
    best = None
    for w in (0.0, 0.25, 0.5, 1.0, 2.0):
        a, b = calib(L + w * lp, tr)
        s = np.mean([miou(nowozin(1 / (1 + np.exp(-(a * (L[i] + w * lp) + b)))), G[i]) for i in tr[::2]])
        if best is None or s > best[0]: best = (s, w, a, b)
    _, w, a, b = best
    print(f"   e12 w={w}", end="")
    return [nowozin(1 / (1 + np.exp(-(a * (L[i] + w * lp) + b)))) for i in te]


def report(base, new, label):
    def ci(dv):
        bs = [rng.choice(dv, len(dv)).mean() for _ in range(2000)]
        return np.percentile(bs, [2.5, 97.5])
    for nm, m in (("RESERVE", RES), ("CV hors reserve", ~RES)):
        for pop, pm in (("toutes", np.ones_like(F)), ("pleines", F), ("partielles", ~F)):
            mm = m & pm; dv = new[mm] - base[mm]; lo, hi = ci(dv)
            print(f"  {label:<22} {nm:<16} {pop:<10} n={mm.sum():3d} base {base[mm].mean():.4f} -> {new[mm].mean():.4f}"
                  f"  ecart {dv.mean():+.4f} [{lo:+.4f} ; {hi:+.4f}]  mieux {int((dv > 1e-9).sum())} pires {int((dv < -1e-9).sum())}")


base = run(per_pop(base_thr), "base")
print()
res = {"base": base}
for label, fn in (("E1 Nowozin", per_pop(e1_now)), ("lissage s=1", per_pop(make_sm_thr(1.0))),
                  ("lissage s=2", per_pop(make_sm_thr(2.0))), ("E2 a priori position", per_pop(e2_prior_thr)),
                  ("E1+E2", per_pop(e12))):
    new = run(fn, label); print(); report(base, new, label); res[label] = new
np.savez(f"crit1_{meth}.npz", names=np.array(names), **{k.replace(" ", "_"): v for k, v in res.items()})
