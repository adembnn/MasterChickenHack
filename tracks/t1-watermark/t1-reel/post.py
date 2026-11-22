"""T1 · post-traitement réglé hors réserve : seuil par population (pleine / partielle, donnée par les bits justes de l'image entière),
lissage, petites composantes, remplissage, rectangle. Réglage glouton sur les prédictions hors-pli (136 images), chiffre honnête sur la réserve (34).
Usage : python post.py <tag> <methode> [--write]"""
import sys, json, itertools, numpy as np
from pathlib import Path
from scipy import ndimage as ndi
ROOT = Path(__file__).resolve().parent; HERE = ROOT.parent if (ROOT.parent / "kit").exists() else ROOT.parent.parent; CACHE = ROOT / "cache"
sys.path.insert(0, str(HERE / "kit"))
from wmloc.utils import miou_image

DEF = dict(sig=0.0, thf=0.0, thp=0.0, minc_f=0, minw_f=0, minw_p=0, keep1=0, fill_p=0, rect=9.0, close_f=0, ne_p=0.0, ne_f=0.0)
GRID = dict(sig=[0, 0.7, 1.2, 2.0], thf=list(np.linspace(-3, 3, 13)), thp=list(np.linspace(-3, 3, 13)), minc_f=[0, 4, 16, 48, 128], minw_f=[0, 4, 16, 48],
            minw_p=[0, 2, 4, 8, 16, 32], keep1=[0, 1], fill_p=[0, 1], rect=[9.0, 0.9, 0.8, 0.7, 0.6], close_f=[0, 1, 2], ne_p=[0, 0.5, 1, 2, 3, 5], ne_f=[0, 0.5, 1, 2, 3, 5])

def drop_small(m, n):                                    # enlève les composantes vraies de moins de n cases
    if n <= 0 or not m.any(): return m
    lab, k = ndi.label(m)
    if k == 0: return m
    sz = ndi.sum(m, lab, np.arange(1, k + 1)); return np.isin(lab, 1 + np.where(sz >= n)[0])

def apply(lg, full, P):
    """lg : logits 128x128 (>0 = filigrané) ; full : drapeau pleine ; renvoie le masque."""
    if P["sig"] > 0: lg = ndi.gaussian_filter(lg.astype(np.float32), P["sig"])
    if full:
        m = lg > P["thf"]
        if P["close_f"]: m = ~ndi.binary_opening(~m, iterations=int(P["close_f"]))        # efface les trous filiformes
        m = ~drop_small(~m, P["minc_f"])                 # un trou propre trop petit est rebouché
        m = drop_small(m, P["minw_f"])                   # un îlot filigrané isolé dans un trou est retiré
        if P["ne_f"] > 0 and (~m).sum() < 2:             # une pleine a toujours un trou : on prend la zone la moins filigranée autour du minimum
            lab, k = ndi.label(lg <= lg.min() + P["ne_f"]); m = lab != lab[np.unravel_index(np.argmin(lg), lg.shape)]
        return m
    m = lg > P["thp"]
    m = drop_small(m, P["minw_p"])
    if P["keep1"] and m.any():
        lab, k = ndi.label(m)
        if k > 1: sz = ndi.sum(m, lab, np.arange(1, k + 1)); m = lab == 1 + int(np.argmax(sz))
    if P["ne_p"] > 0 and m.sum() < 2:                    # une partielle a toujours une zone filigranée : on prend la tache autour du maximum
        lab, k = ndi.label(lg >= lg.max() - P["ne_p"]); m = lab == lab[np.unravel_index(np.argmax(lg), lg.shape)]
    if P["fill_p"]: m = ndi.binary_fill_holes(m)
    if P["rect"] < 1 and m.any():                        # si la forme remplit presque sa boîte, c'est un rectangle : on rend la boîte
        ys, xs = np.where(m); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        if m.sum() >= P["rect"] * (y1 - y0) * (x1 - x0): m = np.zeros_like(m); m[y0:y1, x0:x1] = True
    return m

def score(L, FL, Y, P): return np.array([miou_image(apply(l, f, P), y) for l, f, y in zip(L, FL, Y)])

def tune(L, FL, Y, rounds=2):
    P = dict(DEF); best = score(L, FL, Y, P).mean()
    for _ in range(rounds):
        for k, vals in GRID.items():
            for v in vals:
                Q = dict(P); Q[k] = float(v) if isinstance(v, (float, np.floating)) else v; s = score(L, FL, Y, Q).mean()
                if s > best + 1e-5: best, P = s, Q
    return P, best

def load(tag, method):
    trm = json.loads((HERE / "release/train/metadata.json").read_text()); Y128 = np.load(CACHE / "Y128_train.npy")
    idx = [i for i, m in enumerate(trm) if m["method"] == method]; pos = {i: p for p, i in enumerate(idx)}; AALL = np.load(sorted(ROOT.glob(f"data_{method}*/AALL_train.npy"))[0])
    Ps = [ROOT / "runs" / t / method for t in tag.split("+")]; oi, ol, of, rl = [], [], [], []; ld = lambda f, n: np.mean([np.load(q / f"fold{f}" / n)["logits"].astype(np.float32) for q in Ps], 0)
    for f in range(5):
        z = np.load(Ps[0] / f"fold{f}" / "oof.npz"); oi += list(z["idx"]); ol.append(ld(f, "oof.npz")); of += [f] * len(z["idx"])
        ri = np.load(Ps[0] / f"fold{f}" / "res.npz")["idx"]; rl.append(ld(f, "res.npz"))
    oi = np.array(oi); fl = lambda L: np.array([AALL[pos[i]] > 0.75 for i in L]); tf = lambda L: np.array([bool(trm[i]["full_watermark"]) for i in L])
    return dict(oi=oi, ol=np.concatenate(ol), of=np.array(of), ri=ri, rl=np.mean(rl, 0), oy=Y128[oi], ry=Y128[ri], ofl=fl(oi), rfl=fl(ri), ofull=tf(oi), rfull=tf(ri))

if __name__ == "__main__":
    tag, method = sys.argv[1], sys.argv[2]; d = load(tag, method)
    print(f"## {tag} {method} : drapeau = vraie population dans {np.mean(d['ofl'] == d['ofull']):.3f} (hors-pli) / {np.mean(d['rfl'] == d['rfull']):.3f} (réserve)")
    P, s = tune(d["ol"], d["ofl"], d["oy"])
    # estimation honnête sur les 136 : réglage sur 4 plis, application au 5e
    nested = np.zeros(len(d["oi"]))
    for f in range(5):
        tr = d["of"] != f; Pf, _ = tune(d["ol"][tr], d["ofl"][tr], d["oy"][tr], rounds=1); nested[~tr] = score(d["ol"][~tr], d["ofl"][~tr], d["oy"][~tr], Pf)
    b0 = score(d["ol"], d["ofl"], d["oy"], DEF); r0 = score(d["rl"], d["rfl"], d["ry"], DEF); r1 = score(d["rl"], d["rfl"], d["ry"], P)
    f = lambda v, m: f"{v.mean():.3f} (pleines {v[m].mean():.3f}, partielles {v[~m].mean():.3f})"
    print("hors-pli brut (seuil 0)      :", f(b0, d["ofull"])); print("hors-pli réglé (optimiste)   :", f"{s:.3f}"); print("hors-pli réglé emboîté       :", f(nested, d["ofull"]))
    print("RÉSERVE brute (seuil 0)      :", f(r0, d["rfull"])); print("RÉSERVE post-traitée         :", f(r1, d["rfull"])); print("réglages :", json.dumps(P))
    if "--write" in sys.argv: (ROOT / "runs" / tag / method / "post.json").write_text(json.dumps(P))
