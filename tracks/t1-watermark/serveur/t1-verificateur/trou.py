"""Forme du trou propre des images pleines (vérités du train), plafonds avec vérité (f) meilleur rectangle, (g) meilleur masque SAM existant,
et comparaison avec la zone filigranée des partielles. Base : masques honnêtes du candidat N."""
import sys, glob
import numpy as np
from scipy.ndimage import label
from scipy.stats import ks_2samp
from verif import HERE, METHODS, miou_vec, load_truth
N = sys.argv[1] if len(sys.argv) > 1 else "6"
A = HERE / "t1-assemblage"; meta, g = load_truth(check_png=False); idx = {m["image_name"]: i for i, m in enumerate(meta)}; bym = {m["image_name"]: m for m in meta}
SAM = {}
for f in glob.glob(str(A / "samcand_train_*.npz")):
    z = np.load(f)
    for i, n in enumerate(z["names"].tolist()):
        nv = int(z["nvalid"][i])
        if nv: c = np.unpackbits(z["masks"][i, :nv], axis=1).astype(bool); c = c[c.any(1)]; SAM[n] = c.reshape(len(c), 128, 128)
iou = lambda a, b: (a & b).sum() / max((a | b).sum(), 1)
def geo(Z):
    """Z = petite zone (trou des pleines, ou zone filigranée des partielles)."""
    y, x = np.where(Z); y0, y1, x0, x1 = y.min(), y.max() + 1, x.min(), x.max() + 1; fill = Z.sum() / ((y1 - y0) * (x1 - x0))
    lab, n = label(Z, np.ones((3, 3))); sz = np.bincount(lab.ravel())[1:]; big = int((sz >= max(4, 0.05 * Z.sum())).sum())
    t = [y0 == 0, x0 == 0, y1 == 128, x1 == 128]; nt = sum(t)
    pos = "centre" if nt == 0 else ("bord" if nt == 1 else ("coin" if (nt == 2 and not (t[0] and t[2]) and not (t[1] and t[3])) else "bande ou 3 côtés"))
    return dict(frac=Z.mean(), fill=fill, ncomp=big, pos=pos, box=(y0, x0, y1, x1), asp=(y1 - y0) / (x1 - x0), cy=(y0 + y1) / 256, cx=(x0 + x1) / 256)
def best_sam(n, Z):
    c = SAM.get(n)
    if c is None: return None, None, None
    v = np.array([iou(k, Z) for k in c]); b = int(v.argmax()); u = c[b]; best2 = v[b]; m2 = u
    for k in c:
        w = iou(u | k, Z)
        if w > best2: best2, m2 = w, u | k
    return v[b], u, (best2, m2)
def best_rect(Z):
    y, x = np.where(Z); best = None
    for q in (0, 0.01, 0.02, 0.05, 0.1):
        y0, y1 = np.quantile(y, [q, 1 - q]); x0, x1 = np.quantile(x, [q, 1 - q]); R = np.zeros_like(Z); R[int(round(y0)):int(round(y1)) + 1, int(round(x0)):int(round(x1)) + 1] = True
        v = iou(R, Z)
        if best is None or v > best[0]: best = (v, R)
    return best
def shape(G, s):
    if G["fill"] >= 0.95: return "rectangle"
    if s is not None and s >= 0.8: return "objet (SAM >= 0,8)"
    if G["ncomp"] >= 2: return "plusieurs morceaux"
    return "autre"
SH = ["rectangle", "objet (SAM >= 0,8)", "plusieurs morceaux", "autre"]; POS = ["centre", "bord", "coin", "bande ou 3 côtés"]
def desc(lab, R):
    fr = np.array([r["frac"] for r in R]); q = np.quantile(fr, [0.1, 0.25, 0.5, 0.75, 0.9]); sh = [r["shape"] for r in R]; n = len(R)
    print(f"   {lab} n={n} | forme : " + " ; ".join(f"{k} {sh.count(k) / n:.2f}" for k in SH) + f" | SAM disponible {np.mean([r['sam'] is not None for r in R]):.2f}")
    print(f"      taille (part de l'image) déciles 10/25/50/75/90 : {' / '.join(f'{v:.3f}' for v in q)} ; < 1 % : {np.mean(fr < 0.01):.2f}, 1-3 % : {np.mean((fr >= 0.01) & (fr < 0.03)):.2f}, 3-10 % : {np.mean((fr >= 0.03) & (fr < 0.1)):.2f}, > 10 % : {np.mean(fr >= 0.1):.2f}")
    print(f"      position : " + " ; ".join(f"{k} {[r['pos'] for r in R].count(k) / n:.2f}" for k in POS) + f" | morceaux >= 2 : {np.mean([r['ncomp'] >= 2 for r in R]):.2f} | remplissage de la boîte médian {np.median([r['fill'] for r in R]):.2f} | rapport hauteur/largeur médian {np.median([r['asp'] for r in R]):.2f} | centre (y, x) moyen {np.mean([r['cy'] for r in R]):.2f}, {np.mean([r['cx'] for r in R]):.2f}")
ALL = {}; TOT = {}
for me in METHODS:
    z = np.load(A / f"candidat_{N}_honnete_{me}.npz"); names = [str(a) for a in z["names"]]; P = z["masks"].astype(bool); G = np.stack([g[a] for a in names])
    res = np.array([idx[a] % 5 == 0 for a in names]); full = np.array([bym[a]["full_watermark"] for a in names]); s = miou_vec(P, G)
    R = []
    for i, a in enumerate(names):
        Z = ~G[i] if full[i] else G[i]; r = geo(Z); sv, sm, s2 = best_sam(a, Z); r.update(sam=sv, shape=shape(r, sv), name=a, full=bool(full[i]), res=bool(res[i]), i=i, base=s[i])
        v, Rm = best_rect(Z); r["rect_iou"] = v; r["f"] = max(s[i], miou_vec((~Rm if full[i] else Rm)[None], G[i][None])[0])
        if sv is not None:
            r["g1"] = max(s[i], miou_vec((~sm if full[i] else sm)[None], G[i][None])[0]); r["g2"] = max(r["g1"], miou_vec((~s2[1] if full[i] else s2[1])[None], G[i][None])[0])
        else: r["g1"] = r["g2"] = s[i]
        R.append(r)
    ALL[me] = R; Fp = [r for r in R if r["full"]]; Pp = [r for r in R if not r["full"]]
    print(f"\n===== {me}"); desc("TROU des pleines     ", Fp); desc("ZONE des partielles  ", Pp)
    print(f"      même loi de taille ? test de Kolmogorov-Smirnov trou des pleines contre zone des partielles : D = {ks_2samp([r['frac'] for r in Fp], [r['frac'] for r in Pp]).statistic:.3f}, p = {ks_2samp([r['frac'] for r in Fp], [r['frac'] for r in Pp]).pvalue:.3f}")
    print("   pleines, score du candidat par forme du trou : " + " ; ".join(f"{k} n={sum(r['shape'] == k for r in Fp)} score {np.mean([r['base'] for r in Fp if r['shape'] == k] or [np.nan]):.3f} -> (f) {np.mean([r['f'] for r in Fp if r['shape'] == k] or [np.nan]):.3f} (g) {np.mean([r['g2'] for r in Fp if r['shape'] == k] or [np.nan]):.3f}" for k in SH))
    for lab, key in (("candidat", "base"), ("(f) trou des pleines = meilleur rectangle", "f"), ("(g) trou des pleines = meilleur masque SAM", "g1"), ("(g2) = meilleure union de deux masques SAM", "g2"), ("(h) le meilleur de (f) et (g2)", None)):
        v = np.array([(max(r["f"], r["g2"]) if key is None else r[key]) if r["full"] else r["base"] for r in R]); b = np.array([r["base"] for r in R])
        print(f"   plafond {lab:44s} : réserve {v[res].mean():.4f} ({v[res].mean() - b[res].mean():+.4f}) | hors réserve {v[~res].mean():.4f} ({v[~res].mean() - b[~res].mean():+.4f}) | pleines seules {v[full].mean():.4f} (base {b[full].mean():.4f})")
        TOT.setdefault(lab, []).append((v[res].mean(), v[~res].mean(), b[res].mean(), b[~res].mean()))
    cov = [r for r in Fp if r["sam"] is not None]
    if cov: print(f"   SAM sur les pleines couvertes (n={len(cov)}) : IoU du meilleur masque avec le trou, médiane {np.median([r['sam'] for r in cov]):.2f}, part >= 0,8 : {np.mean([r['sam'] >= 0.8 for r in cov]):.2f}, >= 0,5 : {np.mean([r['sam'] >= 0.5 for r in cov]):.2f}")
    print(f"   meilleur rectangle : IoU avec le trou des pleines médiane {np.median([r['rect_iou'] for r in Fp]):.2f}, part >= 0,8 : {np.mean([r['rect_iou'] >= 0.8 for r in Fp]):.2f} | avec la zone des partielles médiane {np.median([r['rect_iou'] for r in Pp]):.2f}, part >= 0,8 : {np.mean([r['rect_iou'] >= 0.8 for r in Pp]):.2f}")
print("\n===== LES TROIS FILIGRANES ENSEMBLE"); desc("TROU des 255 pleines ", [r for me in METHODS for r in ALL[me] if r["full"]]); desc("ZONE des 255 partielles", [r for me in METHODS for r in ALL[me] if not r["full"]])
for lab, sub in (("réserve", True), ("hors réserve", False)): desc(f"TROU des pleines, {lab}", [r for me in METHODS for r in ALL[me] if r["full"] and r["res"] == sub])
for lab, v in TOT.items(): m = np.mean(v, 0); print(f"   plafond {lab:44s} : réserve {m[0]:.4f} ({m[0] - m[2]:+.4f}) | hors réserve {m[1]:.4f} ({m[1] - m[3]:+.4f})")
