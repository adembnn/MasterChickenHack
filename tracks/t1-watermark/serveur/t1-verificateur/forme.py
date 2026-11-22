"""La zone filigranée des pleines est-elle une forme simple (convexe, rectangle tourné, quadrilatère, ellipse) dont le trou serait le complément ?
Étape a : convexité + plafond enveloppe convexe + planches de contact. Étape b : plafonds des formes paramétriques ajustées avec la vérité.
Base : masques honnêtes du candidat N. Usage : forme.py a|b [N]"""
import sys, glob, os
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from verif import HERE, METHODS, miou_vec, load_truth
ST = sys.argv[1]; N = sys.argv[2] if len(sys.argv) > 2 else "6"; ME = os.path.dirname(os.path.abspath(__file__))
A = HERE / "t1-assemblage"; meta, g = load_truth(check_png=False); idx = {m["image_name"]: i for i, m in enumerate(meta)}; bym = {m["image_name"]: m for m in meta}
def sc(P, G):
    i1 = (P & G).sum(); u1 = (P | G).sum(); n = G.size; i0 = n - u1; u0 = n - i1; return 0.5 * ((i1 / u1 if u1 else 1.0) + (i0 / u0 if u0 else 1.0))
def pmask(pts):
    im = Image.new("1", (128, 128)); ImageDraw.Draw(im).polygon([(float(x), float(y)) for x, y in pts], fill=1); return np.array(im, bool)
def hull_pts(W):
    y, x = np.where(W); P = np.c_[x, y].astype(float)
    try: return P[ConvexHull(P).vertices]
    except Exception: return None
def reduce(h, n):
    """Enveloppe réduite à n sommets : on retire à chaque fois le sommet qui enlève le moins d'aire."""
    h = list(map(tuple, h))
    while len(h) > n:
        a = [abs((h[i - 1][0] - h[i][0]) * (h[(i + 1) % len(h)][1] - h[i][1]) - (h[(i + 1) % len(h)][0] - h[i][0]) * (h[i - 1][1] - h[i][1])) for i in range(len(h))]; h.pop(int(np.argmin(a)))
    return np.array(h)
def rect_pts(p):
    cx, cy, w, h, t = p; c, s = np.cos(t), np.sin(t); d = np.array([[-w, -h], [w, -h], [w, h], [-w, h]]) / 2; return d @ np.array([[c, s], [-s, c]]) + [cx, cy]
def ell_pts(p):
    cx, cy, a, b, t = p; u = np.linspace(0, 2 * np.pi, 64, endpoint=False); c, s = np.cos(t), np.sin(t); d = np.c_[a * np.cos(u), b * np.sin(u)]; return d @ np.array([[c, s], [-s, c]]) + [cx, cy]
def fit(fn, x0, steps, W):
    f = lambda p: -sc(pmask(fn(p)), W); x0 = np.asarray(x0, float); simp = np.vstack([x0] + [x0 + np.eye(len(x0))[i] * steps[i] for i in range(len(x0))])
    r = minimize(f, x0, method="Nelder-Mead", options=dict(initial_simplex=simp, maxiter=500, xatol=0.05, fatol=1e-5)); return -r.fun, r.x
def pca(W):
    y, x = np.where(W); P = np.c_[x, y].astype(float); c = P.mean(0); ev, V = np.linalg.eigh(np.cov((P - c).T)); t = np.arctan2(V[1, 1], V[0, 1]); q = (P - c) @ V[:, ::-1]
    return c, q.max(0) - q.min(0), t, c + ((q.max(0) + q.min(0)) / 2) @ V[:, ::-1].T, 2 * np.sqrt(np.maximum(ev[::-1], 1e-6))
ROWS = []; TOTS = []; SEL = {}
rng = np.random.default_rng(20261121)
for me in METHODS:
    z = np.load(A / f"candidat_{N}_honnete_{me}.npz"); names = [str(a) for a in z["names"]]; P = z["masks"].astype(bool); G = np.stack([g[a] for a in names])
    res = np.array([idx[a] % 5 == 0 for a in names]); full = np.array([bym[a]["full_watermark"] for a in names]); s = miou_vec(P, G)
    fi = np.where(full)[0]; SEL[me] = [names[i] for i in sorted(rng.choice(fi, 16, replace=False))]
    for i in fi:
        W = G[i]; h = hull_pts(W); H = pmask(h) | W if h is not None else W; r = dict(me=me, name=names[i], res=bool(res[i]), base=s[i], conv=W.sum() / H.sum(), nh=0 if h is None else len(h), h=sc(H, W), frac=W.mean())
        if h is not None:
            for n in (4, 5, 6, 8): r[f"k{n}"] = sc(pmask(reduce(h, n)), W)
        if ST == "b":
            c, ext, t, cc, ax = pca(W); y, x = np.where(W); bb = [(x.min() + x.max()) / 2, (y.min() + y.max()) / 2, x.max() - x.min() + 1, y.max() - y.min() + 1, 0.0]
            cands = [fit(rect_pts, [cc[0], cc[1], ext[0], ext[1], t], [4, 4, 8, 8, 0.2], W), fit(rect_pts, bb, [4, 4, 8, 8, 0.2], W)]; r["j"], pj = max(cands, key=lambda v: v[0])
            r["f"] = sc(pmask(rect_pts(bb)), W)   # rectangle aligné = boîte englobante de la zone filigranée
            q0 = [rect_pts(pj).ravel()] + ([reduce(h, 4).ravel()] if h is not None and len(h) >= 4 else []); r["k"] = max(fit(lambda p: p.reshape(4, 2), q, [6] * 8, W)[0] for q in q0)
            r["l"] = max(fit(ell_pts, [c[0], c[1], ax[0], ax[1], t], [4, 4, 6, 6, 0.2], W)[0], fit(ell_pts, [pj[0], pj[1], pj[2] * 0.56, pj[3] * 0.56, pj[4]], [4, 4, 6, 6, 0.2], W)[0])
        ROWS.append(r)
    R = [r for r in ROWS if r["me"] == me]; cv = np.array([r["conv"] for r in R])
    print(f"[{me}] zone filigranée des 85 pleines : aire / aire de l'enveloppe convexe, déciles 10/25/50/75/90 : {' / '.join(f'{v:.3f}' for v in np.quantile(cv, [.1, .25, .5, .75, .9]))} ; part >= 0,97 : {np.mean(cv >= 0.97):.2f} (réserve {np.mean([r['conv'] >= 0.97 for r in R if r['res']]):.2f}, hors réserve {np.mean([r['conv'] >= 0.97 for r in R if not r['res']]):.2f}) ; >= 0,95 : {np.mean(cv >= 0.95):.2f} ; >= 0,90 : {np.mean(cv >= 0.9):.2f} ; sommets de l'enveloppe médiane {np.median([r['nh'] for r in R]):.0f}")
    nf = int(full.sum()); nr = int(res.sum()); nh = len(names) - nr
    keys = [("h", "(h) enveloppe convexe"), ("k4", "enveloppe réduite à 4 sommets"), ("k5", "à 5 sommets"), ("k6", "à 6 sommets"), ("k8", "à 8 sommets")] + ([("f", "(f) rectangle aligné"), ("j", "(j) meilleur rectangle tourné"), ("k", "(k) meilleur quadrilatère"), ("l", "(l) meilleure ellipse")] if ST == "b" else [])
    for k, lab in keys:
        gr = sum(max(r.get(k, 0), r["base"]) - r["base"] for r in R if r["res"]) / nr; gh = sum(max(r.get(k, 0), r["base"]) - r["base"] for r in R if not r["res"]) / nh
        pr = sum(1 - r["base"] for r in R if r["res"]) / nr; ph = sum(1 - r["base"] for r in R if not r["res"]) / nh
        print(f"   plafond {lab:32s} : pleines seules {np.mean([r.get(k, 0) for r in R]):.4f} (base {np.mean([r['base'] for r in R]):.4f}) | gain réserve {gr:+.4f} = {gr / pr:.0%} du trou parfait | hors réserve {gh:+.4f} = {gh / ph:.0%}")
        TOTS.append(dict(tot=k, lab=lab, gr=gr, gh=gh, pr=pr, ph=ph))
T = TOTS; print("\nMOYENNE DES TROIS FILIGRANES")
for k in dict.fromkeys(r["tot"] for r in T):
    v = [r for r in T if r["tot"] == k]; print(f"   {v[0]['lab']:32s} : réserve {np.mean([r['gr'] for r in v]):+.4f} = {np.sum([r['gr'] for r in v]) / np.sum([r['pr'] for r in v]):.0%} du trou parfait ({np.mean([r['pr'] for r in v]):+.4f}) | hors réserve {np.mean([r['gh'] for r in v]):+.4f} = {np.sum([r['gh'] for r in v]) / np.sum([r['ph'] for r in v]):.0%} ({np.mean([r['ph'] for r in v]):+.4f})")
if ST == "a":
    files = {os.path.splitext(os.path.basename(f))[0]: f for f in glob.glob(str(HERE / "release/train/**/*.*"), recursive=True) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")) and "mask" not in f.lower()}
    for tag, withimg in (("planche_pleines.png", False), ("planche_pleines_images.png", True)):
        S = 256; sheet = Image.new("RGB", (8 * (S + 4), 6 * (S + 18)), (40, 40, 40)); d = ImageDraw.Draw(sheet); ok = True
        for mi, me in enumerate(METHODS):
            for j, a in enumerate(SEL[me]):
                x0 = (j % 8) * (S + 4); y0 = (mi * 2 + j // 8) * (S + 18); M = g[a]
                if withimg:
                    if a not in files: ok = False; break
                    im = np.array(Image.open(files[a]).convert("RGB").resize((S, S))).astype(float); hole = np.kron(~M, np.ones((2, 2), bool)); im[hole] = 0.45 * im[hole] + 0.55 * np.array([255, 0, 0]); cell = Image.fromarray(im.astype(np.uint8))
                else: cell = Image.fromarray(np.kron(M, np.ones((2, 2), bool)).astype(np.uint8) * 255).convert("RGB")
                sheet.paste(cell, (x0, y0 + 16)); d.text((x0 + 2, y0 + 2), f"{me[:5]} {a} {M.mean():.2f}{' R' if idx[a] % 5 == 0 else ''}", fill=(255, 255, 0))
        if ok: sheet.save(os.path.join(ME, tag)); print("planche écrite :", tag, sheet.size)
        else: print("planche non écrite (images du train introuvables) :", tag)
