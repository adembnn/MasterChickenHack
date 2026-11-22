"""Familles de trous des pleines : A = taches lisses aléatoires, B = silhouette d'objet ou région de l'image, C = rectangle.
Critère appris sur 47 étiquettes posées à l'œil sur la planche de contact (48 pleines, dont 1 rectangle), puis appliqué aux 255 pleines.
Puis : score et IoU du trou par famille (candidat N), et A contre B lisible depuis le trou PRÉDIT (donc utilisable au test) ?"""
import sys, os
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel, binary_erosion, label
from skimage.measure import perimeter, regionprops
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, LeaveOneOut, StratifiedKFold
from sklearn.metrics import roc_auc_score
from verif import HERE, METHODS, miou_vec, load_truth
N = sys.argv[1] if len(sys.argv) > 1 else "11"
A = HERE / "t1-assemblage"; meta, g = load_truth(check_png=False); idx = {m["image_name"]: i for i, m in enumerate(meta)}; bym = {m["image_name"]: m for m in meta}
EYE = {"pixelseal": "BBBAABABACAABAAA", "trustmark": "ABBBABABABBABAAA", "stegastamp": "AABBBABBABABBAAA"}; FIRST = {"pixelseal": "63ab2489dca0", "trustmark": "59738e22b1ba", "stegastamp": "857f0e93299e"}
def edges(a):
    im = np.asarray(Image.open(HERE / "release" / "train" / "images" / f"{a}.png").convert("L").resize((256, 256), Image.BILINEAR), np.float32); im = gaussian_filter(im, 1.0)
    m = np.hypot(sobel(im, 0), sobel(im, 1)); return (np.argsort(np.argsort(m.ravel())) / m.size).reshape(256, 256)
def feats(Z, E):
    """Z : zone 128x128 (trou vrai ou trou prédit), E : rang du gradient de l'image (256x256)."""
    if Z.sum() < 4: return None
    y, x = np.where(Z); fill = Z.sum() / ((y.max() + 1 - y.min()) * (x.max() + 1 - x.min())); lab, n = label(Z, np.ones((3, 3))); sz = np.bincount(lab.ravel())[1:]
    S = gaussian_filter(Z.astype(np.float32), 1.5) > 0.5; liss = (Z & S).sum() / max((Z | S).sum(), 1); P = perimeter(Z, 8); isoq = min(4 * np.pi * Z.sum() / max(P, 1) ** 2, 1.5)
    rp = max(regionprops(lab), key=lambda r: r.area); Z2 = np.kron(Z, np.ones((2, 2), bool)); bd = Z2 & ~binary_erosion(Z2, np.ones((3, 3))); bd[:3] = bd[-3:] = False; bd[:, :3] = bd[:, -3:] = False
    al = E[bd].mean() if bd.sum() >= 8 else 0.5
    return dict(fill=fill, liss=liss, isoq=isoq, solid=rp.solidity, align=al, lncomp=np.log(int((sz >= max(4, 0.05 * Z.sum())).sum())), lnall=np.log(n), larea=np.log(Z.mean()))
KEYS = ["liss", "isoq", "solid", "align", "lncomp", "lnall", "larea"]
R = []; rng = np.random.default_rng(20261121)
for me in METHODS:
    z6 = np.load(A / "candidat_6_honnete_" f"{me}.npz"); n6 = [str(a) for a in z6["names"]]; f6 = np.array([bym[a]["full_watermark"] for a in n6]); sel = [n6[i] for i in sorted(rng.choice(np.where(f6)[0], 16, replace=False))]; assert sel[0] == FIRST[me], (sel[0], FIRST[me])
    eye = dict(zip(sel, EYE[me])); z = np.load(A / f"candidat_{N}_honnete_{me}.npz"); names = [str(a) for a in z["names"]]; P = z["masks"].astype(bool); G = np.stack([g[a] for a in names]); s = miou_vec(P, G)
    for i, a in enumerate(names):
        if not bym[a]["full_watermark"]: continue
        E = edges(a); Z = ~G[i]; Zp = ~P[i]; f = feats(Z, E); fp = feats(Zp, E); hi = (Z & Zp).sum() / max((Z | Zp).sum(), 1)
        R.append(dict(me=me, name=a, res=idx[a] % 5 == 0, score=s[i], hiou=hi, eye=eye.get(a), f=f, fp=fp, frac=Z.mean()))
X = lambda rows, k="f": np.array([[r[k][q] for q in KEYS] for r in rows])
lab = [r for r in R if r["eye"] in ("A", "B") and r["f"] is not None]; y = np.array([r["eye"] == "B" for r in lab]); mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
pl = cross_val_predict(mk(), X(lab), y, cv=LeaveOneOut(), method="predict_proba")[:, 1]
print(f"[critère] {len(lab)} étiquettes à l'œil (A {int((~y).sum())}, B {int(y.sum())}) ; régression logistique sur {KEYS} : exactitude en retirant une image à la fois {np.mean((pl > 0.5) == y):.2f}, AUC {roc_auc_score(y, pl):.2f}")
for q in KEYS:
    v = X(lab)[:, KEYS.index(q)]; au = roc_auc_score(y, v); ths = np.unique(v); acc = max(max(np.mean((v > t) == y), np.mean((v <= t) == y)) for t in ths); print(f"   un seul critère « {q} » : AUC {max(au, 1 - au):.2f} (B {'plus haut' if au > 0.5 else 'plus bas'}), meilleure exactitude par seuil {acc:.2f} ; médiane A {np.median(v[~y]):.2f}, B {np.median(v[y]):.2f}")
clf = mk().fit(X(lab), y)
for r in R:
    if r["f"] is None: r["fam"] = "A"; continue
    r["pB"] = float(clf.predict_proba(X([r]))[0, 1]); r["fam"] = "C" if r["f"]["fill"] >= 0.95 else ("B" if r["pB"] > 0.5 else "A")
print(f"   accord du critère final avec l'œil sur les 48 : {np.mean([r['fam'] == r['eye'] for r in R if r['eye']]):.2f} (optimiste : appris sur ces images)")
print("\n[familles] parts par filigrane (255 pleines), critère automatique")
for me in list(METHODS) + ["tous"]:
    rr = [r for r in R if me in (r["me"], "tous")]; tl = sum(1 - r["score"] for r in rr)
    print(f" {me:10s} n={len(rr)} : " + " | ".join(f"{k} {np.mean([r['fam'] == k for r in rr]):.2f} (réserve {np.mean([r['fam'] == k for r in rr if r['res']]):.2f})" for k in "ABC"))
    for k in "ABC":
        q = [r for r in rr if r["fam"] == k]
        if not q: continue
        qr = [r for r in q if r["res"]]; qh = [r for r in q if not r["res"]]; m = lambda v, key: np.mean([r[key] for r in v]) if v else float("nan")
        print(f"      {k} n={len(q):3d} : score candidat {N} {m(q, 'score'):.3f} (réserve {m(qr, 'score'):.3f} n={len(qr)}, hors réserve {m(qh, 'score'):.3f}) | IoU du trou {m(q, 'hiou'):.3f} (réserve {m(qr, 'hiou'):.3f}, hors réserve {m(qh, 'hiou'):.3f}) | trou raté (IoU < 0,1) {np.mean([r['hiou'] < 0.1 for r in q]):.2f} | taille médiane {np.median([r['frac'] for r in q]):.3f} | part de la perte des pleines {sum(1 - r['score'] for r in q) / tl:.2f}")
# par taille dans chaque famille
print("\n[familles x taille] IoU du trou, tous filigranes : " + " ; ".join(f"{k} " + ", ".join(f"{lab_} {np.mean([r['hiou'] for r in R if r['fam'] == k and lo <= r['frac'] < hi] or [np.nan]):.2f} (n={sum(r['fam'] == k and lo <= r['frac'] < hi for r in R)})" for lab_, lo, hi in (("< 2 %", 0, .02), ("2-8 %", .02, .08), ("> 8 %", .08, 1))) for k in "AB"))
# ---- A contre B depuis le trou prédit (utilisable au test)
ab = [r for r in R if r["fam"] in "AB"]; ok = [r for r in ab if r["fp"] is not None]; yb = np.array([r["fam"] == "B" for r in ok])
print(f"\n[depuis la carte] pleines A ou B : {len(ab)} ; trou prédit non vide (>= 4 pixels) : {len(ok)} ({len(ok) / len(ab):.2f}) ; parmi elles, trou localisé (IoU >= 0,3) : {np.mean([r['hiou'] >= 0.3 for r in ok]):.2f}")
pp = cross_val_predict(mk(), X(ok, "fp"), yb, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
print(f"   A contre B à partir des mêmes mesures prises sur le trou PRÉDIT (validation croisée 5 plis) : AUC {roc_auc_score(yb, pp):.2f}, exactitude {np.mean((pp > 0.5) == yb):.2f} (classe majoritaire : {max(yb.mean(), 1 - yb.mean()):.2f})")
loc = np.array([r["hiou"] >= 0.3 for r in ok]); print(f"   même chose, trous localisés seulement (n={loc.sum()}) : AUC {roc_auc_score(yb[loc], pp[loc]):.2f} ; trous non localisés (n={(~loc).sum()}) : AUC {roc_auc_score(yb[~loc], pp[~loc]) if 0 < yb[~loc].sum() < (~loc).sum() else float('nan'):.2f}")
for q in KEYS:
    v = X(ok, "fp")[:, KEYS.index(q)]; au = roc_auc_score(yb, v); print(f"      « {q} » du trou prédit seul : AUC {max(au, 1 - au):.2f}")
np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "familles.npz"), names=np.array([r["name"] for r in R]), fam=np.array([r["fam"] for r in R]))
