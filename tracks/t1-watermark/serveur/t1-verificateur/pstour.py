"""Où part le score PixelSeal qui manque : tranches sur les prédictions hors-pli d'une carte t1-reel (défaut ps_f_x1+f_s1+f_deca), au seuil 0 et
après assemblage (candidat_<n>_cv). 170 images = 140 hors réserve + 30 réserve."""
import sys
import numpy as np
from scipy import ndimage as ndi
from verif import HERE, miou_vec, load_truth
ME = sys.argv[1] if len(sys.argv) > 1 else "pixelseal"; TAG = sys.argv[2] if len(sys.argv) > 2 else "ps_f_x1+f_s1+f_deca"; CAND = sys.argv[3] if len(sys.argv) > 3 else "35"
meta, g = load_truth(check_png=False); idx = {m["image_name"]: i for i, m in enumerate(meta)}; bym = {m["image_name"]: m for m in meta}
z = np.load(HERE / "t1-reel" / TAG / f"pred_train_{ME}.npz"); names = [str(a) for a in z["names"]]; L = z["logits"].astype(np.float32); G = np.stack([g[a] for a in names]); N = len(names)
full = np.array([bym[a]["full_watermark"] for a in names]); res = np.array([idx[a] % 5 == 0 for a in names]); f = np.load(HERE / "t1-assemblage" / "feat_train.npz"); acc = dict(zip([str(a) for a in f["names"]], f["acc"])); A = np.array([acc[a] for a in names])
c = np.load(HERE / "t1-assemblage" / f"candidat_{CAND}_cv_{ME}.npz"); cm = {str(a): m.astype(bool) for a, m in zip(c["names"], c["masks"])}; C = np.stack([cm[a] for a in names])
zf = np.load("familles.npz"); fam = dict(zip(zf["names"].tolist(), zf["fam"].tolist()))
P0 = L > 0; RARE = np.where(full[:, None, None], ~G, G); rf = RARE.reshape(N, -1).mean(1)          # zone rare : trou propre d'une pleine, zone filigranée d'une partielle
def iou(p, t):
    p = p.reshape(len(p), -1); t = t.reshape(len(t), -1); u = (p | t).sum(1); return np.where(u > 0, (p & t).sum(1) / np.maximum(u, 1), 1.0)
def rare_of(P): return np.where(full[:, None, None], ~P, P)
s0 = miou_vec(P0, G); sc = miou_vec(C, G); r0 = iou(rare_of(P0), RARE); rc = iou(rare_of(C), RARE)
def auc(i):                                              # la carte range-t-elle les cases rares avant les autres ?
    v = (-L[i] if full[i] else L[i]).ravel(); t = RARE[i].ravel(); n1 = t.sum(); n0 = len(t) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = v.argsort().argsort() + 1; return (r[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
AU = np.array([auc(i) for i in range(N)])
def hit(i):                                              # la case la plus sûre de la classe rare (carte lissée) tombe-t-elle dans la vraie zone ?
    v = ndi.gaussian_filter(-L[i] if full[i] else L[i], 2); return bool(RARE[i].ravel()[v.argmax()])
HIT = np.array([hit(i) for i in range(N)])
print(f"{ME} {TAG} : {N} images ({(~res).sum()} hors réserve + {res.sum()} réserve) ; pleines {full.sum()}, partielles {(~full).sum()}")
print(f"seuil 0 : {s0.mean():.4f} (hors réserve {s0[~res].mean():.4f}, réserve {s0[res].mean():.4f}) | candidat {CAND} en validation croisée : {sc.mean():.4f} (hors réserve {sc[~res].mean():.4f}, réserve {sc[res].mean():.4f})")
def row(lab, k):
    if k.sum() == 0: return
    print(f"  {lab:34s} n={k.sum():3d} ({res[k].sum():2d} rés.) | seuil 0 : mIoU {s0[k].mean():.3f} IoU rare {r0[k].mean():.3f} | cand {CAND} : mIoU {sc[k].mean():.3f} IoU rare {rc[k].mean():.3f} | AUC {np.nanmean(AU[k]):.3f} touchée {HIT[k].mean():.2f} | points perdus {k.sum() / N * (1 - sc[k].mean()):.4f} ({k.sum() * (1 - sc[k].mean()) / (N * (1 - sc.mean())):.0%})")
print("-- 1. population vraie"); row("pleines", full); row("partielles", ~full)
print("-- 2. taille de la zone rare (part de l'image)")
for pop, k0 in (("PLEINES", full), ("PARTIELLES", ~full)):
    for lo, hi in ((0, .01), (.01, .03), (.03, .06), (.06, .15), (.15, 1.01)): row(f"{pop} {lo:.2f}-{min(hi, 1):.2f}", k0 & (rf >= lo) & (rf < hi))
print("-- 3. forme")
def bbfill(m):
    ys, xs = np.where(m); return m.sum() / ((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)) if len(ys) else 0
def minside(m):
    ys, xs = np.where(m); return min(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1) * 4 if len(ys) else 0
BF = np.array([bbfill(RARE[i]) for i in range(N)]); rect = BF >= 0.95; MS = np.array([minside(RARE[i]) for i in range(N)]); ncomp = np.array([ndi.label(RARE[i])[1] for i in range(N)])
row("partielles rectangle", ~full & rect); row("  rectangle, petit côté < 64 px", ~full & rect & (MS < 64)); row("  rectangle, petit côté 64-128 px", ~full & rect & (MS >= 64) & (MS < 128)); row("  rectangle, petit côté >= 128 px", ~full & rect & (MS >= 128))
row("partielles objet", ~full & ~rect); row("  objet en 1 morceau", ~full & ~rect & (ncomp == 1)); row("  objet en plusieurs morceaux", ~full & ~rect & (ncomp > 1))
F = np.array([fam.get(a, "-") for a in names])
for k_, l_ in (("A", "pleines famille A (taches lisses)"), ("B", "pleines famille B (silhouette)"), ("C", "pleines famille C (rectangle)")): row(l_, full & (F == k_))
print("-- 4. population devinée par le décodeur (pleine si bits justes > seuil)")
th = max(np.unique(A), key=lambda t: (np.mean((A > t) == full), -abs(t - 0.75))); pf = A > th; print(f"  seuil {th:.3f} : {np.mean(pf == full):.3f} de bonnes classes ; {(pf != full).sum()} images mal classées")
row("mal classées", pf != full); row("  pleine prise pour partielle", full & ~pf); row("  partielle prise pour pleine", ~full & pf)
print("-- 5. où sont les cases fausses (candidat) : distance au bord de la vraie zone, en cases de 4 px")
E = C != G; D = np.stack([np.maximum(ndi.distance_transform_edt(G[i]), ndi.distance_transform_edt(~G[i])) for i in range(N)])   # distance de chaque case à la classe opposée
for pop, k0 in (("pleines", full), ("partielles", ~full), ("toutes", np.ones(N, bool))):
    e = E[k0]; d = D[k0]; tot = e.sum(); out = f"  {pop:10s} cases fausses {e.mean():.3%} de l'image ; "
    out += " | ".join(f"à <= {r} cases du bord : {(e & (d <= r)).sum() / tot:.0%}" for r in (1, 2, 4, 8)); print(out)
    for r in (1, 2, 4):
        fx = np.where(E & (D <= r), G, C); fy = np.where(E & (D > r), G, C)
        print(f"     si toutes les cases fausses à <= {r} du bord étaient corrigées : {miou_vec(fx[k0], G[k0]).mean():.4f} ; si seules celles à > {r} l'étaient : {miou_vec(fy[k0], G[k0]).mean():.4f} (actuel {sc[k0].mean():.4f})")
print("-- 6. erreurs de type : zone rare manquée, débordement, fausse zone ailleurs (candidat)")
Rp = rare_of(C)
for pop, k0 in (("pleines", full), ("partielles", ~full)):
    n = int(k0.sum()); miss = (RARE & ~Rp)[k0].sum(); fa = (~RARE & Rp)[k0]; dil = np.stack([ndi.binary_dilation(RARE[i], iterations=8) for i in np.where(k0)[0]]); near = (fa & dil).sum()
    print(f"  {pop:10s} zone rare vraie {RARE[k0].mean():.3%} de l'image, prédite {Rp[k0].mean():.3%} ; manquée {miss / RARE[k0].sum():.0%} de la vraie zone ; fausses cases rares : {fa.sum() / max(Rp[k0].sum(), 1):.0%} de la zone prédite, dont {near / max(fa.sum(), 1):.0%} à moins de 8 cases de la vraie zone ; images sans aucune case rare prédite {np.mean(Rp[k0].reshape(n, -1).sum(1) == 0):.2f} ; sans aucun recouvrement {np.mean((RARE & Rp)[k0].reshape(n, -1).sum(1) == 0):.2f}")
print("-- 7. plafonds (en connaissant la réponse)")
def best_thr(i):
    qs = np.quantile(L[i], np.linspace(0.002, 0.998, 250)); return max(float(miou_vec((L[i] > t)[None], G[i][None])[0]) for t in qs)
BT = np.array([best_thr(i) for i in range(N)]); print(f"  meilleur seuil par image sur la carte : {BT.mean():.4f} (pleines {BT[full].mean():.4f}, partielles {BT[~full].mean():.4f}) contre candidat {sc.mean():.4f}")
def topk(i):                                             # taille de la zone rare connue : on prend les k cases les plus sûres
    v = (-L[i] if full[i] else L[i]).ravel(); k = int(RARE[i].sum()); m = np.zeros(v.size, bool); m[np.argsort(-v)[:k]] = True; m = m.reshape(128, 128); return m if not full[i] else ~m
TK = np.stack([topk(i) for i in range(N)]); stk = miou_vec(TK, G); print(f"  taille de la zone rare connue (les k cases les plus sûres) : {stk.mean():.4f} (pleines {stk[full].mean():.4f}, partielles {stk[~full].mean():.4f})")
S = {}
for fn in (f"sam_train_{ME}.npz", f"sam_train_{ME}_pleines.npz"):
    try:
        zz = np.load(f"/tmp/eac-t1-partage/{fn}")
        for a in names:
            if a in zz.files and a not in S: K = int(zz[a + "__n"]); S[a] = np.unpackbits(zz[a])[:K * 16384].reshape(K, 128, 128).astype(bool)
    except Exception as e: print("  SAM", fn, e)
def sam_best(i):
    M = S.get(names[i]); t = RARE[i]
    if M is None or len(M) == 0: return None
    io = iou(M, np.broadcast_to(t, M.shape)); o = np.argsort(-io)[:12]; best = M[o[0]]; b = io[o[0]]
    for _ in range(3):                                   # union gloutonne de 4 masques au plus
        v, j = max((float(iou((best | M[j])[None], t[None])[0]), j) for j in o)
        if v <= b + 1e-6: break
        b = v; best = best | M[j]
    return best if not full[i] else ~best
SB = [sam_best(i) for i in range(N)]; has = np.array([m is not None for m in SB]); SM = np.stack([m if m is not None else C[i] for i, m in enumerate(SB)]); ssam = miou_vec(SM, G); print(f"  masques SAM disponibles pour {has.sum()} images sur {N}")
def crow(lab, k):
    if k.sum(): print(f"  {lab:34s} n={k.sum():3d} | candidat {sc[k].mean():.3f} | meilleure union de masques SAM {ssam[k].mean():.3f} | meilleur des deux {np.maximum(ssam, sc)[k].mean():.3f} | gain possible sur {ME} {k.sum() / N * (np.maximum(ssam, sc)[k].mean() - sc[k].mean()):+.4f}")
crow("partielles objet", ~full & ~rect); crow("partielles rectangle", ~full & rect); crow("pleines A", full & (F == "A")); crow("pleines B", full & (F == "B")); crow("pleines, trou < 3 %", full & (rf < .03)); crow("pleines, trou >= 3 %", full & (rf >= .03)); crow("partielles < 3 %", ~full & (rf < .03)); crow("partielles >= 3 %", ~full & (rf >= .03)); crow("toutes", np.ones(N, bool))
# les vrais rectangles sont-ils la boîte englobante d'un masque SAM (objet de l'image) ?
def bb(m):
    ys, xs = np.where(m); o = np.zeros_like(m)
    if len(ys): o[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
    return o
def sam_bbox(i):
    M = S.get(names[i]); t = RARE[i]
    if M is None or len(M) == 0: return 0.0, 0
    B = np.stack([bb(m) for m in M]); io = iou(B, np.broadcast_to(t, B.shape)); return float(io.max()), len(M)
RB = {i: sam_bbox(i) for i in np.where(~full & rect)[0]}; v = np.array([RB[i][0] for i in RB]); ms = np.array([MS[i] for i in RB])
print(f"  rectangles ({len(v)}) : IoU de la meilleure boîte englobante d'un masque SAM avec le vrai rectangle : médiane {np.median(v):.3f}, >= 0,8 pour {np.mean(v >= .8):.0%}, >= 0,6 pour {np.mean(v >= .6):.0%} ; petit côté < 64 px : médiane {np.median(v[ms < 64]):.3f}, >= 0,8 pour {np.mean(v[ms < 64] >= .8):.0%} ; candidats par image {np.mean([RB[i][1] for i in RB]):.0f}")
ys = [np.where(RARE[i])[0] for i in RB]; xs = [np.where(RARE[i])[1] for i in RB]
print("  rectangles : hauteur x largeur en px (triés par aire) : " + " ".join(f"{(y.max() - y.min() + 1) * 4}x{(x.max() - x.min() + 1) * 4}" for y, x in sorted(zip(ys, xs), key=lambda t: len(t[0]))))
print("  rectangles : touchent le bord de l'image : " + f"{np.mean([y.min() == 0 or x.min() == 0 or y.max() == 127 or x.max() == 127 for y, x in zip(ys, xs)]):.0%}")
perf = lambda k: (sc.sum() + (1 - sc[k]).sum()) / N
print(f"  si ces images étaient parfaites : pleines {perf(full):.4f} ; partielles {perf(~full):.4f} ; zone rare < 3 % {perf(rf < .03):.4f} ; rectangles {perf(~full & rect):.4f} ; objets {perf(~full & ~rect):.4f} ; mal classées {perf(pf != full):.4f}")
cst = miou_vec(np.broadcast_to(full[:, None, None], G.shape).copy(), G); print(f"  constante par population vraie (tout filigrané / tout propre) : {cst.mean():.4f} ; sur zone rare < 3 % : constante {cst[rf < .03].mean():.3f}, candidat {sc[rf < .03].mean():.3f}, seuil 0 {s0[rf < .03].mean():.3f}")
np.savez_compressed(f"pstour_{ME}.npz", names=np.array(names), sc=sc, s0=s0, rf=rf, full=full, rect=rect, auc=AU, hit=HIT, ssam=ssam, ms=MS)
