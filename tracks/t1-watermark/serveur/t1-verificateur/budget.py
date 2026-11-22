"""Budget d'erreur du candidat 6 (masques honnêtes du train) avec oracles : où sont les points perdus ?
Types d'image tirés de la vérité : propre (0 % filigrané), pleine entière (100 %), pleine à trou (full_watermark, 0 < part < 1),
partielle rectangle, partielle objet. Oracles : une seule chose corrigée à la fois."""
import sys, json
import numpy as np
from verif import HERE, METHODS, miou_vec, load_truth
N = sys.argv[1] if len(sys.argv) > 1 else "6"
A = HERE / "t1-assemblage"; meta, g = load_truth(check_png=False); idx = {m["image_name"]: i for i, m in enumerate(meta)}; bym = {m["image_name"]: m for m in meta}
try:
    from scipy.ndimage import gaussian_filter
except Exception: gaussian_filter = None
def ious(P, G):
    o = []
    for c in (True, False):
        i = ((P == c) & (G == c)).reshape(len(P), -1).sum(1); u = ((P == c) | (G == c)).reshape(len(P), -1).sum(1); o.append(np.where(u == 0, 1.0, i / np.maximum(u, 1)))
    return o
def isrect(M):
    if not M.any(): return False
    y, x = np.where(M); return M.sum() >= 0.98 * (y.max() + 1 - y.min()) * (x.max() + 1 - x.min())
def typ(a):
    M = g[a]; f = M.mean()
    if f == 0: return "propre (0 %)"
    if f == 1: return "pleine entière (100 %)"
    if bym[a]["full_watermark"]: return "pleine à trou"
    return "partielle rectangle" if (isrect(M) or isrect(~M)) else "partielle objet"
TYPES = ["propre (0 %)", "pleine entière (100 %)", "pleine à trou", "partielle rectangle", "partielle objet"]
SRC = {"pixelseal": {True: (HERE / "t1-pixelseal/pred_train_pixelseal.npz", 4), False: (A / "pred_train_rang-pixelseal-G_pixelseal.npz", 0)},
       "trustmark": {True: (HERE / "t1-reel/pred_train_trustmark.npz", 0), False: (A / "pred_train_rangP-reel-mixr_trustmark.npz", 0)},
       "stegastamp": {True: (A / "pred_train_rang-reel-pred_stegastamp.npz", 0), False: (A / "pred_train_rang-reel-pred_stegastamp.npz", 0)}}
def lg(p): z = np.load(p); return {str(a): l.astype(np.float32) for a, l in zip(z["names"], z["logits"])}
def best_thr(L, G):
    ts = np.concatenate([[-np.inf], np.quantile(L, np.linspace(0, 1, 81)), [np.inf]]); P = L[None] > ts[:, None, None]
    return miou_vec(P, np.broadcast_to(G, P.shape)).max()
# ---- code officiel : pleine entière prédite à 95 % de 1
try:
    sys.path.insert(0, str(HERE / "kit")); from wmloc.utils import miou_image
    G1 = np.ones((128, 128), np.uint8); P = G1.copy(); P[:6, :] = 0; P[6, :51] = 0
    print(f"[code officiel] vérité 100 % filigranée : prédiction tout-1 -> {miou_image(G1, G1):.4f} ; prédiction à {P.mean():.3f} de 1 -> {miou_image(P, G1):.4f} ; un seul pixel à 0 -> {miou_image(np.where(np.arange(16384).reshape(128, 128) == 0, 0, 1).astype(np.uint8), G1):.4f}")
    G0 = np.zeros((128, 128), np.uint8); P0 = G0.copy(); P0[0, 0] = 1; print(f"[code officiel] vérité 0 % : tout-0 -> {miou_image(G0, G0):.4f} ; un seul pixel à 1 -> {miou_image(P0, G0):.4f}")
    Gh = G1.copy(); Gh[:13, :] = 0; print(f"[code officiel] vérité à {Gh.mean():.2f} de 1 (trou de 10 %) : prédiction tout-1 -> {miou_image(G1, Gh):.4f}")
except Exception as e: print("code officiel non importé :", e)
TOT = {}
for me in METHODS:
    z = np.load(A / f"candidat_{N}_honnete_{me}.npz"); names = [str(a) for a in z["names"]]; P = z["masks"].astype(bool); G = np.stack([g[a] for a in names])
    res = np.array([idx[a] % 5 == 0 for a in names]); full = np.array([bym[a]["full_watermark"] for a in names]); ty = np.array([typ(a) for a in names]); frac = G.reshape(len(G), -1).mean(1)
    s = miou_vec(P, G); i1, i0 = ious(P, G); pf = P.reshape(len(P), -1).mean(1)
    print(f"\n===== {me} : candidat {N}, réserve {s[res].mean():.4f} (n={res.sum()}), les 170 : {s.mean():.4f}")
    print(f"vérité : part filigranée des pleines (full_watermark) min {frac[full].min():.3f} médiane {np.median(frac[full]):.3f} max {frac[full].max():.3f} ; exactement 100 % : {np.mean(frac[full] == 1):.3f} ; partielles : min {frac[~full].min():.3f} médiane {np.median(frac[~full]):.3f} max {frac[~full].max():.3f} ; exactement 0 % : {np.mean(frac[~full] == 0):.3f}")
    print(f"prédiction : pleines sorties exactement tout-1 : {np.mean(pf[full] == 1):.3f} (réserve {np.mean(pf[full & res] == 1):.3f}) ; à plus de 95 % de 1 sans être tout-1 : {np.mean((pf[full] > 0.95) & (pf[full] < 1)):.3f} ; partielles sorties tout-0 : {np.mean(pf[~full] == 0):.3f} ; part prédite médiane des pleines {np.median(pf[full]):.3f}")
    for lab, k in (("réserve", res), ("les 170", np.ones(len(names), bool))):
        loss = (1 - s[k]).sum(); print(f"-- {lab} | type | n | score | IoU filigrané | IoU propre | part filigranée vraie | histogramme du score [0-0,2 | 0,2-0,4 | 0,4-0,6 | 0,6-0,8 | 0,8-1] | part de la perte")
        for t in TYPES:
            m = k & (ty == t)
            if not m.any(): continue
            h = np.histogram(s[m], [0, 0.2, 0.4, 0.6, 0.8, 1.0001])[0]
            print(f"   {t:24s} | {m.sum():3d} | {s[m].mean():.3f} | {i1[m].mean():.3f} | {i0[m].mean():.3f} | {frac[m].mean():.3f} | {' | '.join(str(v) for v in h)} | {(1 - s[m]).sum() / loss:.2f}")
    # ---- oracles
    L = {b: lg(SRC[me][b][0]) for b in (True, False)}
    def logit(a, b):
        l = L[b][a]; sg = SRC[me][b][1]; return gaussian_filter(l, sg) if (sg and gaussian_filter is not None) else l
    O = {"candidat": s.copy()}
    Pa = P.copy(); Pa[ty == "propre (0 %)"] = False; Pa[full] = True; O["(a) type parfait : propre -> tout 0, pleine (full_watermark) -> tout 1"] = miou_vec(Pa, G)
    Pa2 = P.copy(); Pa2[frac == 0] = False; Pa2[frac == 1] = True; O["(a') tout 0 / tout 1 seulement si la vérité l'est exactement"] = miou_vec(Pa2, G)
    Pb = P.copy(); Pb[~full] = G[~full]; O["(b) étendue parfaite sur les partielles"] = miou_vec(Pb, G)
    Pb2 = P.copy(); Pb2[full] = G[full]; O["(b') étendue parfaite sur les pleines"] = miou_vec(Pb2, G)
    Pc = P.copy(); Pc[full] = True; O["(c) pleines forcées à tout 1"] = miou_vec(Pc, G)
    d = np.array([max(s[i], best_thr(logit(a, bool(full[i])), G[i])) for i, a in enumerate(names)]); O["(d) seuil optimal par image (logits de la source retenue)"] = d
    dp = s.copy(); dp[full] = d[full]; O["(d-pleines) seuil optimal par image, pleines seulement"] = dp
    dq = s.copy(); dq[~full] = d[~full]; O["(d-partielles) seuil optimal par image, partielles seulement"] = dq
    e = np.maximum.reduce([s, miou_vec(np.zeros_like(P), G), miou_vec(np.ones_like(P), G)]); O["(e) le meilleur de {candidat, tout 0, tout 1} par image"] = e
    for k2, v in O.items(): print(f"   oracle {k2:75s} : réserve {v[res].mean():.4f} ({v[res].mean() - s[res].mean():+.4f}) | les 170 : {v.mean():.4f} ({v.mean() - s.mean():+.4f})")
    for k2, v in O.items(): TOT.setdefault(k2, []).append((v[res].mean(), v.mean()))
print("\n===== MOYENNE DES TROIS FILIGRANES")
b = np.mean(TOT["candidat"], 0)
for k2, v in TOT.items(): m = np.mean(v, 0); print(f"   {k2:82s} : réserve {m[0]:.4f} ({m[0] - b[0]:+.4f}) | les 510 : {m[1]:.4f} ({m[1] - b[1]:+.4f})")
