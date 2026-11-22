"""Nouvelle carte PixelSeal de t1-reel (ps_pop_x2+x2b+x1+dx__st+x2) : recalcul au seuil 0 contre la carte précédente (ps_f_x2+f_x1+f_dx), composition
reconstituée à partir des composantes, petits rectangles, et sensibilité au choix de la composition « partielles »."""
import numpy as np
from pathlib import Path
from verif import HERE, miou_vec, load_truth
rng = np.random.default_rng(20261122); R = HERE / "t1-reel"; C = Path("comp"); meta, g = load_truth(check_png=False); idx = {m["image_name"]: i for i, m in enumerate(meta)}; bym = {m["image_name"]: m for m in meta}
def L(p):
    z = np.load(p); return {str(n): l.astype(np.float32) for n, l in zip(z["names"], z["logits"])}
NEW = L(R / "ps_pop_x2+x2b+x1+dx__st+x2" / "pred_train_pixelseal.npz"); OLD = L(R / "ps_f_x2+f_x1+f_dx" / "pred_train_pixelseal.npz"); names = sorted(NEW); n = len(names)
K = {t: L(C / f"{t}_train_pixelseal.npz") for t in ("f_x2", "f_x2b", "f_x1", "f_dx", "f_st")}; S = lambda d: np.stack([d[a] for a in names]); G = np.stack([g[a] for a in names])
full = np.array([bym[a]["full_watermark"] for a in names]); res = np.array([idx[a] % 5 == 0 for a in names]); RARE = np.where(full[:, None, None], ~G, G)
def iou(p, t):
    p = p.reshape(len(p), -1); t = t.reshape(len(t), -1); u = (p | t).sum(1); return np.where(u > 0, (p & t).sum(1) / np.maximum(u, 1), 1.0)
def bb(m):
    ys, xs = np.where(m); return (m.sum() / ((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)), min(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1) * 4) if len(ys) else (0, 0)
B = np.array([bb(RARE[i]) for i in range(n)]); rect = ~full & (B[:, 0] >= .95); small = rect & (B[:, 1] < 64)
def ci(x): b = x[rng.integers(0, len(x), (20000, len(x)))].mean(1); return np.percentile(b, 2.5), np.percentile(b, 97.5)
N, O = S(NEW), S(OLD); sN, sO = miou_vec(N > 0, G), miou_vec(O > 0, G); d = sN - sO
print("seuil 0, nouvelle carte contre ps_f_x2+f_x1+f_dx")
for lab, k in (("hors réserve (validation croisée)", ~res), ("réserve", res), ("170 images", np.ones(n, bool)), ("pleines", full), ("partielles", ~full), ("partielles hors réserve", ~full & ~res), ("partielles réserve", ~full & res), ("pleines réserve", full & res)):
    lo, hi = ci(d[k]); print(f"  {lab:34s} n={k.sum():3d} : {sN[k].mean():.4f} contre {sO[k].mean():.4f}, écart {d[k].mean():+.4f} IC95 [{lo:+.4f} ; {hi:+.4f}], mieux/pires {(d[k] > 1e-9).sum()}/{(d[k] < -1e-9).sum()}")
rN = iou(np.where(full[:, None, None], ~(N > 0), N > 0), RARE); rO = iou(np.where(full[:, None, None], ~(O > 0), O > 0), RARE)
for lab, k in (("vrais rectangles, petit côté < 64 px, hors réserve", small & ~res), ("idem, réserve", small & res), ("idem, tous", small)):
    lo, hi = ci((rN - rO)[k]); print(f"  IoU de la zone, {lab} (n={k.sum()}) : {rN[k].mean():.3f} contre {rO[k].mean():.3f}, écart {(rN - rO)[k].mean():+.3f} IC95 [{lo:+.3f} ; {hi:+.3f}] ; mIoU {sN[k].mean():.3f} contre {sO[k].mean():.3f}")
# composition reconstituée
A = np.mean([S(K[t]) for t in ("f_x2", "f_x2b", "f_x1", "f_dx")], 0); P = np.mean([S(K[t]) for t in ("f_st", "f_x2")], 0)
for lab, D in (("la moyenne « pleines »", A), ("la moyenne « partielles »", P), ("la moyenne des deux", (A + P) / 2)):
    pf = (D > 0).reshape(n, -1).mean(1) > 0.5; rec = np.where(pf[:, None, None], A, P); print(f"  composition reconstituée, population décidée sur {lab} : écart max avec le fichier déposé {np.abs(rec - N).max():.4f} ; images différentes {(np.abs(rec - N).reshape(n, -1).max(1) > 0.05).sum()} ; population juste {np.mean(pf == full):.3f}")
pf = (A > 0).reshape(n, -1).mean(1) > 0.5
# test : même contrôle
NT = L(R / "ps_pop_x2+x2b+x1+dx__st+x2" / "pred_test_pixelseal.npz"); KT = {t: L(C / f"{t}_test_pixelseal.npz") for t in K}; nt = sorted(NT); ST = lambda dd: np.stack([dd[a] for a in nt]); AT = np.mean([ST(KT[t]) for t in ("f_x2", "f_x2b", "f_x1", "f_dx")], 0); PT = np.mean([ST(KT[t]) for t in ("f_st", "f_x2")], 0)
for lab, D in (("la moyenne « pleines »", AT), ("la moyenne « partielles »", PT)):
    pft = (D > 0).reshape(len(nt), -1).mean(1) > 0.5; rec = np.where(pft[:, None, None], AT, PT); print(f"  TEST, population décidée sur {lab} : écart max {np.abs(rec - ST(NT)).max():.4f} ; images différentes {(np.abs(rec - ST(NT)).reshape(len(nt), -1).max(1) > 0.05).sum()} sur {len(nt)} ; pleines prédites {pft.mean():.3f}")
# sensibilité au choix de la composition des partielles (pleines fixées)
print("  choix de la composition « partielles » (population prédite, pleines = moyenne des 4) : validation croisée / réserve / partielles hors réserve / partielles réserve")
for combo in (("f_x2",), ("f_st",), ("f_st", "f_x2"), ("f_st", "f_x2", "f_x1"), ("f_st", "f_x2", "f_x2b"), ("f_st", "f_x2", "f_x2b", "f_x1", "f_dx"), ("f_x2", "f_x2b", "f_x1", "f_dx")):
    Pc = np.mean([S(K[t]) for t in combo], 0); rec = np.where(pf[:, None, None], A, Pc); s = miou_vec(rec > 0, G); print(f"     {' + '.join(combo):34s} {s[~res].mean():.4f} / {s[res].mean():.4f} / {s[~full & ~res].mean():.4f} / {s[~full & res].mean():.4f}")
