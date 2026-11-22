"""Note l'inférence zoomée (zoom_<methode>_<tag>.npz) : écart apparié contre l'inférence normale, par population et par taille de zone.
La population vient du drapeau mesurable au test (bits justes de l'image entière > 0,75), pas de la vérité."""
import sys
from pathlib import Path
import numpy as np
from verif import HERE, miou_vec, boot_diff, load_truth
me, tag = sys.argv[1], sys.argv[2]
meta, g = load_truth(check_png=False); bym = {m["image_name"]: m for m in meta}
z = np.load(Path(__file__).with_name(f"zoom_{me}_{tag}.npz")); names = [str(n) for n in z["names"]]; grp = z["grp"]
L0, Z2, Z4 = (z[k].astype(np.float32) for k in ("L0", "Z2", "Z4")); G = np.stack([g[n] for n in names])
f = np.load(HERE / "t1-assemblage" / "feat_train.npz"); acc = {str(n): float(a) for n, a in zip(f["names"], f["acc"])}
flag = np.array([acc[n] > 0.75 for n in names]); full = np.array([bym[n]["full_watermark"] for n in names])
fr = G.reshape(len(G), -1).mean(1); amin = np.minimum(fr, 1 - fr)
def isrect(n):
    M = g[n]; ys, xs = np.where(M); return (not bym[n]["full_watermark"]) and M.sum() >= 0.98 * (ys.max() + 1 - ys.min()) * (xs.max() + 1 - xs.min())
rect = np.array([isrect(n) for n in names])
print(f"[{me} {tag}] {len(names)} images ({(grp == 'res').sum()} réserve en ensemble, {(grp == 'oof').sum()} hors-pli) | drapeau exact {np.mean(flag == full):.3f}")
V = {"Z2 seul": Z2, "Z4 seul": Z4, "max(L0, Z2)": np.maximum(L0, Z2), "moyenne(L0, Z2)": (L0 + Z2) / 2, "max(L0, Z2, Z4)": np.maximum(np.maximum(L0, Z2), Z4), "moyenne(L0, Z2, Z4)": (L0 + Z2 + Z4) / 3,
     "min(L0, Z2)": np.minimum(L0, Z2)}
s0 = miou_vec(L0 > 0, G)
sub = {"toutes": np.ones(len(names), bool), "classées partielles": ~flag, "classées pleines": flag, "partielles, zone < 3 %": (~full) & (amin < 0.03), "partielles, zone >= 3 %": (~full) & (amin >= 0.03),
       "rectangles": rect, "rectangles < 3 %": rect & (amin < 0.03), "rectangles >= 3 %": rect & (amin >= 0.03), "objets": (~full) & (~rect), "pleines, trou < 3 %": full & (amin < 0.03), "réserve seule": grp == "res", "hors-pli seul": grp == "oof"}
print("normal (L0) :", {k: round(float(s0[v].mean()), 4) for k, v in sub.items()})
for name, L in V.items():
    s = miou_vec(L > 0, G); print(f"-- {name}")
    for k, v in sub.items():
        if v.sum() < 5: continue
        d, lo, hi, pr = boot_diff(s[v], s0[v])
        print(f"   {k:26s} n={v.sum():3d} {s[v].mean():.4f} contre {s0[v].mean():.4f} écart {d:+.4f} IC95 [{lo:+.3f} ; {hi:+.3f}] P(mieux) {pr:.2f}{'  *' if lo > 0 or hi < 0 else ''}")
for name in ("max(L0, Z2)", "moyenne(L0, Z2)", "moyenne(L0, Z2, Z4)", "Z2 seul"):
    L = np.where(rect[:, None, None], V[name], L0); s = miou_vec(L > 0, G); d, lo, hi, pr = boot_diff(s, s0)
    print(f"ORACLE rectangle ({name} si VRAI rectangle, normal sinon) toutes n={len(s)} {s.mean():.4f} contre {s0.mean():.4f} écart {d:+.4f} IC95 [{lo:+.3f} ; {hi:+.3f}]")
# règle d'aiguillage telle qu'au test : zoom pour les images classées partielles, normal pour les classées pleines
for name in ("max(L0, Z2)", "moyenne(L0, Z2)", "max(L0, Z2, Z4)", "moyenne(L0, Z2, Z4)", "Z2 seul"):
    L = np.where(flag[:, None, None], L0, V[name]); s = miou_vec(L > 0, G)
    for k in ("toutes", "réserve seule", "hors-pli seul"):
        v = sub[k]; d, lo, hi, pr = boot_diff(s[v], s0[v])
        print(f"AIGUILLÉ ({name} si classée partielle) {k:14s} n={v.sum():3d} {s[v].mean():.4f} contre {s0[v].mean():.4f} écart {d:+.4f} IC95 [{lo:+.3f} ; {hi:+.3f}] P(mieux) {pr:.2f}{'  *' if lo > 0 or hi < 0 else ''}")
