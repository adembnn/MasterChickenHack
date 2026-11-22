"""Écart apparié entre deux dépôts d'un même filigrane. Chaque dépôt = un fichier pred_train (names, logits) ou un dossier de plis t1-reel
(réserve = moyenne des res.npz, autres images = oof.npz). Usage : python cmpnpz.py <methode> <base> <candidat> [fichier de rôles PixelSeal]
ATTENTION : hors réserve, le chiffre n'est honnête que si les deux dépôts y sont réellement hors-pli et sans réglage fait dessus."""
import sys, glob
from pathlib import Path
import numpy as np
from verif import miou_vec, boot, boot_diff, load_truth
me, pb, pc = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]); meta, g = load_truth(check_png=False); idx = {m["image_name"]: i for i, m in enumerate(meta)}; bym = {m["image_name"]: m for m in meta}
def load(p):
    if p.is_dir():
        acc = {}; out = {}
        for f in sorted(glob.glob(str(p / "fold*" / "res.npz"))):
            z = np.load(f)
            for n, l in zip(z["names"], z["logits"].astype(np.float32)): acc.setdefault(str(n), []).append(l)
            z = np.load(Path(f).with_name("oof.npz")); out.update({str(n): l for n, l in zip(z["names"], z["logits"].astype(np.float32))})
        out.update({n: np.mean(v, 0) for n, v in acc.items()}); return out
    z = np.load(p); return {str(n): l for n, l in zip(z["names"], z["logits"].astype(np.float32))}
B, C = load(pb), load(pc); names = sorted([n for n in set(B) & set(C) if bym[n]["method"] == me], key=lambda n: idx[n])
G = np.stack([g[n] for n in names]); sb = miou_vec(np.stack([B[n] for n in names]) > 0, G); sc = miou_vec(np.stack([C[n] for n in names]) > 0, G)
res = np.array([idx[n] % 5 == 0 for n in names]); full = np.array([bym[n]["full_watermark"] for n in names]); groups = {"réserve": res, "hors réserve": ~res, "toutes": np.ones(len(names), bool)}
if len(sys.argv) > 4:
    z = np.load(sys.argv[4]); role = {str(n): int(r) for n, r in zip(z["names"], z["role"])}; r = np.array([role[n] for n in names])
    groups = {"réserve": res, "pli tenu à l'écart (rôle 1)": r == 1, "apprises (rôle 2, ne compte pas)": r == 2}
print(f"[{me}] base {pb} | candidat {pc} | {len(names)} images communes")
for k, v in groups.items():
    if v.sum() < 5: continue
    d, lo, hi, pr = boot_diff(sc[v], sb[v]); la, ha = boot(sc[v])
    print(f"  {k:34s} n={v.sum():3d} candidat {sc[v].mean():.4f} [{la:.3f} ; {ha:.3f}] (pleines {sc[v & full].mean():.3f}, partielles {sc[v & ~full].mean():.3f}) base {sb[v].mean():.4f} écart {d:+.4f} IC95 [{lo:+.3f} ; {hi:+.3f}] P(mieux) {pr:.2f}{'  *' if lo > 0 or hi < 0 else ''}")
