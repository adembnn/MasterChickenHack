"""Mission : IoU de la classe PROPRE (le trou) sur les images pleines. Diagnostic sur les prédictions hors-pli (et réserve) d'une variante :
IoU du trou par taille de trou, trous manqués, faux trous, ce que donnerait le meilleur seuil par image (classement bon mais seuil mauvais ?),
et l'effet de règles simples (seuil dédié aux pleines, suppression des petits faux trous, plus grand trou seul)."""
import sys, json, numpy as np
from pathlib import Path
from scipy import ndimage as ndi
ROOT = Path(__file__).resolve().parent; HERE = ROOT.parent if (ROOT.parent / "kit").exists() else ROOT.parent.parent; CACHE = ROOT / "cache"
tag, method = sys.argv[1], sys.argv[2]; trm = json.loads((HERE / "release/train/metadata.json").read_text()); Y = np.load(CACHE / "Y128_train.npy")
P = ROOT / "runs" / tag / method; oi, ol, rl = [], [], []
for f in range(5):
    z = np.load(P / f"fold{f}" / "oof.npz"); oi += list(z["idx"]); ol.append(z["logits"].astype(np.float32)); z = np.load(P / f"fold{f}" / "res.npz"); ri = z["idx"]; rl.append(z["logits"].astype(np.float32))
oi = np.array(oi); ol = np.concatenate(ol); rl = np.mean(rl, 0)
def iou0(p, g): u = ((p == 0) | (g == 0)).sum(); return 1.0 if u == 0 else ((p == 0) & (g == 0)).sum() / u
def iou1(p, g): u = ((p == 1) | (g == 1)).sum(); return 1.0 if u == 0 else ((p == 1) & (g == 1)).sum() / u
def clean_small(m, k):                                   # m = masque filigrané ; on rebouche les trous prédits de moins de k cases
    lab, n = ndi.label(~m)
    if n == 0: return m
    sz = ndi.sum(~m, lab, np.arange(1, n + 1)); return ~np.isin(lab, 1 + np.where(sz >= k)[0])
def biggest(m):
    lab, n = ndi.label(~m)
    if n <= 1: return m
    sz = ndi.sum(~m, lab, np.arange(1, n + 1)); return ~(lab == 1 + int(np.argmax(sz)))
for nm, idx, L in (("HORS-PLI", oi, ol), ("RÉSERVE", ri, rl)):
    full = np.array([bool(trm[i]["full_watermark"]) for i in idx]); I = idx[full]; Lf = L[full]; G = Y[I].astype(bool); h = 1 - G.mean((1, 2)); Pm = Lf > 0
    a0 = np.array([iou0(p, g) for p, g in zip(Pm, G)]); a1 = np.array([iou1(p, g) for p, g in zip(Pm, G)]); ph = 1 - Pm.mean((1, 2))
    rec = np.array([((~p) & (~g)).sum() / max((~g).sum(), 1) for p, g in zip(Pm, G)]); pre = np.array([((~p) & (~g)).sum() / max((~p).sum(), 1) for p, g in zip(Pm, G)])
    print(f"\n## {tag} {method} {nm} : {len(I)} pleines | IoU du trou {a0.mean():.3f} | IoU filigrané {a1.mean():.3f} | mIoU {((a0 + a1) / 2).mean():.3f}")
    print(f"   trous sans aucune case trouvée : {int((rec == 0).sum())} | rappel moyen du trou {rec.mean():.2f} | précision moyenne {pre.mean():.2f} | part de trou vraie {h.mean():.3f}, prédite {ph.mean():.3f}")
    for lo, hi in ((0, .02), (.02, .08), (.08, .2), (.2, 1)):
        s = (h >= lo) & (h < hi)
        if s.sum(): print(f"   trou {lo:.2f}-{hi:.2f} (n={int(s.sum()):2d}) : IoU du trou {a0[s].mean():.3f} | rappel {rec[s].mean():.2f} | précision {pre[s].mean():.2f}")
    ths = np.linspace(-6, 6, 49); per = np.array([[iou0(l > t, g) for t in ths] for l, g in zip(Lf, G)])
    print(f"   meilleur seuil PAR IMAGE (plafond du classement) : IoU du trou {per.max(1).mean():.3f} | meilleur seuil commun aux pleines : {per.mean(0).max():.3f} à {ths[per.mean(0).argmax()]:+.2f}")
    for k in (4, 16, 48): print(f"   petits faux trous (< {k} cases) rebouchés : IoU du trou {np.mean([iou0(clean_small(p, k), g) for p, g in zip(Pm, G)]):.3f}", end=" ;")
    print(f"\n   plus grand trou prédit seul : {np.mean([iou0(biggest(p), g) for p, g in zip(Pm, G)]):.3f}")
