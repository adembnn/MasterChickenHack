"""Mesure décisive : la carte locale de « re-tamponnage » sépare-t-elle les TROUS des images pleines (et les objets) ?
Carte R(p) = énergie locale de ce que le tampon ajouterait avec le VRAI message / avec K messages au hasard (lissage gaussien), sur une toile
= image prolongée en miroir de p px par côté (recette de t1-reel), p dans une liste ; chemin exact du tampon (`tmpl`), tout filigrane.
Sorties par population : AUC par pixel (zone filigranée contre zone propre, score = -R) à marge fixe et à la meilleure marge par image (plafond)."""
import argparse, time
import numpy as np, torch, torch.nn.functional as F
from rlib import *
ap = argparse.ArgumentParser(); ap.add_argument("--method", required=True); ap.add_argument("--pop", default="pleines"); ap.add_argument("--n", type=int, default=20)
ap.add_argument("--K", type=int, default=4); ap.add_argument("--sig", type=float, default=6.0); ap.add_argument("--margins", default="0,4,8,12,16,20,24"); a = ap.parse_args()
dev = "cuda"; wm = load(a.method, dev); rng = np.random.default_rng(0); MARG = [int(v) for v in a.margins.split(",")]
sel = []
for e in meta("train"):
    if e["method"] != a.method or e["gidx"] % 5 == 0: continue
    mk = read_mask(e["image_name"]); full = bool(e.get("full_watermark")); ys, xs = np.where(mk); rect = (not full) and mk[ys.min():ys.max() + 1, xs.min():xs.max() + 1].mean() > 0.98
    kind = "pleines" if full else ("rectangles" if rect else "objets")
    if kind == a.pop and 0.02 < mk.mean() < 0.98: sel.append((e, mk))
sel = sel[:a.n]; print("R3AUC", a.method, a.pop, "n=%d (hors réserve, zone minoritaire >= 2 %% de l'image), lissage %.0f px, K=%d" % (len(sel), a.sig, a.K), flush=True)
A = np.zeros((len(sel), len(MARG))); IN = np.zeros_like(A); OUT_ = np.zeros_like(A); t0 = time.time()
for i, (e, mk) in enumerate(sel):
    img = read_img("train", e["image_name"]); m = bits_of(e); msgs = np.concatenate([m[None], rng.integers(0, 2, (a.K, wm.nbits)).astype(np.float32)]); g = downsample_mask(mk)
    for j, p in enumerate(MARG):
        c = np.pad(img, ((p, p), (p, p), (0, 0)), mode="reflect") if p else img
        en = tmpl(wm, np.ascontiguousarray(c), msgs).pow(2).sum(1, keepdim=True)[:, :, p:p + 512, p:p + 512]; en = gauss(en, a.sig)
        den = en[1:].mean(0, keepdim=True); R = ((en[:1] + 0.05 * den.mean()) / (den + 0.05 * den.mean())); R = F.avg_pool2d(R, 4)[0, 0].cpu().numpy()
        A[i, j] = auc(-R[g > 0], -R[g == 0]); IN[i, j] = R[g > 0].mean(); OUT_[i, j] = R[g == 0].mean()
    if i % 5 == 0: print("  %d/%d %d s | part filigranée %.2f | AUC par marge : %s" % (i + 1, len(sel), time.time() - t0, mk.mean(), " ".join("%d:%.2f" % (p, v) for p, v in zip(MARG, A[i]))), flush=True)
print("RESULT", a.method, a.pop, "n=%d" % len(sel))
print("  marge (px)        : " + "  ".join("%5d" % p for p in MARG))
print("  AUC moyenne       : " + "  ".join("%.3f" % v for v in A.mean(0)))
print("  rapport dedans    : " + "  ".join("%.3f" % v for v in IN.mean(0)))
print("  rapport dehors    : " + "  ".join("%.3f" % v for v in OUT_.mean(0)))
b = A.argmax(1); print("  plafond (meilleure marge par image, choisie avec la vérité) : AUC moyenne %.3f, médiane %.3f ; images à AUC > 0,8 : %d/%d ; marge choisie : %s" % (A.max(1).mean(), np.median(A.max(1)), (A.max(1) > 0.8).sum(), len(sel), np.bincount(b, minlength=len(MARG)).tolist()))
