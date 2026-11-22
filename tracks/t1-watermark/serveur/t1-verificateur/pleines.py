"""Compare des cartes t1-reel sur les pleines (vraies et prédites), seuil 0 : hors réserve, réserve, écart apparié contre une référence."""
import sys, hashlib
import numpy as np
from verif import HERE, miou_vec, load_truth
ME, REF = sys.argv[1], sys.argv[2]; TAGS = sys.argv[3:]; rng = np.random.default_rng(3); meta, g = load_truth(check_png=False); idx = {m["image_name"]: i for i, m in enumerate(meta)}; bym = {m["image_name"]: m for m in meta}
def L(t):
    p = HERE / "t1-reel" / t / f"pred_train_{ME}.npz"; z = np.load(p); return {str(a): l.astype(np.float32) for a, l in zip(z["names"], z["logits"])}, hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
r, hr = L(REF); names = sorted(r); n = len(names); G = np.stack([g[a] for a in names]); full = np.array([bym[a]["full_watermark"] for a in names]); res = np.array([idx[a] % 5 == 0 for a in names])
R = np.stack([r[a] for a in names]); sR = miou_vec(R > 0, G); pfR = (R > 0).reshape(n, -1).mean(1) > .5
def iou_hole(P): H = ~P & full[:, None, None]; T = ~G & full[:, None, None]; u = (H | T).reshape(n, -1).sum(1); return np.where(u > 0, (H & T).reshape(n, -1).sum(1) / np.maximum(u, 1), np.nan)
print(f"{ME} : référence {REF} (sha256 {hr}) ; pleines vraies {full.sum()} ({(full & res).sum()} en réserve)")
for t in TAGS:
    d_, h = L(t); assert set(d_) == set(names); P = np.stack([d_[a] for a in names]); s = miou_vec(P > 0, G); d = s - sR; pf = (P > 0).reshape(n, -1).mean(1) > .5; hn, ho = iou_hole(P > 0), iou_hole(R > 0)
    out = f"  {t} (sha256 {h}) :"
    for lab, k in (("pleines hors réserve", full & ~res), ("pleines réserve", full & res), ("pleines prédites hors réserve", pf & ~res), ("pleines prédites réserve", pf & res), ("toutes", np.ones(n, bool))):
        b = d[k][rng.integers(0, k.sum(), (20000, k.sum()))].mean(1); out += f"\n     {lab:30s} n={k.sum():3d} : {s[k].mean():.4f} contre {sR[k].mean():.4f}, écart {d[k].mean():+.4f} IC95 [{np.percentile(b, 2.5):+.4f} ; {np.percentile(b, 97.5):+.4f}]"
    out += f"\n     IoU du trou des pleines : hors réserve {np.nanmean(hn[full & ~res]):.3f} contre {np.nanmean(ho[full & ~res]):.3f} ; réserve {np.nanmean(hn[full & res]):.3f} contre {np.nanmean(ho[full & res]):.3f}"
    print(out)
