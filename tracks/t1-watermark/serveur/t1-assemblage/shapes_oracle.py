"""A1 + A3. (1) Lois de formes des vrais masques du train. (2) Plafond "oracle" des decoupes en regions :
chaque region prend l'etiquette majoritaire de la vraie reponse ; mIoU officiel du masque obtenu.
Sortie : shapes_oracle.json + tableau imprime. CPU seulement."""
import sys, json, time, numpy as np
from multiprocessing import Pool
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); from t1lib import *
from PIL import Image
from skimage.segmentation import slic, felzenszwalb
from skimage.measure import label, regionprops
from skimage.morphology import convex_hull_image

def down(m):  # regle officielle 512 -> 128
    return m.reshape(128, 4, 128, 4).mean(axis=(1, 3)) >= 0.5

def majority(seg, gt):
    n = seg.max() + 1; tot = np.bincount(seg.ravel(), minlength=n); pos = np.bincount(seg.ravel(), weights=gt.ravel().astype(np.float64), minlength=n)
    return (pos * 2 >= tot)[seg]

def one(r):
    img = load_img("train", r["image_name"]); gt = np.asarray(Image.open(HERE / "release/train/masks" / f"{r['image_name']}.png")) > 127
    g128 = r["gt"]; out = dict(name=r["image_name"], method=r["method"], full=r["full"], reserve=r["reserve"], frac=float(gt.mean()))
    # ---- lois de formes : la "forme" = zone propre des pleines, zone filigranee des partielles
    shp = ~gt if r["full"] else gt
    lab = label(shp, connectivity=2); props = regionprops(lab); areas = sorted([p.area for p in props], reverse=True)
    out["n_comp"] = len(props); out["n_comp_gt64"] = int(sum(a >= 64 for a in areas)); out["main_share"] = float(areas[0] / max(shp.sum(), 1)) if areas else 0.0
    out["small_share"] = float(sum(a for a in areas if a < 256) / max(shp.sum(), 1))
    out["holes"] = int(label(~shp, connectivity=1).max() - 1)          # composantes du complement en plus de la principale
    if shp.any():
        ys, xs = np.where(shp); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        h, w = y1 - y0, x1 - x0; out.update(bbox=[int(y0), int(x0), int(h), int(w)], rect=float(shp.sum() / (h * w)), ar=float(max(h, w) / max(min(h, w), 1)),
                                            touch=bool(y0 == 0 or x0 == 0 or y1 == 512 or x1 == 512), solidity=float(shp.sum() / max(convex_hull_image(shp).sum(), 1)),
                                            mod4=bool(y0 % 4 == 0 and x0 % 4 == 0 and h % 4 == 0 and w % 4 == 0), mod8=bool(y0 % 8 == 0 and x0 % 8 == 0 and h % 8 == 0 and w % 8 == 0))
        box = np.zeros_like(shp); box[y0:y1, x0:x1] = True; bm = box if not r["full"] else ~box
        out["o_bbox"] = float(miou(down(bm)[None], g128[None])[0])       # la forme remplacee par son rectangle englobant
    # ---- plafonds des decoupes
    for k in (8, 16, 32, 64):
        blk = gt.reshape(512 // k, k, 512 // k, k).mean(axis=(1, 3)) >= 0.5
        out[f"o_bloc{k}"] = float(miou(down(np.kron(blk, np.ones((k, k), bool)))[None], g128[None])[0])
    for n in (64, 256, 1024):
        for comp in (10, 30):
            seg = slic(img, n_segments=n, compactness=comp, start_label=0, convert2lab=True)
            out[f"o_slic{n}c{comp}"] = float(miou(down(majority(seg, gt))[None], g128[None])[0]); out[f"n_slic{n}c{comp}"] = int(seg.max() + 1)
    for sc in (100, 400, 1600):
        seg = felzenszwalb(img, scale=sc, sigma=0.8, min_size=max(20, sc // 4))
        out[f"o_felz{sc}"] = float(miou(down(majority(seg, gt))[None], g128[None])[0]); out[f"n_felz{sc}"] = int(seg.max() + 1)
    return out

if __name__ == "__main__":
    rows = load_train(); t0 = time.time()
    with Pool(4) as p: res = p.map(one, rows, chunksize=4)
    (ME / "shapes_oracle.json").write_text(json.dumps(res))
    print("fait", len(res), round(time.time() - t0), "s")
    import collections
    keys = [k for k in res[0] if k.startswith("o_")]
    print("\n=== PLAFONDS ORACLE (mIoU si chaque region prenait la vraie etiquette majoritaire) ; n = nombre moyen de regions")
    print(f"{'decoupe':14s} {'n':>6s} | " + " | ".join(f"{m[:5]:>5s} pl. part." for m in METHODS) + " | tous")
    for k in keys:
        nk = "n_" + k[2:]; nn = np.mean([r[nk] for r in res]) if nk in res[0] else {"o_bloc8": 4096, "o_bloc16": 1024, "o_bloc32": 256, "o_bloc64": 64}.get(k, 1)
        cells = []
        for m in METHODS:
            a = np.mean([r[k] for r in res if r["method"] == m and r["full"] and k in r]); b = np.mean([r[k] for r in res if r["method"] == m and not r["full"] and k in r])
            cells.append(f"{a:.3f} {b:.3f}  ")
        print(f"{k[2:]:14s} {nn:6.0f} | " + " | ".join(cells) + f" | {np.mean([r[k] for r in res if k in r]):.3f}")
    print("\n=== LOIS DE FORMES (forme = zone propre des pleines, zone filigranee des partielles), mediane [p10 ; p90]")
    def q(v): v = np.array(v, float); return f"{np.median(v):.3f} [{np.quantile(v, .1):.3f} ; {np.quantile(v, .9):.3f}]"
    for m in METHODS:
        for full in (True, False):
            S = [r for r in res if r["method"] == m and r["full"] == full]
            print(f"{m:10s} {'pleines' if full else 'partielles':10s} n={len(S):3d} fraction filigranee {q([r['frac'] for r in S])} | composantes {q([r['n_comp'] for r in S])} (>=64px: {q([r['n_comp_gt64'] for r in S])})"
                  f" | part principale {q([r['main_share'] for r in S])} | trous {q([r['holes'] for r in S])}")
            print(f"{'':21s} rectangularite {q([r['rect'] for r in S])} | rectangles parfaits (>0.98) {np.mean([r['rect'] > .98 for r in S]):.2f} | solidite {q([r['solidity'] for r in S])}"
                  f" | rapport h/l {q([r['ar'] for r in S])} max {max(r['ar'] for r in S):.2f} | touche le bord {np.mean([r['touch'] for r in S]):.2f}"
                  f" | rect. alignes x4 {np.mean([r['mod4'] for r in S if r['rect'] > .98] or [0]):.2f} x8 {np.mean([r['mod8'] for r in S if r['rect'] > .98] or [0]):.2f}")
