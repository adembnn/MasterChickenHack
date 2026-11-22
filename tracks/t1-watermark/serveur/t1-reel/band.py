"""Les pleines dont le trou est une bande ou un coin : la zone filigranée y est-elle un RECTANGLE (image source un peu plus petite que l'image finale, étendue par
outpainting) ? Si oui, le filigrane est calé sur ce rectangle et non sur l'image entière. Test par le décodeur (StegaStamp est sensible au cadre à 4 px près) :
accord au message en décodant (a) l'image entière, (b) l'image entière sur toile +12 px, (c) le rognage de la boîte de la zone filigranée, (d) cette boîte sur toile +4 px.
Séparé selon la forme de la zone filigranée : rectangle exact (bande / coin) ou non (trous en taches)."""
import sys, json, numpy as np, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parent; HERE = ROOT.parent if (ROOT.parent / "kit").exists() else ROOT.parent.parent; CACHE = ROOT / "cache"; sys.path.insert(0, str(HERE / "kit"))
from wmloc.watermarks import load_watermark
trm = json.loads((HERE / "release/train/metadata.json").read_text()); RT = np.load(CACHE / "REAL_train.npy", mmap_mode="r"); MT = np.load(CACHE / "MASK_train.npy", mmap_mode="r")
def pad(x, p): return np.pad(x, ((p, p), (p, p), (0, 0)), mode="reflect") if p else x
for method in sys.argv[1:]:
    wm = load_watermark(method, models_dir=HERE / "kit" / "models", device="cuda")
    def marg(x, sgn):
        l = wm.logits(torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None].float() / 255)[0].cpu().numpy() * sgn; return float(l.mean()), float((l > 0).mean())
    rows = []
    for i, m in enumerate(trm):
        if m["method"] != method or not m["full_watermark"] or i % 5 == 0: continue
        M = np.array(MT[i]).astype(bool); ys, xs = np.where(M); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1; rect = M.sum() / ((y1 - y0) * (x1 - x0)); frac = M.mean()
        touch = int(y0 > 0) + int(x0 > 0) + int(y1 < 512) + int(x1 < 512)          # nombre de côtés de l'image que la zone filigranée ne touche pas
        x = np.ascontiguousarray(RT[i]); sgn = np.array([1.0 if c == "1" else -1.0 for c in m["message"]]); c = np.ascontiguousarray(x[y0:y1, x0:x1])
        rows.append(dict(i=i, rect=rect, frac=frac, touch=touch, h=int(y1 - y0), w=int(x1 - x0), a=marg(x, sgn), b=marg(pad(x, 12), sgn), c=marg(c, sgn), d=marg(pad(c, 4), sgn), e=marg(pad(c, 12), sgn)))
    R_ = [r for r in rows if r["rect"] > 0.98 and r["frac"] < 0.97]; B = [r for r in rows if r["rect"] <= 0.98]; O = [r for r in rows if r["rect"] > 0.98 and r["frac"] >= 0.97]
    print(f"\n## {method} : {len(rows)} pleines hors réserve | zone filigranée = rectangle exact plus petit que l'image : {len(R_)} | presque toute l'image (> 97 %) : {len(O)} | autre forme (trous en taches) : {len(B)}")
    for nm, L in (("rectangle exact (trou = bande ou coin)", R_), ("autre forme (taches)", B), ("presque toute l'image", O)):
        if not L: continue
        f = lambda k: f"{np.mean([r[k][0] for r in L]):.2f} ({np.mean([r[k][1] for r in L]):.3f})"
        print(f"  {nm:40s} n={len(L):2d} | accord (bits justes) : image entière {f('a')} | image sur toile +12 {f('b')} | boîte {f('c')} | boîte sur toile +4 {f('d')} | boîte sur toile +12 {f('e')}")
    if R_: print("  exemples de rectangles (h, w, part filigranée, côtés non touchés) :", [(r["h"], r["w"], round(r["frac"], 2), r["touch"]) for r in R_[:14]])
    del wm; torch.cuda.empty_cache()
