import sys, json, numpy as np, torch
from pathlib import Path
from PIL import Image
HERE = Path("/p/scratch/training2666/destors1/MasterChickenHack/Hackathon_Watermark_Localisation_Task")
sys.path.insert(0, str(HERE / "kit"))
from wmloc.watermarks import load_watermark
from wmloc.data import LocalisationDataset
torch.set_num_threads(8)
real = LocalisationDataset(HERE / "release", "train"); meta = real.meta
def T(a): return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)[None].float() / 255
for method in ["trustmark", "stegastamp", "pixelseal"]:
    wm = load_watermark(method, models_dir=HERE / "kit" / "models", device="cuda")
    rows = []
    for i, m in enumerate(meta):
        if m["method"] != method: continue
        it = real[i]; mk = it["mask"].astype(bool); img = it["image"]; msg = it["message"]
        ys, xs = np.where(mk); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        full = bool(it.get("full_watermark")); frac = mk.mean(); rect = mk.sum() / ((y1 - y0) * (x1 - x0))
        acc = lambda a: float(((wm.logits(T(a))[0].cpu().numpy() > 0) == msg).mean())
        a_all = acc(img); a_box = acc(img[y0:y1, x0:x1])
        # image dont la zone propre est remplacée par du gris (les bits ne viennent que de la zone filigranée)
        g = img.copy(); g[~mk] = 127; a_mask = acc(g)
        g2 = img.copy(); g2[mk] = 127; a_out = acc(g2)
        rows.append(dict(i=i, full=full, frac=frac, rect=rect, h=int(y1 - y0), w=int(x1 - x0), y0=int(y0), x0=int(x0), a_all=a_all, a_box=a_box, a_mask=a_mask, a_out=a_out))
    for pop in (True, False):
        r = [x for x in rows if x["full"] == pop]
        f = np.array([x["frac"] for x in r]); print(f"\n## {method} {'PLEINES' if pop else 'PARTIELLES'} n={len(r)}")
        print("frac filigranée quantiles 0/10/25/50/75/90/100 :", np.round(np.quantile(f, [0, .1, .25, .5, .75, .9, 1]), 3))
        for k in ("a_all", "a_box", "a_mask", "a_out"):
            v = np.array([x[k] for x in r]); print(k, "moy", round(v.mean(), 3), "quantiles 10/50/90", np.round(np.quantile(v, [.1, .5, .9]), 3))
        if not pop:
            rc = np.array([x["rect"] for x in r]); isr = rc > 0.98
            print("rectangles exacts :", int(isr.sum()), "/", len(r), " rectangularité quantiles", np.round(np.quantile(rc, [0, .25, .5, .75, 1]), 2))
            for nm, sel in (("RECT", isr), ("OBJET", ~isr)):
                rr = [x for x, s in zip(r, sel) if s]
                if not rr: continue
                print(" ", nm, "n", len(rr), "a_all", round(np.mean([x["a_all"] for x in rr]), 3), "a_box", round(np.mean([x["a_box"] for x in rr]), 3),
                      "a_mask", round(np.mean([x["a_mask"] for x in rr]), 3), "frac med", round(float(np.median([x["frac"] for x in rr])), 3))
                print("   h,w:", [(x["h"], x["w"]) for x in rr][:25])
                print("   y0,x0:", [(x["y0"], x["x0"]) for x in rr][:25])
                ar = np.array([max(x["h"], x["w"]) / min(x["h"], x["w"]) for x in rr]); print("   aspect max", round(ar.max(), 2), "med", round(float(np.median(ar)), 2))
    json.dump(rows, open(HERE / "adem" / "t1-reel" / f"stats_{method}.json", "w"))
    del wm; torch.cuda.empty_cache()
