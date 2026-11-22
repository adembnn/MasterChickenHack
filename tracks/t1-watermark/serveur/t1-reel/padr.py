"""Rectangles : quelle marge de toile reproduit les bits justes réels dans la boîte ? Marge fixe (px) ou proportionnelle au côté. Et pleines : marge fine."""
import sys, json, numpy as np, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parent; HERE = ROOT.parent if (ROOT.parent / "kit").exists() else ROOT.parent.parent; CACHE = ROOT / "cache"; sys.path.insert(0, str(HERE / "kit"))
from wmloc.watermarks import load_watermark
trm = json.loads((HERE / "release/train/metadata.json").read_text()); RT = np.load(CACHE / "REAL_train.npy", mmap_mode="r"); MT = np.load(CACHE / "MASK_train.npy", mmap_mode="r")
for method in sys.argv[1:]:
    wm = load_watermark(method, models_dir=HERE / "kit" / "models", device="cuda"); rng = np.random.default_rng(0); st = json.load(open(ROOT / f"stats_{method}.json"))
    others = [i for i, m in enumerate(trm) if m["method"] != method and i % 5 != 0]
    def dec(img, bits):
        l = wm.logits(torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None].float() / 255)[0].cpu().numpy() * (bits * 2 - 1); return float((l > 0).mean()), float(l.mean())
    def emb(b, bits, py, px):
        if py == 0 and px == 0: return wm.embed(b, bits)
        c = np.pad(b, ((py, py), (px, px), (0, 0)), mode="reflect"); return wm.embed(np.ascontiguousarray(c), bits)[py:py + b.shape[0], px:px + b.shape[1]]
    rects = [r for r in st if not r["full"] and r["rect"] > 0.98 and r["i"] % 5 != 0]; real = np.array([r["a_box"] for r in rects]); small = np.array([min(r["h"], r["w"]) < 64 for r in rects])
    print(f"## {method} : {len(rects)} rectangles ; réel : bits justes dans la boîte {real.mean():.3f} (petit côté < 64 px : {real[small].mean():.3f} ; >= 64 px : {real[~small].mean():.3f})")
    crops = [np.ascontiguousarray(RT[int(rng.choice(others))][r["y0"]:r["y0"] + r["h"], r["x0"]:r["x0"] + r["w"]]) for r in rects]; msgs = [rng.integers(0, 2, wm.nbits) for _ in rects]
    def run(nm, f):
        a = []
        for c, m in zip(crops, msgs):
            h, w = c.shape[:2]; py, px = f(h, w); py, px = min(py, h - 1), min(px, w - 1); a.append(dec(emb(c, m, py, px), m)[0])
        a = np.array(a); print(f"  {nm:24s} {a.mean():.3f} (petits {a[small].mean():.3f} ; grands {a[~small].mean():.3f}) ; corrélation par image avec le réel {np.corrcoef(a, real)[0, 1]:.2f}")
    for p in (0, 2, 4, 8, 12): run(f"marge fixe {p} px", lambda h, w, p=p: (p, p))
    for f_ in (0.01, 0.02, 0.03, 0.045, 0.07): run(f"marge {f_ * 100:.1f} % du côté", lambda h, w, f_=f_: (int(round(f_ * h)), int(round(f_ * w))))
    fulls = [r for r in st if r["full"] and r["i"] % 5 != 0][:40]; bases = [np.ascontiguousarray(RT[int(rng.choice(others))]) for _ in fulls]; fm = [rng.integers(0, 2, wm.nbits) for _ in fulls]
    rb = np.array([dec(np.ascontiguousarray(RT[r["i"]]), np.array([int(c) for c in trm[r["i"]]["message"]])) for r in fulls]); print(f"  pleines RÉEL : bits {rb[:, 0].mean():.3f} marge décodeur {rb[:, 1].mean():.2f}")
    for p in (0, 6, 8, 10, 12, 14, 16, 20, 24):
        d = np.array([dec(np.where(~np.array(MT[r["i"]]).astype(bool)[..., None], b, emb(b, m, p, p)), m) for r, b, m in zip(fulls, bases, fm)]); print(f"  pleines marge {p:2d} px : bits {d[:, 0].mean():.3f} marge décodeur {d[:, 1].mean():.2f}")
    del wm; torch.cuda.empty_cache()
