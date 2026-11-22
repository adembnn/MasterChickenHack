"""Contrôle de la recette : mêmes masques que le réel, fonds propres réels (images des autres filigranes), bits justes comparés au réel."""
import sys, json, time, numpy as np, torch
from pathlib import Path
HERE = Path("/p/scratch/training2666/destors1/MasterChickenHack/Hackathon_Watermark_Localisation_Task")
sys.path.insert(0, str(HERE / "kit"))
from wmloc.watermarks import load_watermark
from wmloc.data import LocalisationDataset
torch.set_num_threads(8); rng = np.random.default_rng(0)
real = LocalisationDataset(HERE / "release", "train"); meta = real.meta
def T(a): return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)[None].float() / 255
for method in sys.argv[1:]:
    wm = load_watermark(method, models_dir=HERE / "kit" / "models", device="cuda")
    st = {r["i"]: r for r in json.load(open(HERE / "adem" / "t1-reel" / f"stats_{method}.json"))}
    others = [i for i, m in enumerate(meta) if m["method"] != method and i % 5 != 0]
    acc = lambda a, b: float(((wm.logits(T(a))[0].cpu().numpy() > 0) == b).mean())
    R = {k: [] for k in ("full", "rect", "objA", "objB")}; tt = []
    for i, m in enumerate(meta):
        if m["method"] != method or i % 5 == 0 or len(R["full"]) + len(R["rect"]) + len(R["objA"]) > 75: continue
        it = real[i]; M = it["mask"].astype(bool); s = st[i]; C = real[int(rng.choice(others))]["image"]; bits = rng.integers(0, 2, wm.nbits)
        y0, x0, h, w = s["y0"], s["x0"], s["h"], s["w"]
        def gray(im): g = im.copy(); g[~M] = 127; return g
        if s["full"]:
            t = time.time(); W = wm.embed(C, bits); tt.append(time.time() - t); im = np.where(M[..., None], W, C)
            R["full"].append((s["a_all"], acc(im, bits), np.abs(W.astype(int) - C).mean()))
        elif s["rect"] > 0.98:
            im = C.copy(); im[y0:y0 + h, x0:x0 + w] = wm.embed(np.ascontiguousarray(C[y0:y0 + h, x0:x0 + w]), bits)
            R["rect"].append((s["a_box"], acc(im[y0:y0 + h, x0:x0 + w], bits), s["a_all"], acc(im, bits)))
        else:
            W = wm.embed(C, bits); im = np.where(M[..., None], W, C)
            R["objA"].append((s["a_all"], acc(im, bits), s["a_box"], acc(im[y0:y0 + h, x0:x0 + w], bits), s["a_mask"], acc(gray(im), bits)))
            try:
                im = C.copy(); im[y0:y0 + h, x0:x0 + w] = wm.embed(np.ascontiguousarray(C[y0:y0 + h, x0:x0 + w]), bits); im = np.where(M[..., None], im, C)
                R["objB"].append((s["a_all"], acc(im, bits), s["a_box"], acc(im[y0:y0 + h, x0:x0 + w], bits), s["a_mask"], acc(gray(im), bits)))
            except Exception as e: print("objB err", h, w, str(e)[:80])
    print(f"\n## {method}  embed 512 : {np.mean(tt[1:])*1000:.0f} ms/image")
    v = np.array(R["full"]); print("PLEINES n", len(v), " a_all réel", v[:, 0].mean().round(3), " fabriqué", v[:, 1].mean().round(3), " |résidu| moyen (niveaux)", v[:, 2].mean().round(2))
    v = np.array(R["rect"]); print("RECT n", len(v), " a_box réel", v[:, 0].mean().round(3), " fabriqué (cadre = rectangle)", v[:, 1].mean().round(3), " corr", np.corrcoef(v[:, 0], v[:, 1])[0, 1].round(2), " a_all réel", v[:, 2].mean().round(3), "fabriqué", v[:, 3].mean().round(3))
    for k, nm in (("objA", "cadre = image entière"), ("objB", "cadre = boîte de l'objet")):
        v = np.array(R[k]); print("OBJET", nm, "n", len(v), " a_all réel/fab", v[:, 0].mean().round(3), v[:, 1].mean().round(3), " a_box réel/fab", v[:, 2].mean().round(3), v[:, 3].mean().round(3), " a_mask réel/fab", v[:, 4].mean().round(3), v[:, 5].mean().round(3), " corr a_all", np.corrcoef(v[:, 0], v[:, 1])[0, 1].round(2))
    del wm; torch.cuda.empty_cache()
