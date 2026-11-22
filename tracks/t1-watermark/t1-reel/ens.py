"""Moyenne de plusieurs cartes d'un filigrane, calculée sur le PC à partir des fichiers pred_train (hors-pli + réserve) et pred_test rapatriés.
Usage : python ens.py <methode> <dossier1> <dossier2> ... [--out DOSSIER]   (chaque dossier contient pred_train_<methode>.npz et pred_test_<methode>.npz)
Affiche le mIoU officiel (seuil 0) : validation croisée (136 images hors réserve), pleines, partielles, réserve (34), et l'IoU du trou des pleines."""
import sys, json, itertools, numpy as np
from pathlib import Path
REL = Path(r"E:/EAC-lab/t1/release/train"); meta = json.loads((REL / "metadata.json").read_text()); z = np.load(REL / "masks_128.npz"); Y = dict(zip(z["names"].tolist(), z["masks"].astype(bool)))
pos = {m["image_name"]: i for i, m in enumerate(meta)}; full = {m["image_name"]: bool(m.get("full_watermark")) for m in meta}
def miou(p, g):
    s = 0.0
    for c in (False, True):
        u = ((p == c) | (g == c)).sum(); s += 1.0 if u == 0 else ((p == c) & (g == c)).sum() / u
    return s / 2
def iou0(p, g): u = ((~p) | (~g)).sum(); return 1.0 if u == 0 else ((~p) & (~g)).sum() / u
def load(d, m, s): z = np.load(Path(d) / f"pred_{s}_{m}.npz"); return z["names"].tolist(), z["logits"].astype(np.float32)
def report(m, dirs):
    names, L = None, []
    for d in dirs:
        n, l = load(d, m, "train"); assert names is None or n == names; names = n; L.append(l)
    A = np.mean(L, 0); res = np.array([pos[n] % 5 == 0 for n in names]); fu = np.array([full[n] for n in names]); sc = np.array([miou(a > 0, Y[n]) for a, n in zip(A, names)]); h = np.array([iou0(a > 0, Y[n]) for a, n in zip(A, names)])
    return dict(cv=sc[~res].mean(), cv_pl=sc[~res & fu].mean(), cv_pa=sc[~res & ~fu].mean(), res=sc[res].mean(), res_pl=sc[res & fu].mean(), res_pa=sc[res & ~fu].mean(), trou_cv=h[~res & fu].mean(), trou_res=h[res & fu].mean())
if __name__ == "__main__":
    m = sys.argv[1]; args = [a for a in sys.argv[2:] if a != "--out"]; out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    dirs = [a for a in args if a != out]
    fmt = lambda r: f"CV {r['cv']:.3f} (pleines {r['cv_pl']:.3f}, partielles {r['cv_pa']:.3f}) | réserve {r['res']:.3f} (pleines {r['res_pl']:.3f}, partielles {r['res_pa']:.3f}) | trou des pleines {r['trou_cv']:.3f} / {r['trou_res']:.3f}"
    for d in dirs: print(f"{Path(d).parent.name + '/' + Path(d).name:34s}", fmt(report(m, [d])))
    for k in range(2, len(dirs) + 1):
        for c in itertools.combinations(dirs, k): print("moyenne " + " + ".join(Path(x).parent.name for x in c).ljust(40), fmt(report(m, list(c))))
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        for s in ("train", "test"):
            names, L = None, []
            for d in dirs: n, l = load(d, m, s); assert names is None or n == names; names = n; L.append(l)
            np.savez_compressed(Path(out) / f"pred_{s}_{m}.npz", names=np.array(names), logits=np.mean(L, 0).astype(np.float16))
        print("écrit", out, "| moyenne de", len(dirs), "cartes |", fmt(report(m, [out])))
