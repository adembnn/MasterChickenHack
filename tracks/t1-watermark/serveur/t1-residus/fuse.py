"""Fusion « réseau + rectangle trouvé » : pour chaque image classée partielle dont la boîte a une confiance z > seuil, le masque du réseau
est remplacé par le rectangle (cadre affiné rétréci de 4 px si disponible, sinon boîte du décodeur). Local, numpy seul.
Usage : python fuse.py <methode> <pred_train.npz> <pred_test.npz> <boxes_train.json> <boxes_test.json> <nom_sortie> [seuil_z] [seuil_r3]"""
import sys, os, json, numpy as np
SH = float(os.environ.get("SHRINK", "3.5"))
from pathlib import Path
method, ptr, pte, btr, bte, name = sys.argv[1:7]; ZMIN = float(sys.argv[7]) if len(sys.argv) > 7 else 6.5; R3MAX = float(sys.argv[8]) if len(sys.argv) > 8 else 0.7
R = Path("E:/EAC-lab/t1/release/train"); meta = json.loads((R / "metadata.json").read_text()); m128 = np.load(R / "masks_128.npz"); G = dict(zip(m128["names"].tolist(), m128["masks"]))
def miou(p, g):
    s = 0.0
    for c in (0, 1):
        u = ((p == c) | (g == c)).sum(); s += 1.0 if u == 0 else ((p == c) & (g == c)).sum() / u
    return s / 2
def down(mk): return (mk.reshape(128, 4, 128, 4).mean((1, 3)) >= 0.5)
def rect_logits(r):
    if "frame" in r and r.get("r3", 9) < R3MAX: f = r["frame"]; b = (f[0] + SH, f[1] + SH, f[2] - SH, f[3] - SH); kind = "r3"
    else: b = r["box"]; kind = "dec"
    mk = np.zeros((512, 512), np.float32); mk[max(0, int(round(b[0]))):max(0, int(round(b[2]))), max(0, int(round(b[1]))):max(0, int(round(b[3])))] = 1
    return np.where(down(mk), 6.0, -6.0).astype(np.float16), kind
def fuse(pred, boxes):
    z = np.load(pred, allow_pickle=True); names = z["names"].tolist(); L = z["logits"].astype(np.float16).copy(); B = {r["name"]: r for r in json.load(open(boxes))}; n = {"dec": 0, "r3": 0}
    for i, nm in enumerate(names):
        r = B.get(nm)
        if r and "box" in r and r["z"] > ZMIN: L[i], k = rect_logits(r); n[k] += 1
    return names, z["logits"], L, n
names, L0, L1, n = fuse(ptr, btr); info = {e["image_name"]: (i, bool(e.get("full_watermark"))) for i, e in enumerate(meta) if e["method"] == method}
s0 = np.array([miou(L0[i] > 0, G[nm] > 0) for i, nm in enumerate(names)]); s1 = np.array([miou(L1[i] > 0, G[nm] > 0) for i, nm in enumerate(names)])
res = np.array([info[nm][0] % 5 == 0 for nm in names]); full = np.array([info[nm][1] for nm in names]); ch = np.array([(L0[i] > 0).tolist() != (L1[i] > 0).tolist() for i in range(len(names))])
print("FUSE %s base=%s | rectangles posés sur le train : %d (dont affinés R3 : %d), seuil z > %.1f" % (method, Path(ptr).name, n["dec"] + n["r3"], n["r3"], ZMIN))
for nm, sel in (("toutes (170)", np.ones(len(names), bool)), ("RÉSERVE", res), ("réserve, partielles", res & ~full), ("réserve, pleines", res & full), ("hors réserve", ~res), ("hors réserve, partielles", ~res & ~full)):
    d = s1[sel] - s0[sel]; print("  %-26s n=%3d  avant %.4f  après %.4f  écart %+.4f  (erreur type de l'écart %.4f ; images modifiées %d, améliorées %d, dégradées %d)" % (nm, sel.sum(), s0[sel].mean(), s1[sel].mean(), d.mean(), d.std() / np.sqrt(max(1, sel.sum())), (ch & sel).sum(), (d > 1e-6).sum(), (d < -1e-6).sum()))
out = Path("E:/EAC-lab/t1s/t1-residus/out"); np.savez_compressed(out / f"pred_train_{name}_{method}.npz", names=np.array(names), logits=L1)
tn, T0, T1, nt = fuse(pte, bte); np.savez_compressed(out / f"pred_test_{name}_{method}.npz", names=np.array(tn), logits=T1)
print("  test : %d images, rectangles posés %d (dont affinés R3 : %d) ; part filigranée moyenne avant %.3f après %.3f ; fichiers pred_{train,test}_%s_%s.npz" % (len(tn), nt["dec"] + nt["r3"], nt["r3"], (T0 > 0).mean(), (T1 > 0).mean(), name, method))
