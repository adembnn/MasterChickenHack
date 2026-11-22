"""Outils communs de la session t1-assemblage : chargement du train/test, reserve, plis, mIoU officiel vectorise.
Marche tel quel sur Julich (adem/t1-assemblage/) et sur la H200 (~/eac/t1/t1-assemblage/)."""
import json, numpy as np
from pathlib import Path
ME = Path(__file__).resolve().parent                     # dossier de cette session
SESS = ME.parent                                         # dossier qui contient les dossiers des sessions (adem/ sur Julich, ~/eac/t1 sur la H200)
HERE = next(q for q in (ME.parent, ME.parent.parent) if (q / "kit").exists())   # dossier du track (kit/, release/)
METHODS = ["pixelseal", "trustmark", "stegastamp"]

def load_train(method=None):
    """Liste de dicts (ordre du metadata) : image_name, method, message, full, idx, reserve, fold (idx % 5 ; 0 = reserve), gt (128x128 bool)."""
    meta = json.loads((HERE / "release/train/metadata.json").read_text())
    z = np.load(HERE / "release/train/masks_128.npz"); m128 = dict(zip(z["names"].tolist(), z["masks"]))
    out = []
    for i, m in enumerate(meta):
        if method and m["method"] != method: continue
        out.append(dict(image_name=m["image_name"], method=m["method"], message=m["message"], full=bool(m.get("full_watermark")),
                        idx=i, reserve=(i % 5 == 0), fold=i % 5, gt=m128[m["image_name"]].astype(bool)))
    return out

def load_test(method=None):
    meta = json.loads((HERE / "release/test/metadata.json").read_text())
    return [dict(image_name=m["image_name"], method=m["method"], message=m["message"], idx=i) for i, m in enumerate(meta) if not method or m["method"] == method]

def load_img(split, name):
    from PIL import Image
    return np.asarray(Image.open(HERE / "release" / split / "images" / f"{name}.png").convert("RGB"))

def miou(pred, gt):
    """mIoU officiel par image, vectorise. pred, gt : N x 128 x 128 (bool). Renvoie un vecteur N."""
    p = np.asarray(pred).astype(bool).reshape(len(pred), -1); g = np.asarray(gt).astype(bool).reshape(len(gt), -1)
    i1 = (p & g).sum(1); u1 = (p | g).sum(1); i0 = (~p & ~g).sum(1); u0 = (~p | ~g).sum(1)
    return 0.5 * (np.where(u1 == 0, 1.0, i1 / np.maximum(u1, 1)) + np.where(u0 == 0, 1.0, i0 / np.maximum(u0, 1)))

def report(sc, rows):
    """sc : scores alignes sur rows (sortie de load_train). tous / reserve / hors reserve / pleines / partielles."""
    sc = np.asarray(sc); res = np.array([r["reserve"] for r in rows]); full = np.array([r["full"] for r in rows])
    f = lambda m: round(float(sc[m].mean()), 4) if m.any() else None
    return dict(tous=f(np.ones_like(res)), reserve=f(res), hors_reserve=f(~res), pleines=f(full), partielles=f(~full),
                res_pleines=f(res & full), res_partielles=f(res & ~full), n=len(sc), n_res=int(res.sum()))
