"""T1 · post-traitement : reconnaitre la population (pleine ou partielle) puis regler le masque en consequence.

Idee reprise du tableau de l'equipe : le taux de bits justes de l'extracteur separe proprement les deux
populations (images entierement filigranees contre zones greffees). Une fois la population connue, on
peut appliquer la bonne regle geometrique :

- pleine  : la zone propre est un trou. Le masque doit etre le complement du plus grand trou connexe,
            donc on rebouche les regions propres enfermees dans le masque predit.
- partielle : la zone est un rectangle ou un objet compact. On garde la plus grande composante connexe
            et on jette les miettes.

On mesure tout sur la reserve (indice % 5 == 0), qui n'a jamais servi.

Usage : python post_process_v3.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "kit"))
from wmloc.data import LocalisationDataset
from wmloc.utils import miou_image
from wmloc.watermarks import load_watermark

DIRS = {"pixelseal": "seg_v3_pixelseal", "trustmark": "seg_v3_trustmark", "stegastamp": "seg_v3r_stegastamp"}


def fill_holes(m):
    """Rebouche les trous : garde le complement du plus grand trou connexe de la zone propre."""
    clean = ~m
    lab, n = ndimage.label(clean)
    if n == 0:
        return np.ones_like(m)
    sizes = ndimage.sum(clean, lab, range(1, n + 1))
    big = int(np.argmax(sizes)) + 1
    # le plus grand trou qui touche le bord est le fond, on le garde propre ; les autres sont rebouches
    edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    out = m.copy()
    for i in range(1, n + 1):
        if i == big or i in edge:
            continue
        out[lab == i] = True
    return out


def largest_component(m):
    lab, n = ndimage.label(m)
    if n <= 1:
        return m
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    big = int(np.argmax(sizes)) + 1
    keep = np.zeros_like(m)
    keep[lab == big] = True
    return keep


def main():
    train = LocalisationDataset(HERE / "release", "train")
    meta = train.meta
    gt = {m["image_name"]: train.m128[m["image_name"]] for m in meta}
    reserve = {m["image_name"] for i, m in enumerate(meta) if i % 5 == 0}
    is_full = {m["image_name"]: bool(m.get("full_watermark")) for m in meta}

    # 1) taux de bits justes par image, avec l'extracteur d'origine
    acc = {}
    for meth in DIRS:
        wm = load_watermark(meth, models_dir=HERE / "kit" / "models", device="cuda")
        idx = [i for i, m in enumerate(meta) if m["method"] == meth]
        for k in range(0, len(idx), 16):
            batch = [train[i] for i in idx[k:k + 16]]
            imgs = np.stack([b["image"] for b in batch])
            xt = torch.from_numpy(imgs).permute(0, 3, 1, 2).float().div(255.0)
            bits = np.atleast_2d(wm.decode(xt).cpu().numpy())
            for b, pred in zip(batch, bits):
                msg = np.asarray(b["message"]).astype(int)
                pred = (np.asarray(pred) > 0).astype(int).ravel()[: len(msg)]
                acc[b["image_name"]] = float((pred == msg).mean())
        del wm
        torch.cuda.empty_cache()

    # 2) seuil qui separe le mieux les deux populations sur le train
    report, TH = {}, {}
    for meth in DIRS:
        names = [m["image_name"] for m in meta if m["method"] == meth]
        fulls = [acc[n] for n in names if is_full[n]]
        parts = [acc[n] for n in names if not is_full[n]]
        best = (0.0, None, 0)
        for th in np.arange(0.5, 0.95, 0.01):
            ok = sum(a > th for a in fulls) + sum(a <= th for a in parts)
            if ok > best[2]:
                best = (round(float(th), 2), None, ok)
        report[meth] = {"seuil": best[0], "exactitude_train": round(best[2] / max(len(names), 1), 3),
                        "acc_moy_pleines": round(float(np.mean(fulls)), 3),
                        "acc_moy_partielles": round(float(np.mean(parts)), 3)}
        TH[meth] = best[0]

    # 3) appliquer la regle et noter sur la reserve
    res = {}
    for meth, d in DIRS.items():
        z = np.load(HERE / "adem" / d / f"pred_train_{meth}.npz")
        raw, ruled = [], []
        for n, l in zip(z["names"], z["logits"].astype(np.float32)):
            n = str(n)
            m = l > 0
            if n in reserve:
                raw.append(miou_image(m, gt[n]))
            full_pred = acc.get(n, 0.0) > TH[meth]
            m2 = fill_holes(m) if full_pred else largest_component(m)
            if n in reserve:
                ruled.append(miou_image(m2, gt[n]))
        res[meth] = {"reserve_brut": round(float(np.mean(raw)), 4), "reserve_regle": round(float(np.mean(ruled)), 4)}

    print(json.dumps({"seuils": report, "reserve": res}, indent=1))
    Path(HERE / "adem" / "rapport_postproc_v3.json").write_text(json.dumps({"seuils": report, "reserve": res}, indent=1))


if __name__ == "__main__":
    main()
