"""Tache 1, baseline sans entrainement : localiser le watermark en decodant.

Principe. Le message est donne pour chaque image de test. Un extracteur officiel rend des
logits par bit. Sur une fenetre qui contient du watermark, les bits decodes collent au message.
Sur une fenetre propre, ils sont a pile ou face, soit 50 pour cent d'accord. Donc le taux
d'accord est directement la carte de localisation, sans apprentissage ni masque de train.

Deux sources de signal, qu'on additionne :
  1. fenetre glissante, decodage global par fenetre (precis, couteux)
  2. logits par position de l'extracteur (grossier, quasi gratuit), quand ils existent

Usage :
  python t1_decode_localize.py --data release --out submission.pt
  python t1_decode_localize.py --data release --calibrate      # balaye les seuils sur le train
  python t1_decode_localize.py --data release --limit 20        # essai rapide

A lancer depuis le dossier de la tache, wmloc importable, modeles deja telecharges.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from wmloc.data import LocalisationDataset
from wmloc.utils import GRID, score

METHOD_FOR_LOAD = {"pixelseal": "pixelseal", "trustmark": "trustmark", "stegastamp": "stegastamp"}


def to_tensor(img_uint8):
    return torch.from_numpy(np.asarray(img_uint8, dtype=np.uint8)).permute(2, 0, 1)[None].float() / 255.0


def agreement(logits, msg):
    """logits B x nbits, msg nbits. Rend B, taux d'accord dans [0, 1], 0.5 = hasard."""
    m = torch.from_numpy(np.asarray(msg)).float() * 2 - 1        # -1 / +1
    m = m[None, :]
    return (0.5 + 0.5 * torch.sign(logits) * m).mean(dim=1)


@torch.no_grad()
def window_map(wm, img_u8, msg, win=256, stride=64, mode="canvas"):
    """Carte d'accord en pleine resolution par fenetre glissante, accumulee sur les recouvrements.

    mode="canvas" : la fenetre est replacee a sa position exacte dans un canevas de la taille de
    l'image, le reste en gris neutre. Les trois decodeurs redimensionnent l'entree a une resolution
    fixe, donc c'est le seul mode qui garde l'echelle du motif. Hypotheses d'Adem, 21/11.
    mode="crop" : la fenetre est passee telle quelle au decodeur, donc agrandie. Plus sensible mais
    l'echelle du motif change, ce qui peut faire echouer le decodage.
    """
    H, W = img_u8.shape[:2]
    total = np.zeros((H, W), np.float32)
    count = np.zeros((H, W), np.float32)
    ys = list(range(0, H - win + 1, stride))
    xs = list(range(0, W - win + 1, stride))
    if ys[-1] != H - win:
        ys.append(H - win)
    if xs[-1] != W - win:
        xs.append(W - win)
    H_img, W_img = img_u8.shape[:2]
    crops, boxes = [], []
    for y in ys:
        for x in xs:
            crop = img_u8[y:y + win, x:x + win]
            if mode == "canvas":
                canvas = np.full((H_img, W_img, 3), 128, np.uint8)
                canvas[y:y + crop.shape[0], x:x + crop.shape[1]] = crop
                crops.append(canvas)
            else:
                crops.append(crop)
            boxes.append((y, x))
    batch = torch.cat([to_tensor(c) for c in crops], 0)
    scores = agreement(wm.logits(batch), msg).cpu().numpy()
    for (y, x), s in zip(boxes, scores):
        total[y:y + win, x:x + win] += s
        count[y:y + win, x:x + win] += 1
    return total / np.maximum(count, 1)


@torch.no_grad()
def position_map(wm, img_u8, msg):
    """Logits par position de l'extracteur, remontes a la taille de l'image. None si indisponible."""
    try:
        feats = wm.features(to_tensor(img_u8))
    except Exception:
        return None
    per = feats.get("per_position_logits")
    if per is None:
        return None
    m = torch.from_numpy(np.asarray(msg)).float().mul(2).sub(1)
    m = m.to(per.device)[:, None, None]
    bits = per[0][: m.shape[0]]
    s = 0.5 + 0.5 * torch.sign(bits) * m
    s = s.mean(dim=0)[None, None]
    s = F.interpolate(s, size=img_u8.shape[:2], mode="bilinear", align_corners=False)
    return s[0, 0].cpu().numpy()


def predict_image(wm, item, win=256, stride=64, w_win=1.0, w_pos=0.5, use_window=True, mode="canvas"):
    img = item["image"]
    msg = item["message"]
    acc = np.zeros(img.shape[:2], np.float32)
    norm = 0.0
    if use_window:
        acc += w_win * window_map(wm, img, msg, win, stride, mode)
        norm += w_win
    pm = position_map(wm, img, msg)
    if pm is not None and w_pos > 0:
        acc += w_pos * pm
        norm += w_pos
    acc /= max(norm, 1e-6)
    return acc


def to_grid(acc, thr):
    """Carte continue -> grille 128 x 128. Regle officielle : cellule a 1 si la moyenne depasse le seuil."""
    t = torch.from_numpy(acc)[None, None].float()
    g = F.adaptive_avg_pool2d(t, GRID)[0, 0].numpy()
    return (g >= thr).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="release")
    ap.add_argument("--out", default="submission.pt")
    ap.add_argument("--models", default="models")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--win", type=int, default=256)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--w-pos", type=float, default=0.5)
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--decode-mode", default="canvas", choices=["canvas", "crop"],
                    help="canvas garde l'echelle du motif, crop agrandit la fenetre")
    ap.add_argument("--thr", type=float, default=None, help="seuil global, sinon celui du fichier de calibration")
    ap.add_argument("--calibrate", action="store_true", help="balaye le seuil sur le train par methode")
    ap.add_argument("--calib-out", default="calib_t1.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-maps", default=None, help="dossier ou sauver les cartes continues .npy")
    a = ap.parse_args()

    from wmloc.watermarks import load_watermark
    methods = ["pixelseal", "trustmark", "stegastamp"]
    wms = {m: load_watermark(METHOD_FOR_LOAD[m], a.models, a.device) for m in methods}

    def maps(split):
        ds = LocalisationDataset(a.data, split)
        if a.limit:
            ds.meta = ds.meta[: a.limit]
        out = []
        for i, it in enumerate(ds):
            acc = predict_image(wms[it["method"]], it, a.win, a.stride, 1.0, a.w_pos,
                                not a.no_window, a.decode_mode)
            out.append((it, acc))
            if i % 20 == 0:
                print(f"{split} {i}/{len(ds)}", flush=True)
        return out

    if a.calibrate:
        tr = maps("train")
        if a.save_maps:
            Path(a.save_maps).mkdir(parents=True, exist_ok=True)
            for it, acc in tr:
                np.save(Path(a.save_maps) / f"{it['image_name']}.npy", acc.astype(np.float16))
        calib = {}
        for m in methods:
            sub = [(it, acc) for it, acc in tr if it["method"] == m]
            best = (-1, None)
            for thr in np.arange(0.30, 0.76, 0.02):
                s = score([to_grid(acc, thr) for _, acc in sub], [it["mask128"] for it, _ in sub])
                if s > best[0]:
                    best = (s, float(thr))
            calib[m] = {"thr": best[1], "train_miou": round(best[0], 4), "n": len(sub)}
            print(m, calib[m])
        overall = score([to_grid(acc, calib[it["method"]]["thr"]) for it, acc in tr],
                        [it["mask128"] for it, _ in tr])
        calib["decode_mode"] = a.decode_mode
        calib["overall_train_miou"] = round(overall, 4)
        Path(a.calib_out).write_text(json.dumps(calib, indent=2))
        print("train mIoU avec seuils par methode :", round(overall, 4))
        return

    thr = json.loads(Path(a.calib_out).read_text()) if Path(a.calib_out).exists() else {}
    te = maps("test")
    names = [it["image_name"] for it, _ in te]
    masks = []
    for it, acc in te:
        t = a.thr if a.thr is not None else thr.get(it["method"], {}).get("thr", 0.5)
        masks.append(to_grid(acc, t))
    torch.save({"image_names": names, "masks": torch.from_numpy(np.stack(masks)).bool()}, a.out)
    print("ecrit", a.out, len(names), "masques")


if __name__ == "__main__":
    main()
