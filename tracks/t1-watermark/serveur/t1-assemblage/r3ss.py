"""Calage au pixel du cadre des rectangles par re-tamponnage (methode de t1-residus), ici pour StegaStamp et PixelSeal.
R3(boite) = energie du residu que l'encodeur ajoute au rognage avec le VRAI message / energie moyenne avec des messages au hasard.
Si le rognage est exactement le cadre d'origine du filigrane, l'encodeur ajoute beaucoup moins d'energie avec le vrai message : R3 a un creux.
Mode test : profil de R3 autour du vrai cadre (vraie boite elargie de 3,5 px) sur les rectangles du train.
Mode ecrit : part de mes boites du decodeur, descend R3 bord par bord, ecrit le cadre retreci de 3,5 px.
Usage : python r3ss.py stegastamp test [n] | python r3ss.py stegastamp write"""
import sys, json, time, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); from t1lib import *
sys.path.insert(0, str(HERE / "kit")); from wmloc.watermarks import load_watermark
m, mode = sys.argv[1], sys.argv[2]; NMAX = int(sys.argv[3]) if len(sys.argv) > 3 else 0; dev = "cuda"; torch.set_num_threads(4); K = 2; rng = np.random.default_rng(0)
wm = load_watermark(m, models_dir=HERE / "kit" / "models", device=dev); NB = wm.nbits
@torch.no_grad()
def r3(x, boxes, msgs):
    """x : 1x3x512x512 [0,1] ; msgs : (1+K) x nbits (le vrai en premier). Renvoie R3 par boite."""
    out = []
    for i in range(0, len(boxes), 12):
        bb = boxes[i:i + 12]
        if m == "stegastamp":
            cr = torch.cat([F.interpolate(x[:, :, b[0]:b[2], b[1]:b[3]], size=(400, 400), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1) for b in bb])
            im = cr.permute(0, 2, 3, 1).cpu().numpy().repeat(len(msgs), 0); sec = np.tile(msgs, (len(bb), 1)).astype(np.float32)
            res = wm._run(im, ["residual"], sec)[0]; res = res - res.mean(axis=(1, 2), keepdims=True); en = (res ** 2).sum(axis=(1, 2, 3)).reshape(len(bb), len(msgs))
        else:                                                              # pixelseal : encodeur a 256
            cr = torch.cat([F.interpolate(x[:, :, b[0]:b[2], b[1]:b[3]], size=(256, 256), mode="bilinear", align_corners=False, antialias=True).clamp(0, 1) for b in bb])
            c = cr.repeat_interleave(len(msgs), 0); sec = torch.from_numpy(np.tile(msgs, (len(bb), 1))).to(dev)
            o = wm.model.embed(c, msgs=sec, is_video=False)["imgs_w"].clamp(0, 1) - c; en = o.float().pow(2).sum((1, 2, 3)).view(len(bb), len(msgs)).cpu().numpy()
        out.append(en[:, 0] / (en[:, 1:].mean(1) + 1e-12))
    return np.concatenate(out)
def clip(b): y0, x0, y1, x1 = [int(v) for v in b]; y0 = max(0, min(y0, 496)); x0 = max(0, min(x0, 496)); return (y0, x0, max(y0 + 16, min(y1, 512)), max(x0 + 16, min(x1, 512)))
def descend(x, b, msgs, steps=(4, 2, 1, 1)):
    b = clip(b); best = float(r3(x, [b], msgs)[0])
    for st in steps:
        for _ in range(12):
            c = []
            for k in range(4):
                for d in (-st, st): t = list(b); t[k] += d; c.append(clip(t))
            for d in (-st, st): c += [clip((b[0] + d, b[1], b[2] + d, b[3])), clip((b[0], b[1] + d, b[2], b[3] + d))]
            c = [t for t in set(c) if t != b]; v = r3(x, c, msgs); j = int(np.argmin(v))
            if v[j] < best - 1e-4: best = float(v[j]); b = c[j]
            else: break
    return b, best
def iou(b, g):
    iy = max(0, min(b[2], g[2]) - max(b[0], g[0])); ix = max(0, min(b[3], g[3]) - max(b[1], g[1])); i = iy * ix
    return i / ((b[2] - b[0]) * (b[3] - b[1]) + (g[2] - g[0]) * (g[3] - g[1]) - i)
def prep(split, row):
    x = torch.from_numpy(load_img(split, row["image_name"]).copy()).permute(2, 0, 1)[None].float().div(255).to(dev); mm = np.array([int(c) for c in row["message"]], np.float32)
    return x, np.concatenate([mm[None], rng.integers(0, 2, (K, NB)).astype(np.float32)])
THR = {"stegastamp": 0.85, "pixelseal": 0.85}[m]
def load(split):
    """Boites de depart, au meme format pour les deux filigranes : name, box (y0, x0, y1, x1), z, ok (boite assez sure pour etre calee), + gt_box / rect sur le train."""
    if m == "stegastamp":
        R = json.load(open(ME / "boxnet" / f"boxes_stegastamp_{split}_net.json"))
        for r in R: r["ok"] = r.get("box") is not None and r.get("z", 0) >= 5
        return [r for r in R if r.get("box") is not None]
    sh = {r["name"]: r for r in json.loads((ME / "shapes_oracle.json").read_text())} if split == "train" else {}; full = {r["image_name"]: r["full"] for r in load_train(m)} if split == "train" else {}
    out = []
    for r in json.load(open(SESS / "t1-pixelseal" / f"box_{split}.json")):                 # format de t1-pixelseal : box = (y, x, h, w), bits, glob
        if "box" not in r: continue
        y, x_, h, w = [int(v) for v in r["box"]]; q = dict(name=r["name"], box=[y, x_, y + h, x_ + w], z=(float(r["bits"]) - 0.5) * 32, ok=bool(r["bits"] >= 0.85 and r["bits"] - r["glob"] >= 0.35))
        if split == "train" and not full.get(r["name"], True):
            g = sh[r["name"]]["bbox"]; q["gt_box"] = [g[0], g[1], g[0] + g[2], g[1] + g[3]]; q["rect"] = bool(sh[r["name"]]["rect"] > 0.98)
        out.append(q)
    return out
if mode == "test":
    rows = {r["image_name"]: r for r in load_train(m)}; R = [r for r in load("train") if r.get("rect") and r["ok"]][:NMAX or None]; prof = []; I0, I1, RR, D = [], [], [], []; t0 = time.time()
    for r in R:
        x, msgs = prep("train", rows[r["name"]]); g = r["gt_box"]; fr = clip((g[0] - 4, g[1] - 4, g[2] + 3, g[3] + 3))
        sh_ = [clip((fr[0] + d, fr[1], fr[2] + d, fr[3])) for d in (-6, -4, -2, -1, 0, 1, 2, 4, 6)] + [clip(tuple(g))]; prof.append(r3(x, sh_, msgs))
        b, v = descend(x, r["box"], msgs); mk = clip((b[0] + 4, b[1] + 4, b[2] - 3, b[3] - 3)); I0.append(iou(r["box"], g)); I1.append(iou(mk, g)); RR.append(v); D.append([b[k] - g[k] for k in range(4)])
        print(r["name"], "R3 au vrai cadre %.2f | apres descente %.2f | recouvrement %.2f -> %.2f | cadre trouve - vraie boite %s | %d s" % (prof[-1][4], v, I0[-1], I1[-1], D[-1], time.time() - t0), flush=True)
    P_ = np.array(prof); print("profil moyen de R3 selon le decalage vertical du cadre (-6,-4,-2,-1,0,+1,+2,+4,+6 px) :", P_[:, :9].mean(0).round(2), "| a la vraie boite (sans marge) :", P_[:, 9].mean().round(2))
    I0, I1, RR, D = map(np.array, (I0, I1, RR, D)); ok = RR < THR
    print(f"{m} rectangles n={len(R)} : recouvrement boite de depart {I0.mean():.3f} -> cadre cale retreci {I1.mean():.3f} | R3 < {THR} dans {ok.mean():.0%} des cas : {I1[ok].mean() if ok.any() else 0:.3f} ; regle finale (cale si R3 < {THR}, sinon boite de depart) : {np.where(ok, I1, I0).mean():.3f} | ecart median cadre - vraie boite {np.median(D[ok], 0) if ok.any() else None}")
else:
    for split in ("train", "test"):
        rows = {r["image_name"]: r for r in (load_train(m) if split == "train" else load_test(m))}; out = []; t0 = time.time(); nf = 0
        for r in load(split):
            q = dict(name=r["name"], z=r["z"], box=r["box"], box0=r["box"])
            if r["ok"]:
                x, msgs = prep(split, rows[r["name"]]); b, v = descend(x, r["box"], msgs); q["r3"] = v; q["frame"] = list(b)
                if v < THR: q["box"] = list(clip((b[0] + 4, b[1] + 4, b[2] - 3, b[3] - 3))); nf += 1
            out.append(q)
        d = ME / ("boxnet2" if m == "stegastamp" else "boxps2"); d.mkdir(exist_ok=True); json.dump(out, open(d / f"boxes_{m}_{split}_cadre.json", "w")); print("ecrit", split, len(out), "boites dont", nf, "cadres cales", round(time.time() - t0), "s", flush=True)
