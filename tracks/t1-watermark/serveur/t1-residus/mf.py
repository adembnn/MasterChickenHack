"""Étape 2 : filtre adapté propre sur les VRAIES images, cadre = image entière.
Pour chaque débruiteur F : h = F(x) (résidu estimé), e = ce que le gabarit devient après F.
Cartes (128x128) : gain g = lissé(w.h.e)/lissé(w.e.e) ; t = lissé(w.h.e)/racine(lissé((w.h.e)^2)) ; zK = écart aux K messages au hasard.
Variantes de gabarit : plain (message vrai), diff (vrai moins moyenne des faux), it (gabarit recalculé après avoir retiré le premier gabarit).
Mesures : AUC par pixel (pleines / partielles), mIoU officiel sur la réserve avec un seuil global réglé HORS réserve."""
import argparse, json, time
import numpy as np, torch, torch.nn.functional as F
from rlib import *

ap = argparse.ArgumentParser(); ap.add_argument("--method", required=True); ap.add_argument("--split", default="train")
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--K", type=int, default=6)
ap.add_argument("--dens", default="g1,g2,g4,g8,dog1_4,dog2_8,med3,med5,ud,tmrm"); ap.add_argument("--save", default="")
a = ap.parse_args()
dev = "cuda"; wm = load(a.method, dev); rm = load_remover(dev) if "tmrm" in a.dens else None; rng = np.random.default_rng(0)
M = [e for e in meta(a.split) if e["method"] == a.method]
if a.limit: M = M[:a.limit]
DENS = a.dens.split(","); SIG = (1.5, 3.0, 6.0); keep = set(a.save.split(",")) if a.save else None
maps = {}; G = []; full = []; names = []; gidx = []; t0_ = time.time()
def put(k, v):
    if keep is None or k in keep: maps.setdefault(k, []).append(v.cpu().numpy().astype(np.float16))
for n_, e in enumerate(M):
    img = read_img(a.split, e["image_name"]); x = to_t(img, dev); m = bits_of(e)
    T = tmpl(wm, img, np.stack([m] + [rng.integers(0, 2, wm.nbits) for _ in range(a.K)]))       # 1 + K gabarits
    Tit = tmpl(wm, (x - T[:1]).clamp(0, 1), m[None])                                              # gabarit itéré
    for d in DENS:
        h = hp(d, x, rm); E = expected(d, x, torch.cat([T, Tit]), rm)
        es = {"plain": E[:1], "diff": E[:1] - E[1:1 + a.K].mean(0, keepdim=True), "it": E[-1:]}
        for wname in ("u", "w"):
            w = 1.0 if wname == "u" else 1.0 / (gauss(h.pow(2).sum(1, keepdim=True), 4.0) + (0.5 / 255) ** 2)
            for tv, ev in es.items():
                p = (w * h * ev).sum(1, keepdim=True); q = (w * ev * ev).sum(1, keepdim=True)
                A = F.avg_pool2d(p, 4); B = F.avg_pool2d(q, 4); A2 = F.avg_pool2d(p * p, 4)
                for s in SIG:
                    S = gauss(A, s); Es = gauss(B, s); V = gauss(A2, s)
                    put(f"{d}|{tv}|{wname}|{int(s*4)}|g", (S / (Es + 0.02 * Es.mean() + 1e-12))[0, 0])
                    put(f"{d}|{tv}|{wname}|{int(s*4)}|t", (S / (V + 1e-4 * V.mean() + 1e-30).sqrt())[0, 0])
            pk = (w * h * E[1:1 + a.K]).sum(1, keepdim=True); Ak = F.avg_pool2d(pk, 4); A = F.avg_pool2d((w * h * E[:1]).sum(1, keepdim=True), 4)
            for s in SIG:
                Sk = gauss(Ak, s); S = gauss(A, s)
                put(f"{d}|plain|{wname}|{int(s*4)}|zK", ((S - Sk.mean(0, keepdim=True)) / (Sk.std(0, keepdim=True) + 1e-12))[0, 0])
    names.append(e["image_name"]); gidx.append(e["gidx"])
    if a.split == "train":
        G.append(downsample_mask(read_mask(e["image_name"]))); full.append(bool(e.get("full_watermark", False)))
    if n_ % 20 == 0: print(a.method, n_, round(time.time() - t0_), "s", flush=True)
maps = {k: np.stack(v) for k, v in maps.items()}
if keep is not None:
    np.savez_compressed(OUT / f"maps_{a.method}_{a.split}_mf.npz", names=np.array(names), **{k.replace("|", "_"): v for k, v in maps.items()})
if G:
    G = np.stack(G); full = np.array(full); gidx = np.array(gidx); res_ = gidx % 5 == 0; rows = []
    for k, Mk in maps.items():
        Mk = np.nan_to_num(Mk.astype(np.float32), posinf=1e4, neginf=-1e4)
        auc_f = auc(Mk[full][G[full] > 0], Mk[full][G[full] == 0]); auc_p = auc(Mk[~full][G[~full] > 0], Mk[~full][G[~full] == 0])
        th, s_fit = fit_threshold(Mk, G, ~res_, n=25); sc = score(Mk, G, th, np.arange(len(G)))
        thf, _ = fit_threshold(Mk, G, ~res_ & full, n=25); scf = score(Mk, G, thf, np.where(res_ & full)[0])
        rows.append((k, auc_f, auc_p, s_fit, float(sc[res_].mean()), float(sc[res_ & full].mean()), float(sc[res_ & ~full].mean()), float(scf.mean())))
    cols = ("carte", "auc_pleines", "auc_partielles", "miou_hors_reserve", "miou_reserve", "reserve_pleines", "reserve_partielles", "reserve_pleines_seuil_pleines")
    json.dump([dict(zip(cols, r)) for r in rows], open(OUT / f"mf_{a.method}_{a.split}.json", "w"), indent=0)
    print("RESULT", a.method, "n=%d reserve=%d sec=%d" % (len(G), res_.sum(), time.time() - t0_))
    print("carte | AUC pleines | AUC partielles | mIoU hors-réserve (seuil réglé là) | mIoU RÉSERVE | rés. pleines | rés. partielles | rés. pleines avec seuil propre aux pleines")
    fmt = "  %-26s %.3f %.3f %.3f %.3f %.3f %.3f %.3f"
    for r in sorted(rows, key=lambda r: -r[1])[:14]: print(fmt % r)
    print("-- meilleur par débruiteur (AUC pleines)")
    for d in DENS:
        print(fmt % max((r for r in rows if r[0].startswith(d + "|")), key=lambda r: r[1]))
    print("-- meilleur par variante de gabarit / poids / statistique")
    for key in ("|plain|", "|diff|", "|it|", "|u|", "|w|", "|g", "|t", "|zK"):
        r = max((r for r in rows if (key in r[0] if key[-1] == "|" else r[0].endswith(key))), key=lambda r: r[1]); print(("  %-8s" % key) + fmt % r)
