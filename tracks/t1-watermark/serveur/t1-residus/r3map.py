"""Carte locale de « re-tamponnage » (R3) pour les images dont le filigrane est au cadre d'une TOILE un peu plus grande que l'image
(pleines et partielles « objet »). TrustMark, par lots.
1) on cherche les 4 marges de la toile (haut, gauche, bas, droite) qui rendent la carte la plus contrastée (moyenne des 10 % de cases les plus basses) ;
2) carte finale = énergie de ce que le tampon ajouterait avec le VRAI message / avec K messages au hasard, lissée, ramenée en 128x128.
   Bas (≈0,3) = filigrane présent ; ≈1 = absent. Sortie : maps_trustmark_<split>_r3map.npz (names, ratio, frame, obj) + mesures sur le train."""
import argparse, json, time, itertools
import numpy as np, torch, torch.nn.functional as F
from torchvision.ops import roi_align
from rlib import *
ap = argparse.ArgumentParser(); ap.add_argument("--method", default="trustmark"); ap.add_argument("--split", default="train"); ap.add_argument("--pop", default="objets")
ap.add_argument("--n", type=int, default=0); ap.add_argument("--K", type=int, default=6); ap.add_argument("--bs", type=int, default=64); ap.add_argument("--sig", type=float, default=3.0)
ap.add_argument("--boxes", default=""); ap.add_argument("--rad", type=int, default=0); ap.add_argument("--oracle", action="store_true"); ap.add_argument("--win", type=float, default=0.0); ap.add_argument("--tag", default=""); a = ap.parse_args()
dev = "cuda"; wm = load(a.method, dev); rng = np.random.default_rng(0); RAD = a.rad or (32 if a.pop == "pleines" else 160); PAD = RAD + 48; assert a.method == "trustmark"

@torch.no_grad()
def ratio_maps(xp, frames, msgs):
    """frames : liste (y0,x0,y1,x1) en coordonnées image (peuvent sortir de [0,512]). Retour : cartes de rapport 128x128 sur l'image, N x 128 x 128."""
    out = []; M = torch.from_numpy(msgs).to(dev); k1 = len(msgs)
    for i in range(0, len(frames), a.bs):
        fr = frames[i:i + a.bs]; b = torch.tensor([(0, t[1] + PAD, t[0] + PAD, t[3] + PAD, t[2] + PAD) for t in fr], dtype=torch.float32, device=dev)
        cover = roi_align(xp, b, output_size=(256, 256), spatial_scale=1.0, sampling_ratio=2, aligned=True).clamp(0, 1) * 2 - 1
        cov = cover.repeat_interleave(k1, 0); sec = M.repeat(len(fr), 1)
        with torch.autocast("cuda", dtype=torch.float16):
            stego, _ = wm.tm.encoder(cov, sec)
        res = stego.float().clamp(-1, 1) - cov; res = res - res.mean((2, 3), keepdim=True)
        en = gauss(res.pow(2).sum(1, keepdim=True), a.sig).view(len(fr), k1, 256, 256); den = en[:, 1:].mean(1, keepdim=True); eps = 0.05 * den.mean((2, 3), keepdim=True); R = (en[:, :1] + eps) / (den + eps)      # n x 1 x 256 x 256, repère de la toile
        # ré-échantillonnage vers l'image : la case (u,v) de l'image 128x128 a pour centre le pixel (4u+2, 4v+2)
        c = torch.arange(128, device=dev).float() * 4 + 2; grid = []
        for t in fr:
            gy = (c - t[0]) / (t[2] - t[0]) * 2 - 1; gx = (c - t[1]) / (t[3] - t[1]) * 2 - 1; grid.append(torch.stack(torch.meshgrid(gx, gy, indexing="xy"), -1))
        out.append(F.grid_sample(R - 1, torch.stack(grid), mode="bilinear", padding_mode="zeros", align_corners=False)[:, 0] + 1)      # hors toile : 1,0 (aucun indice)
    return torch.cat(out)

GT = None
def objective(maps):
    if a.oracle: w = torch.from_numpy(GT).to(dev).float()[None]; return (maps * w).sum((1, 2)) / w.sum()          # plafond : moyenne dans le vrai masque
    if a.win > 0: maps = gauss(maps[:, None], a.win)[:, 0]                                                         # lissage large (en cases de 4 px) avant de prendre le creux
    return maps.flatten(1).sort(1).values[:, :328].mean(1)                                                         # moyenne des 2 % de cases les plus basses

def find_frame(xp, msgs3, rad):
    """Toile = carré de côté S décalé de (dy,dx). Grille de décalages (pas 8) pour S = 512 et 536, puis descente fine sur les 4 bords."""
    grid = []
    for S in (512, 536):
        o = (S - 512) // 2
        for dy in range(-rad, rad + 1, 8):
            for dx in range(-rad, rad + 1, 8): grid.append((dy - o, dx - o, dy - o + S, dx - o + S))
    v = objective(ratio_maps(xp, grid, msgs3)).cpu().numpy(); j = int(np.argmin(v)); b = grid[j]; best = float(v[j]); n = len(grid)
    for st in (6, 3, 1):
        while True:
            c = [(b[0] + d, b[1], b[2] + d, b[3]) for d in (-st, st)] + [(b[0], b[1] + d, b[2], b[3] + d) for d in (-st, st)] + [(b[0] - d, b[1] - d, b[2] + d, b[3] + d) for d in (-st, st)]
            c += [tuple(b[k] + (d if k == q else 0) for k in range(4)) for q in range(4) for d in (-st, st)]
            c = [t for t in c if t[2] - t[0] > 300 and t[3] - t[1] > 300]
            v2 = objective(ratio_maps(xp, c, msgs3)).cpu().numpy(); n += len(c); j2 = int(np.argmin(v2))
            if v2[j2] < best - 1e-4: best = float(v2[j2]); b = c[j2]
            else: break
    return b, best, float(np.median(v)), n

M_ = [e for e in meta(a.split) if e["method"] == a.method]
B = {r["name"]: r for r in json.load(open(OUT / a.boxes))} if a.boxes else {}
sel = []
for e in M_:
    if a.split == "train" and a.pop != "tout":
        mk = read_mask(e["image_name"]); full = bool(e.get("full_watermark")); ys, xs = np.where(mk); rect = (not full) and mk[ys.min():ys.max() + 1, xs.min():xs.max() + 1].mean() > 0.98
        kind = "pleines" if full else ("rectangles" if rect else "objets")
        if kind != a.pop: continue
    elif B:                                                # test (ou « tout ») : on saute les rectangles déjà trouvés par le décodeur
        r = B.get(e["image_name"], {})
        if r.get("z", 0) > 6.5: continue
    sel.append(e)
if a.n: sel = sel[:a.n]
print("R3MAP", a.method, a.split, a.pop, "n=%d" % len(sel), flush=True); names, maps, frames, objs, G, gidx, full_ = [], [], [], [], [], [], []; t0 = time.time()
for i, e in enumerate(sel):
    x = to_t(read_img(a.split, e["image_name"]), dev); xp = F.pad(x, (PAD,) * 4, mode="reflect"); m = bits_of(e)
    msgs = np.concatenate([m[None], rng.integers(0, 2, (a.K, wm.nbits)).astype(np.float32)])
    if a.split == "train": GT = downsample_mask(read_mask(e["image_name"]))
    f, o, med, n = find_frame(xp, msgs[:5], RAD); R = ratio_maps(xp, [f], msgs)[0].cpu().numpy()
    names.append(e["image_name"]); maps.append(R.astype(np.float16)); frames.append(f); objs.append(o); gidx.append(e["gidx"])
    if a.split == "train": G.append(downsample_mask(read_mask(e["image_name"]))); full_.append(bool(e.get("full_watermark")))
    if i % 5 == 0: print("  %d/%d, %d s ; toile %s (marges %d,%d,%d,%d) contraste %.2f (médiane de la grille %.2f)%s" % (i + 1, len(sel), time.time() - t0, f, -f[0], -f[1], f[2] - 512, f[3] - 512, o, med,
        (" | dedans %.2f dehors %.2f" % (R[G[-1] > 0].mean(), R[G[-1] == 0].mean())) if a.split == "train" else ""), flush=True)
maps = np.stack(maps); np.savez_compressed(OUT / f"maps_{a.method}_{a.split}_r3map{a.tag}.npz", names=np.array(names), ratio=maps, frame=np.array(frames), obj=np.array(objs), gidx=np.array(gidx))
print("FIN R3MAP n=%d sec=%d" % (len(sel), time.time() - t0))
if G:
    G = np.stack(G); gidx = np.array(gidx); res_ = gidx % 5 == 0; S = -maps.astype(np.float32)                      # score haut = filigrané
    fr = np.array(frames); print("marges trouvées (haut, gauche, bas, droite) : médiane %s ; q10 %s ; q90 %s" % (np.median(np.abs(fr - [0, 0, 512, 512]), 0), np.quantile(np.abs(fr - [0, 0, 512, 512]), .1, 0), np.quantile(np.abs(fr - [0, 0, 512, 512]), .9, 0)))
    print("rapport moyen dedans %.3f, dehors %.3f | AUC par pixel (toutes images) %.3f | AUC moyenne par image %.3f" % (maps[G > 0].astype(np.float32).mean(), maps[G == 0].astype(np.float32).mean(), auc(S[G > 0], S[G == 0]),
          np.nanmean([auc(S[j][G[j] > 0], S[j][G[j] == 0]) for j in range(len(S))])))
    th, sfit = fit_threshold(S, G, ~res_, n=41); sc = score(S, G, th, np.arange(len(S)))
    print("mIoU officiel, seuil global rapport < %.3f réglé hors réserve : hors réserve %.3f (n=%d) | RÉSERVE %.3f (n=%d) | si tout-0 : %.3f ; si tout-1 : %.3f" % (-th, sc[~res_].mean(), (~res_).sum(), sc[res_].mean() if res_.any() else float("nan"), res_.sum(),
          np.mean([miou_image(np.zeros((128, 128)), g) for g in G]), np.mean([miou_image(np.ones((128, 128)), g) for g in G])))
