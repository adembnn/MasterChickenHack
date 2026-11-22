"""Images PLEINES TrustMark : trouver la toile (4 marges inégales : haut, gauche, bas, droite) par le critère « re-tamponnage » GLOBAL, puis carte locale.
Idée (mesure de t1-assemblage) : le cadre du filigrane est une toile plus grande que l'image, avec des marges différentes par côté (ex. 22, 18, 12, 16).
1) grille des 4 marges (pas 8, 0 à 40), critère = énergie(vrai message) / énergie(messages au hasard) sur la partie de la toile qui est dans l'image ;
2) descente au pixel ; 3) carte locale du même rapport (K messages au hasard, lissage), 128x128. Bas = filigrané, ~1 = trou.
Sorties : maps_trustmark_<split>_pleine<tag>.npz (names, ratio, frame, r3) ; sur le train : AUC des trous, mIoU."""
import argparse, json, time, itertools
import numpy as np, torch, torch.nn.functional as F
from torchvision.ops import roi_align
from rlib import *
ap = argparse.ArgumentParser(); ap.add_argument("--split", default="train"); ap.add_argument("--n", type=int, default=0); ap.add_argument("--K", type=int, default=6); ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--sig", type=float, default=3.0); ap.add_argument("--hi", type=int, default=40); ap.add_argument("--step", type=int, default=8); ap.add_argument("--tag", default="")
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1); ap.add_argument("--pad", default="reflect"); a = ap.parse_args()
dev = "cuda"; wm = load("trustmark", dev); rng = np.random.default_rng(0); PAD = a.hi + 24

@torch.no_grad()
def energies(xp, frames, msgs):
    """Retour : énergies lissées n x (1+K) x 256 x 256 dans le repère de chaque toile."""
    out = []; M = torch.from_numpy(msgs).to(dev); k1 = len(msgs)
    for i in range(0, len(frames), a.bs):
        fr = frames[i:i + a.bs]; b = torch.tensor([(0, t[1] + PAD, t[0] + PAD, t[3] + PAD, t[2] + PAD) for t in fr], dtype=torch.float32, device=dev)
        cover = roi_align(xp, b, output_size=(256, 256), spatial_scale=1.0, sampling_ratio=2, aligned=True).clamp(0, 1) * 2 - 1
        cov = cover.repeat_interleave(k1, 0); sec = M.repeat(len(fr), 1)
        with torch.autocast("cuda", dtype=torch.float16):
            stego, _ = wm.tm.encoder(cov, sec)
        res = stego.float().clamp(-1, 1) - cov; res = res - res.mean((2, 3), keepdim=True); out.append(res.pow(2).sum(1).view(len(fr), k1, 256, 256))
    return torch.cat(out)

def global_r3(xp, frames, msgs):
    """Rapport global, restreint à la partie de la toile couverte par l'image (les marges, de contenu inconnu, sont exclues)."""
    en = energies(xp, frames, msgs); v = []
    for e, t in zip(en, frames):
        h, w = t[2] - t[0], t[3] - t[1]; y0 = int(round(-t[0] / h * 256)); y1 = int(round((512 - t[0]) / h * 256)); x0 = int(round(-t[1] / w * 256)); x1 = int(round((512 - t[1]) / w * 256))
        s = e[:, max(0, y0):y1, max(0, x0):x1].sum((1, 2)); v.append(float(s[0] / s[1:].mean()))
    return np.array(v)

def find_frame(xp, msgs3):
    ms = list(range(0, a.hi + 1, a.step)); grid = [(-m[0], -m[1], 512 + m[2], 512 + m[3]) for m in itertools.product(ms, repeat=4)]
    v = global_r3(xp, grid, msgs3); j = int(np.argmin(v)); b = grid[j]; best = float(v[j]); n = len(grid)
    for st in (4, 2, 1):
        while True:
            c = [tuple(b[k] + (d if k == q else 0) for k in range(4)) for q in range(4) for d in (-st, st)] + [(b[0] + d, b[1], b[2] + d, b[3]) for d in (-st, st)] + [(b[0], b[1] + d, b[2], b[3] + d) for d in (-st, st)]
            c = [t for t in c if t[0] <= 0 and t[1] <= 0 and t[2] >= 512 and t[3] >= 512 and t[0] >= -PAD + 2 and t[1] >= -PAD + 2 and t[2] <= 510 + PAD and t[3] <= 510 + PAD]
            if not c: break
            v2 = global_r3(xp, c, msgs3); n += len(c); j2 = int(np.argmin(v2))
            if v2[j2] < best - 1e-4: best = float(v2[j2]); b = c[j2]
            else: break
    return b, best, float(np.median(v)), float(v[grid.index((0, 0, 512, 512))]), n

SIGS = (3.0, 6.0, 10.0, 16.0, 24.0)
def local_map(xp, f, msgs):
    """Cartes du rapport à 5 lissages (en px du repère 256 de la toile), ramenées sur l'image : 5 x 128 x 128."""
    raw = energies(xp, [f], msgs).view(-1, 1, 256, 256); out = []
    c = torch.arange(128, device=dev).float() * 4 + 2; gy = (c - f[0]) / (f[2] - f[0]) * 2 - 1; gx = (c - f[1]) / (f[3] - f[1]) * 2 - 1; grid = torch.stack(torch.meshgrid(gx, gy, indexing="xy"), -1)[None]
    for sg in SIGS:
        en = gauss(raw, sg).view(1, -1, 256, 256); den = en[:, 1:].mean(1, keepdim=True); eps = 0.05 * den.mean(); R = (en[:, :1] + eps) / (den + eps)
        out.append(F.grid_sample(R - 1, grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0, 0] + 1)
    return torch.stack(out)

M_ = [e for e in meta(a.split) if e["method"] == "trustmark"]
if a.split == "train": M_ = [e for e in M_ if e.get("full_watermark")]
M_ = M_[a.shard::a.nshards]
if a.nshards > 1: a.tag += "_s%dof%d" % (a.shard, a.nshards)
names, maps, frames, r3s, r0s, G, gidx = [], [], [], [], [], [], []; t0 = time.time(); print("PLEINE trustmark", a.split, "n=%d (grille %d marges par côté)" % (len(M_) if not a.n else min(a.n, len(M_)), len(range(0, a.hi + 1, a.step))), flush=True)
for i, e in enumerate(M_):
    x = to_t(read_img(a.split, e["image_name"]), dev); m = bits_of(e); msgs = np.concatenate([m[None], rng.integers(0, 2, (a.K, wm.nbits)).astype(np.float32)])
    if a.split == "test":                                   # test : seulement les images que le routeur classe « pleines »
        lg = wm.logits(x)[0].cpu().numpy()
        if ((lg > 0) == (m > 0)).mean() < 0.75: continue
    xp = F.pad(x, (PAD,) * 4, mode=a.pad); f, v, med, r0, n = find_frame(xp, msgs[:3]); R = local_map(xp, f, msgs).cpu().numpy()
    names.append(e["image_name"]); maps.append(R.astype(np.float16)); frames.append(f); r3s.append(v); r0s.append(r0); gidx.append(e["gidx"]); extra = ""
    if a.split == "train":
        g = downsample_mask(read_mask(e["image_name"])); G.append(g)
        if 10 < (g == 0).sum(): extra = " | part filigranée %.2f, dedans %.2f, trou %.2f, AUC par lissage %s" % (g.mean(), R[2][g > 0].mean(), R[2][g == 0].mean(), " ".join("%.2f" % auc(-R[k][g > 0], -R[k][g == 0]) for k in range(len(SIGS))))
    print("  %d : %d s | R3 image entière %.2f -> toile marges (%d,%d,%d,%d) R3 %.2f (médiane grille %.2f)%s" % (len(names), time.time() - t0, r0, -f[0], -f[1], f[2] - 512, f[3] - 512, v, med, extra), flush=True)
    if len(names) % 10 == 0 or (a.n and len(names) >= a.n):
        np.savez_compressed(OUT / f"maps_trustmark_{a.split}_pleine{a.tag}.npz", names=np.array(names), ratio=np.stack(maps), frame=np.array(frames), r3=np.array(r3s), r0=np.array(r0s), gidx=np.array(gidx))
    if a.n and len(names) >= a.n: break
np.savez_compressed(OUT / f"maps_trustmark_{a.split}_pleine{a.tag}.npz", names=np.array(names), ratio=np.stack(maps), frame=np.array(frames), r3=np.array(r3s), r0=np.array(r0s), gidx=np.array(gidx))
print("FIN PLEINE n=%d sec=%d" % (len(names), time.time() - t0))
if G:
    G = np.stack(G); maps = np.stack(maps).astype(np.float32); r3s = np.array(r3s); fr = np.array(frames); mg = np.abs(fr - np.array([0, 0, 512, 512]))
    for k, sg in enumerate(SIGS):
        Ak = np.array([auc(-maps[j][k][G[j] > 0], -maps[j][k][G[j] == 0]) if (G[j] == 0).sum() > 10 else np.nan for j in range(len(G))])
        print("lissage %4.0f : AUC des trous moyenne %.3f, médiane %.3f, > 0,8 : %d/%d" % (sg, np.nanmean(Ak), np.nanmedian(Ak), (Ak > 0.8).sum(), len(Ak)))
    A = np.array([auc(-maps[j][3][G[j] > 0], -maps[j][3][G[j] == 0]) if (G[j] == 0).sum() > 10 else np.nan for j in range(len(G))])
    print("toile trouvée (R3 < 0,6) : %d/%d ; R3 médian %.2f ; marges médianes %s, somme haut+bas médiane %.0f, gauche+droite %.0f" % ((r3s < 0.6).sum(), len(r3s), np.median(r3s), np.median(mg, 0), np.median(mg[:, 0] + mg[:, 2]), np.median(mg[:, 1] + mg[:, 3])))
    for nm, sel in (("toutes", np.ones(len(A), bool)), ("toile trouvée (R3 < 0,6)", r3s < 0.6), ("toile non trouvée", r3s >= 0.6)):
        if sel.sum(): print("  %-26s n=%2d | AUC des trous : moyenne %.3f, médiane %.3f, > 0,8 : %d" % (nm, sel.sum(), np.nanmean(A[sel]), np.nanmedian(A[sel]), (A[sel] > 0.8).sum()))
