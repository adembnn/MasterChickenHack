"""Pleines TrustMark, au BON cadre (toile trouvée par pleine.py) : filtre adapté refait proprement avec l'effaceur de TrustMark.
  propre^ = effaceur(toile)            (réseau de retrait de TrustMark, qui travaille au cadre 256 de la toile)
  r^      = toile - propre^            (filigrane estimé présent : ~0 dans un trou)
  t0      = tampon(propre^, m) - propre^   (motif attendu pour NOTRE message sur l'image propre)
Cartes (5 lissages, 128x128) : gain g = lissé(r^.t0) / lissé(t0.t0)  (≈1 filigrané, ≈0 trou) ; zg = écart de g aux gains de K faux messages ;
e = lissé(|r^|²) / lissé(|t0|²) (énergie retirée, sans message). Sortie : maps_trustmark_<split>_rm.npz + AUC des trous."""
import argparse, time
import numpy as np, torch, torch.nn.functional as F
from torchvision.ops import roi_align
from rlib import *
ap = argparse.ArgumentParser(); ap.add_argument("--split", default="train"); ap.add_argument("--src", default="maps_trustmark_train_pleine_s0of2.npz,maps_trustmark_train_pleine_s1of2.npz")
ap.add_argument("--K", type=int, default=12); ap.add_argument("--tag", default=""); ap.add_argument("--frame", default="found"); a = ap.parse_args()
dev = "cuda"; wm = load("trustmark", dev); rng = np.random.default_rng(7); PAD = 96; SIGS = (3.0, 6.0, 10.0, 16.0, 24.0)
names, frames, r3, gidx = [], [], [], []
for f in a.src.split(","):
    m = np.load(OUT / f, allow_pickle=True); names += m["names"].tolist(); frames += [tuple(int(v) for v in t) for t in m["frame"]]; r3 += m["r3"].tolist(); gidx += m["gidx"].tolist()
if a.frame == "image": frames = [(0, 0, 512, 512)] * len(frames)                      # témoin : cadre image entière (mauvais cadre)
mb = {e["image_name"]: e for e in meta(a.split)}; OUTM = {"g": [], "zg": [], "e": []}; G = []; t0_ = time.time()
with torch.no_grad():
    for nm, f in zip(names, frames):
        x = to_t(read_img(a.split, nm), dev); xp = F.pad(x, (PAD,) * 4, mode="reflect"); m = bits_of(mb[nm]); msgs = np.concatenate([m[None], rng.integers(0, 2, (a.K, wm.nbits)).astype(np.float32)])
        b = torch.tensor([(0, f[1] + PAD, f[0] + PAD, f[3] + PAD, f[2] + PAD)], dtype=torch.float32, device=dev)
        cover = roi_align(xp, b, output_size=(256, 256), spatial_scale=1.0, sampling_ratio=2, aligned=True).clamp(0, 1) * 2 - 1
        clean = wm.tm.removal(cover).clamp(-1, 1); rhat = cover - clean; M = torch.from_numpy(msgs).to(dev)
        stego, _ = wm.tm.encoder(clean.expand(len(M), -1, -1, -1), M); T = stego.clamp(-1, 1) - clean; T = T - T.mean((2, 3), keepdim=True)        # (1+K) x 3 x 256 x 256
        c = torch.arange(128, device=dev).float() * 4 + 2; gy = (c - f[0]) / (f[2] - f[0]) * 2 - 1; gx = (c - f[1]) / (f[3] - f[1]) * 2 - 1; grid = torch.stack(torch.meshgrid(gx, gy, indexing="xy"), -1)[None]
        samp = lambda t: F.grid_sample(t[None, None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0, 0]
        num = (rhat * T).sum(1, keepdim=True); den = (T * T).sum(1, keepdim=True); er = rhat.pow(2).sum(1, keepdim=True); gs, zs, es = [], [], []
        for sg in SIGS:
            n_ = gauss(num, sg)[:, 0]; d_ = gauss(den, sg)[:, 0]; gk = n_ / (d_ + 0.02 * d_.mean()); gs.append(samp(gk[0])); zs.append(samp((gk[0] - gk[1:].mean(0)) / (gk[1:].std(0) + 0.02)))
            es.append(samp(gauss(er, sg)[0, 0] / (d_[0] + 0.02 * d_.mean())))
        for k, v in (("g", gs), ("zg", zs), ("e", es)): OUTM[k].append(torch.stack(v).cpu().numpy().astype(np.float16))
        if a.split == "train": G.append(downsample_mask(read_mask(nm)))
OUTM = {k: np.stack(v) for k, v in OUTM.items()}; np.savez_compressed(OUT / f"maps_trustmark_{a.split}_rm{a.tag}.npz", names=np.array(names), r3=np.array(r3), gidx=np.array(gidx), frame=np.array(frames), **OUTM)
print("RM", a.split, "cadre =", a.frame, "n=%d K=%d sec=%d" % (len(names), a.K, time.time() - t0_))
if G:
    G = np.stack(G); r3 = np.array(r3); ok = r3 < 0.6
    for key in ("g", "zg", "e"):
        Mx = OUTM[key].astype(np.float32)
        for k, sg in enumerate(SIGS):
            A = np.array([auc(Mx[j, k][G[j] > 0], Mx[j, k][G[j] == 0]) if (G[j] == 0).sum() > 10 else np.nan for j in range(len(G))])
            print("  carte %-2s lissage %2.0f : AUC des trous moyenne %.3f, médiane %.3f, > 0,8 : %d, > 0,9 : %d (sur %d) | toile trouvée seulement : moyenne %.3f, médiane %.3f | valeur médiane dedans %.2f, trou %.2f" % (
                key, sg, np.nanmean(A), np.nanmedian(A), (A > 0.8).sum(), (A > 0.9).sum(), len(A), np.nanmean(A[ok]), np.nanmedian(A[ok]), np.median(Mx[:, k][G > 0]), np.median(Mx[:, k][G == 0])))
