"""Pleines TrustMark, carte affinée : au lieu du simple rapport, un z-score local. À la toile déjà trouvée (fichier maps_*_pleine*.npz, clé frame),
on calcule l'énergie lissée de ce que le tampon ajouterait avec le VRAI message et avec K messages au hasard, puis par case :
z = (ln E_vrai - moyenne_k ln E_hasard) / écart-type_k ln E_hasard. Très négatif = filigrane reconnu ; ~0 = trou. Tient compte du bruit local.
Sortie : maps_trustmark_<split>_pz<tag>.npz (names, z : N x 5 lissages x 128 x 128, r3, gidx) + AUC des trous sur le train."""
import argparse, time
import numpy as np, torch, torch.nn.functional as F
from torchvision.ops import roi_align
from rlib import *
ap = argparse.ArgumentParser(); ap.add_argument("--split", default="train"); ap.add_argument("--src", default="maps_trustmark_train_pleine_s0of2.npz,maps_trustmark_train_pleine_s1of2.npz")
ap.add_argument("--K", type=int, default=24); ap.add_argument("--tag", default=""); a = ap.parse_args()
dev = "cuda"; wm = load("trustmark", dev); rng = np.random.default_rng(5); PAD = 96; SIGS = (3.0, 6.0, 10.0, 16.0, 24.0)
names, frames, r3, gidx = [], [], [], []
for f in a.src.split(","):
    m = np.load(OUT / f, allow_pickle=True); names += m["names"].tolist(); frames += [tuple(int(v) for v in t) for t in m["frame"]]; r3 += m["r3"].tolist(); gidx += m["gidx"].tolist()
mb = {e["image_name"]: e for e in meta(a.split)}; Z = []; G = []; t0 = time.time()
with torch.no_grad():
    for i, (nm, f) in enumerate(zip(names, frames)):
        x = to_t(read_img(a.split, nm), dev); xp = F.pad(x, (PAD,) * 4, mode="reflect"); m = bits_of(mb[nm]); msgs = np.concatenate([m[None], rng.integers(0, 2, (a.K, wm.nbits)).astype(np.float32)])
        b = torch.tensor([(0, f[1] + PAD, f[0] + PAD, f[3] + PAD, f[2] + PAD)], dtype=torch.float32, device=dev)
        cover = roi_align(xp, b, output_size=(256, 256), spatial_scale=1.0, sampling_ratio=2, aligned=True).clamp(0, 1) * 2 - 1; M = torch.from_numpy(msgs).to(dev); raw = []
        for j in range(0, len(M), 13):
            with torch.autocast("cuda", dtype=torch.float16): stego, _ = wm.tm.encoder(cover.expand(len(M[j:j + 13]), -1, -1, -1), M[j:j + 13])
            res = stego.float().clamp(-1, 1) - cover; res = res - res.mean((2, 3), keepdim=True); raw.append(res.pow(2).sum(1, keepdim=True))
        raw = torch.cat(raw); c = torch.arange(128, device=dev).float() * 4 + 2; gy = (c - f[0]) / (f[2] - f[0]) * 2 - 1; gx = (c - f[1]) / (f[3] - f[1]) * 2 - 1
        grid = torch.stack(torch.meshgrid(gx, gy, indexing="xy"), -1)[None]; zs = []
        for sg in SIGS:
            en = gauss(raw, sg)[:, 0]; l = torch.log(en + 0.02 * en[1:].mean()); z = (l[0] - l[1:].mean(0)) / (l[1:].std(0) + 0.05)
            zs.append(F.grid_sample(z[None, None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0, 0])
        Z.append(torch.stack(zs).cpu().numpy().astype(np.float16))
        if a.split == "train": G.append(downsample_mask(read_mask(nm)))
Z = np.stack(Z); np.savez_compressed(OUT / f"maps_trustmark_{a.split}_pz{a.tag}.npz", names=np.array(names), z=Z, r3=np.array(r3), gidx=np.array(gidx), frame=np.array(frames))
print("PZ", a.split, "n=%d K=%d sec=%d" % (len(names), a.K, time.time() - t0))
if G:
    G = np.stack(G); r3 = np.array(r3); Zf = Z.astype(np.float32)
    for k, sg in enumerate(SIGS):
        A = np.array([auc(Zf[j, k][G[j] == 0], Zf[j, k][G[j] > 0]) if (G[j] == 0).sum() > 10 else np.nan for j in range(len(G))]); ok = r3 < 0.6
        print("  lissage %2.0f : AUC des trous moyenne %.3f, médiane %.3f, > 0,8 : %d/%d, > 0,9 : %d | toile trouvée seulement : moyenne %.3f, médiane %.3f | z médian dedans %.1f, trou %.1f" % (
            sg, np.nanmean(A), np.nanmedian(A), (A > 0.8).sum(), len(A), (A > 0.9).sum(), np.nanmean(A[ok]), np.nanmedian(A[ok]), np.median(Zf[:, k][G > 0]), np.median(Zf[:, k][G == 0])))
