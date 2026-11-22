"""Pleines TrustMark, au bon cadre : variantes proposées par l'orchestrateur.
 A  ln(e_m / e_inv)      : énergie ajoutée avec le vrai message / avec le message INVERSÉ bit à bit
 B  z-score de ln e_m contre K messages au hasard (carte actuelle, pour comparaison dans la même passe)
 C  cosinus local entre le résidu du vrai message et celui du message inversé
 E  ln(e_inv / moyenne e_k) : le message inversé coûte-t-il plus cher que le hasard là où le filigrane est présent ?
 F  cosinus local entre le résidu du vrai message et la moyenne des résidus des messages au hasard
AUC des trous (0,5 = hasard ; on donne max(AUC, 1-AUC) et le sens)."""
import argparse, time
import numpy as np, torch, torch.nn.functional as F
from torchvision.ops import roi_align
from rlib import *
ap = argparse.ArgumentParser(); ap.add_argument("--src", default="maps_trustmark_train_pleine_s0of2.npz,maps_trustmark_train_pleine_s1of2.npz"); ap.add_argument("--K", type=int, default=12); a = ap.parse_args()
dev = "cuda"; wm = load("trustmark", dev); rng = np.random.default_rng(11); PAD = 96; SIGS = (6.0, 10.0, 16.0)
names, frames, r3 = [], [], []
for f in a.src.split(","):
    m = np.load(OUT / f, allow_pickle=True); names += m["names"].tolist(); frames += [tuple(int(v) for v in t) for t in m["frame"]]; r3 += m["r3"].tolist()
mb = {e["image_name"]: e for e in meta("train")}; KEYS = ("A", "B", "C", "E", "F"); MAPS = {k: [] for k in KEYS}; G = []; t0 = time.time()
with torch.no_grad():
    for nm, f in zip(names, frames):
        x = to_t(read_img("train", nm), dev); xp = F.pad(x, (PAD,) * 4, mode="reflect"); m = bits_of(mb[nm]); msgs = np.concatenate([m[None], (1 - m)[None], rng.integers(0, 2, (a.K, wm.nbits)).astype(np.float32)])
        b = torch.tensor([(0, f[1] + PAD, f[0] + PAD, f[3] + PAD, f[2] + PAD)], dtype=torch.float32, device=dev)
        cover = roi_align(xp, b, output_size=(256, 256), spatial_scale=1.0, sampling_ratio=2, aligned=True).clamp(0, 1) * 2 - 1; M = torch.from_numpy(msgs).to(dev)
        stego, _ = wm.tm.encoder(cover.expand(len(M), -1, -1, -1), M); res = stego.clamp(-1, 1) - cover; res = res - res.mean((2, 3), keepdim=True)
        c = torch.arange(128, device=dev).float() * 4 + 2; gy = (c - f[0]) / (f[2] - f[0]) * 2 - 1; gx = (c - f[1]) / (f[3] - f[1]) * 2 - 1; grid = torch.stack(torch.meshgrid(gx, gy, indexing="xy"), -1)[None]
        samp = lambda t: F.grid_sample(t[None, None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[0, 0]
        en = res.pow(2).sum(1, keepdim=True); pmi = (res[0] * res[1]).sum(0)[None, None]; pmk = (res[0] * res[2:].mean(0)).sum(0)[None, None]; ek2 = res[2:].mean(0).pow(2).sum(0)[None, None]; out = {k: [] for k in KEYS}
        for sg in SIGS:
            e = gauss(en, sg)[:, 0]; eps = 0.02 * e[2:].mean(); l = torch.log(e + eps)
            out["A"].append(samp(l[0] - l[1])); out["B"].append(samp((l[0] - l[2:].mean(0)) / (l[2:].std(0) + 0.05))); out["E"].append(samp(l[1] - l[2:].mean(0)))
            out["C"].append(samp(gauss(pmi, sg)[0, 0] / (e[0] * e[1] + eps * eps).sqrt())); out["F"].append(samp(gauss(pmk, sg)[0, 0] / (e[0] * gauss(ek2, sg)[0, 0] + eps * eps).sqrt()))
        for k in KEYS: MAPS[k].append(torch.stack(out[k]).cpu().numpy())
        G.append(downsample_mask(read_mask(nm)))
G = np.stack(G); r3 = np.array(r3); ok = r3 < 0.6; print("PINV n=%d (toile fiable %d) K=%d sec=%d" % (len(names), ok.sum(), a.K, time.time() - t0))
np.savez_compressed(OUT / "maps_trustmark_train_pinv.npz", names=np.array(names), r3=r3, **{k: np.stack(v).astype(np.float16) for k, v in MAPS.items()})
for k in KEYS:
    Mx = np.stack(MAPS[k])
    for j, sg in enumerate(SIGS):
        A = np.array([auc(Mx[i, j][G[i] == 0], Mx[i, j][G[i] > 0]) if (G[i] == 0).sum() > 10 else np.nan for i in range(len(G))]); sens = "trou = valeur haute" if np.nanmean(A) >= 0.5 else "trou = valeur basse"
        B = np.maximum(A, 1 - A) if False else (A if np.nanmean(A) >= 0.5 else 1 - A)
        print("  carte %s lissage %2.0f (%s) : AUC des trous moyenne %.3f, médiane %.3f, > 0,8 : %d, > 0,9 : %d | toile fiable : moyenne %.3f, médiane %.3f | médiane dedans %+.2f, trou %+.2f" % (
            k, sg, sens, np.nanmean(B), np.nanmedian(B), (B > 0.8).sum(), (B > 0.9).sum(), np.nanmean(B[ok]), np.nanmedian(B[ok]), np.median(Mx[:, j][G > 0]), np.median(Mx[:, j][G == 0])))
