"""Le vrai filigrane des pleines est-il zoomé ou décalé par rapport au cadre 512 ? On ré-échantillonne les VRAIES images (zoom s autour du centre, décalage)
et on regarde si le réseau fabriqué-seul (et le décodeur) reconnaissent mieux le filigrane, surtout aux bords."""
import sys, argparse, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); import reel2 as R
from wmloc.watermarks import load_watermark
method, tag, N = sys.argv[1], sys.argv[2], int(sys.argv[3]); dev = "cuda"
trm, tem = R.metas(); RT = np.load(R.CACHE / "REAL_train.npy", mmap_mode="r"); Y = np.load(R.CACHE / "Y128_train.npy")
run = R.Runner(argparse.Namespace(enc="resnet34", arch="unet", pretrained=0, stem1=0, flag=0, hp=0, prec="fp16"), dev)
P = R.ROOT / "runs" / tag / method / "fold-1"; run.net.load_state_dict(torch.load(P / "ema.pt", map_location=dev)["state"] if (P / "ema.pt").exists() else torch.load(P / "ckpt.pt", map_location=dev)["ema"])
wm = load_watermark(method, models_dir=R.HERE / "kit" / "models", device=dev)
fulls = [i for i, m in enumerate(trm) if m["method"] == method and i % 5 != 0 and m["full_watermark"]][:N]
X = torch.from_numpy(np.stack([np.ascontiguousarray(RT[i]) for i in fulls])).permute(0, 3, 1, 2).float().to(dev); G = torch.from_numpy(np.stack([Y[i] for i in fulls])).float().to(dev)[:, None]
sgn = torch.tensor(np.stack([[1.0 if c == "1" else -1.0 for c in trm[i]["message"]] for i in fulls]), device=dev).float()
import math
def warp(T, s, dx=0.0, dy=0.0, mode="bicubic", rot=0.0):   # sortie(p) = entrée(centre + s*R(rot)*(p - centre) + d) : s < 1 agrandit l'image
    c, n = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    th = torch.tensor([[s * c, -s * n, 2 * dx / 512], [s * n, s * c, 2 * dy / 512]], device=dev).float()[None].repeat(len(T), 1, 1)
    return F.grid_sample(T, F.affine_grid(th, T.shape, align_corners=False), mode=mode, padding_mode="reflection", align_corners=False)
edge = torch.zeros(1, 1, 128, 128, device=dev); edge[..., :16, :] = 1; edge[..., -16:, :] = 1; edge[..., :, :16] = 1; edge[..., :, -16:] = 1
def measure(s, dx=0.0, dy=0.0, rot=0.0):
    Xw = warp(X, s, dx, dy, rot=rot).clamp(0, 255).round(); Gw = (warp(G, s, dx, dy, "nearest", rot=rot) > 0.5).float()
    lg = torch.from_numpy(run.predict(run.net, Xw.permute(0, 2, 3, 1).byte().cpu().numpy(), np.ones(N), tta=1, bs=4)).to(dev)[:, None]
    dl = torch.cat([wm.logits(Xw[j:j + 16] / 255).to(dev) for j in range(0, N, 16)]) * sgn
    return f"logit filigrané centre {float((lg * Gw * (1 - edge)).sum() / (Gw * (1 - edge)).sum()):+.2f} | bords {float((lg * Gw * edge).sum() / (Gw * edge).sum()):+.2f} | bits justes {float((dl > 0).float().mean()):.3f} | marge décodeur {float(dl.mean()):.2f}"
print(f"## {method} : {N} vraies pleines ré-échantillonnées")
for s in (0.90, 0.94, 0.97, 0.985, 1.0, 1.015, 1.03, 1.06, 1.10): print(f"zoom s={s:<6}", measure(s))
for d in (-8, -4, -2, 2, 4, 8): print(f"décalage x {d:+d} px", measure(1.0, dx=d)); print(f"décalage y {d:+d} px", measure(1.0, dy=d))
for r in (-4, -2, -1, -0.5, 0.5, 1, 2, 4): print(f"rotation {r:+} deg ", measure(1.0, rot=r))
# par image : la meilleure transformation selon la marge du décodeur (si chaque image a subi sa propre transformation, le gain par image sera grand)
grid = [(s_, r_) for s_ in (0.94, 0.96, 0.98, 0.99, 1.0, 1.01, 1.02, 1.04, 1.06) for r_ in (-3, -2, -1, 0, 1, 2, 3)]
M = torch.stack([torch.cat([(wm.logits(warp(X[j:j + 16], s_, rot=r_).clamp(0, 255).round() / 255).to(dev) * sgn[j:j + 16]).mean(1) for j in range(0, N, 16)]) for s_, r_ in grid])   # grille x images
b = M.argmax(0).cpu().numpy(); i0 = grid.index((1.0, 0)); gain = (M.max(0).values - M[i0]).cpu().numpy()
print("par image, meilleure (zoom, rotation) selon le décodeur :", sorted(set((grid[k], int((b == k).sum())) for k in b), key=lambda t: -t[1])[:8])
print(f"marge du décodeur : identité {float(M[i0].mean()):.2f} -> meilleure par image {float(M.max(0).values.mean()):.2f} ; gain médian {np.median(gain):.2f} ; part des images où l'identité est la meilleure : {(b == i0).mean():.2f}")
