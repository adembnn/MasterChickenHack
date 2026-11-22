"""Images PLEINES : le z du décodeur monte-t-il pour un cadre plus grand, plus petit ou décalé par rapport à l'image entière ?
(l'image est prolongée par recopie du bord quand le cadre dépasse). Balayage large : échelle commune, échelle par axe, décalage."""
import argparse, time
import numpy as np, torch, torch.nn.functional as F
from torchvision.ops import roi_align
from rlib import *
ap = argparse.ArgumentParser(); ap.add_argument("--method", default="trustmark"); ap.add_argument("--n", type=int, default=8); a = ap.parse_args()
dev = "cuda"; wm = load(a.method, dev); RES = {"pixelseal": 256, "trustmark": 245, "stegastamp": 400}[a.method]; PAD = 256
if a.method == "stegastamp": ss_decoder_only(wm)
@torch.no_grad()
def zdec(xp, boxes, s):
    b = torch.tensor([(0, t[1] + PAD, t[0] + PAD, t[3] + PAD, t[2] + PAD) for t in boxes], dtype=torch.float32, device=dev); zs = []
    for i in range(0, len(b), 32):
        cr = roi_align(xp, b[i:i + 32], output_size=(RES, RES), spatial_scale=1.0, sampling_ratio=2, aligned=True).clamp(0, 1)
        lg = wm.logits(cr).float().to(dev); zs.append((lg * s).sum(1) / lg.pow(2).sum(1).sqrt().clamp_min(1e-9))
    return torch.cat(zs).cpu().numpy()
M_ = [e for e in meta("train") if e["method"] == a.method and e["gidx"] % 5 != 0 and e.get("full_watermark")][:a.n]
SC = [0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0]; SH = [-64, -32, -16, -8, -4, 0, 4, 8, 16, 32, 64]
res = {"échelle commune": np.zeros(len(SC)), "échelle hauteur seule": np.zeros(len(SC)), "échelle largeur seule": np.zeros(len(SC)), "décalage vertical": np.zeros(len(SH)), "décalage horizontal": np.zeros(len(SH))}
def fr(sy, sx, dy=0, dx=0):
    h = 512 * sy; w = 512 * sx; return (256 - h / 2 + dy, 256 - w / 2 + dx, 256 + h / 2 + dy, 256 + w / 2 + dx)
for e in M_:
    x = to_t(read_img("train", e["image_name"]), dev); xp = F.pad(x, (PAD,) * 4, mode="replicate"); s = torch.from_numpy(2 * bits_of(e) - 1).float().to(dev)
    res["échelle commune"] += zdec(xp, [fr(c, c) for c in SC], s); res["échelle hauteur seule"] += zdec(xp, [fr(c, 1) for c in SC], s); res["échelle largeur seule"] += zdec(xp, [fr(1, c) for c in SC], s)
    res["décalage vertical"] += zdec(xp, [fr(1, 1, d, 0) for d in SH], s); res["décalage horizontal"] += zdec(xp, [fr(1, 1, 0, d) for d in SH], s)
print("FRAME2", a.method, "n=%d pleines ; z moyen du décodeur (maximum %.1f)" % (len(M_), np.sqrt(wm.nbits)))
for k, v in res.items():
    ax = SC if "échelle" in k else SH; print("  %-22s " % k + "  ".join("%s:%.1f" % (c, z / len(M_)) for c, z in zip(ax, v)))
