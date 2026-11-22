"""StegaStamp : le z du décodeur est-il séparable par axe ? et débit du décodeur. Rectangles réels hors réserve."""
import time, numpy as np, torch
from torchvision.ops import roi_align
from rlib import *
dev = "cuda"; wm = load("stegastamp", dev)
xt = torch.rand(8, 3, 400, 400); l0 = wm.logits(xt); ss_decoder_only(wm); l1 = wm.logits(xt); print("décodeur seul : écart max des logits %.5f (échelle %.2f)" % (float((l0 - l1).abs().max()), float(l0.abs().mean())))
def zdec(x, boxes, s, bs=64):
    zs = []
    for i in range(0, len(boxes), bs):
        b = torch.tensor([(0, bb[1], bb[0], bb[3], bb[2]) for bb in boxes[i:i + bs]], dtype=torch.float32, device=dev)
        cr = roi_align(x, b, output_size=(400, 400), spatial_scale=1.0, sampling_ratio=2, aligned=True).clamp(0, 1)
        lg = wm.logits(cr, bs=bs).to(dev); zs.append((lg * s).sum(1) / lg.pow(2).sum(1).sqrt().clamp_min(1e-9))
    return torch.cat(zs).cpu().numpy()
def bbox(mask):
    ys, xs = np.where(mask > 0); return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1
def cb(b):
    y0, x0, y1, x1 = b; y0 = max(0, min(y0, 500)); x0 = max(0, min(x0, 500)); return (y0, x0, max(y0 + 8, min(y1, 512)), max(x0 + 8, min(x1, 512)))
items = []
for e in meta("train"):
    if e["method"] != "stegastamp" or e.get("full_watermark") or e["gidx"] % 5 == 0: continue
    mk = read_mask(e["image_name"]); g = bbox(mk)
    if mk[g[0]:g[2], g[1]:g[3]].mean() > 0.98: items.append((e, g))
P = {"exact": lambda g, h, w: [g]}
for f in (0.05, 0.1, 0.2, 0.4):
    P["x décalé %d%% de la largeur" % (f * 100)] = lambda g, h, w, f=f: [cb((g[0], g[1] + int(round(f * w)), g[2], g[3] + int(round(f * w)))), cb((g[0], g[1] - int(round(f * w)), g[2], g[3] - int(round(f * w))))]
    P["y décalé %d%% de la hauteur" % (f * 100)] = lambda g, h, w, f=f: [cb((g[0] + int(round(f * h)), g[1], g[2] + int(round(f * h)), g[3])), cb((g[0] - int(round(f * h)), g[1], g[2] - int(round(f * h)), g[3]))]
    P["largeur +%d%%" % (f * 100)] = lambda g, h, w, f=f: [cb((g[0], g[1] - int(round(f * w / 2)), g[2], g[3] + int(round(f * w / 2))))]
    P["largeur -%d%%" % (f * 100)] = lambda g, h, w, f=f: [cb((g[0], g[1] + int(round(f * w / 2)), g[2], g[3] - int(round(f * w / 2))))]
    P["hauteur +%d%%" % (f * 100)] = lambda g, h, w, f=f: [cb((g[0] - int(round(f * h / 2)), g[1], g[2] + int(round(f * h / 2)), g[3]))]
    P["les deux +%d%%" % (f * 100)] = lambda g, h, w, f=f: [cb((g[0] - int(round(f * h / 2)), g[1] - int(round(f * w / 2)), g[2] + int(round(f * h / 2)), g[3] + int(round(f * w / 2))))]
acc = {k: [] for k in P}; zex = []; sizes = []
for e, g in items:
    x = to_t(read_img("train", e["image_name"]), dev); s = torch.from_numpy(2 * bits_of(e) - 1).float().to(dev); h, w = g[2] - g[0], g[3] - g[1]
    for k, f in P.items(): acc[k].append(zdec(x, f(g, h, w), s).mean())
    zex.append(acc["exact"][-1]); sizes.append(min(h, w))
zex = np.array(zex); sizes = np.array(sizes)
print("SSTEST n=%d rectangles ; z exact par taille du petit côté : <24 px %.1f (n=%d) ; 24-48 %.1f (n=%d) ; >48 %.1f (n=%d)" % (len(items), zex[sizes < 24].mean(), (sizes < 24).sum(), zex[(sizes >= 24) & (sizes < 48)].mean(), ((sizes >= 24) & (sizes < 48)).sum(), zex[sizes >= 48].mean(), (sizes >= 48).sum()))
ok = zex > 5
for k, v in acc.items(): print("  %-32s z moyen %.2f (sur les %d lisibles à la boîte exacte)" % (k, np.array(v)[ok].mean(), ok.sum()))
x = torch.rand(1, 3, 512, 512, device=dev); s = torch.ones(100, device=dev); bx = [(0, 0, 100, 100)] * 1024; t0 = time.time(); zdec(x, bx, s); print("débit décodeur : %.0f rognages/s" % (1024 / (time.time() - t0)))
