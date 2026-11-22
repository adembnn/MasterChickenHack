"""Test court (vérificateur) : existe-t-il une trace d'édition COMMUNE aux trois filigranes ?
On entraîne un segmenteur sur les vraies images de DEUX filigranes (hors réserve, vrais masques) et on le note sur le TROISIÈME,
qu'il n'a jamais vu et dont il ne connaît pas le signal. S'il dépasse nettement la constante, les zones propres (contenu généré ou greffé,
d'après la consigne) se reconnaissent sans lire le filigrane. Ce n'est pas une méthode candidate : c'est une mesure.
Usage : python xmeth.py <filigrane tenu à l'écart> [pas]"""
import sys, json, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
import segmentation_models_pytorch as smp
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verif import miou_vec, boot

HERE = Path(__file__).resolve().parent.parent; REL = HERE / "release"
held = sys.argv[1]; steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
torch.set_num_threads(4); torch.manual_seed(0); rng = np.random.default_rng(0); dev = "cuda"
meta = json.load(open(REL / "train" / "metadata.json"))
rd = lambda n: np.asarray(Image.open(REL / "train" / "images" / f"{n}.png").convert("RGB"))
rm = lambda n: (np.asarray(Image.open(REL / "train" / "masks" / f"{n}.png").convert("L")) > 127)
X = np.stack([rd(m["image_name"]) for m in meta]); M = np.stack([rm(m["image_name"]) for m in meta])
Y128 = M.reshape(len(M), 128, 4, 128, 4).mean((2, 4)) >= 0.5
meth = np.array([m["method"] for m in meta]); full = np.array([m["full_watermark"] for m in meta]); idx = np.arange(len(meta))
tr = np.where((meth != held) & (idx % 5 != 0))[0]; te_other = np.where(meth == held)[0]; te_res = np.where((meth != held) & (idx % 5 == 0))[0]
print(f"tenu à l'écart : {held} | entraînement {len(tr)} vraies images des deux autres (hors réserve) | test autre filigrane {len(te_other)} | réserve des deux appris {len(te_res)}", flush=True)
Xg = torch.from_numpy(X).to(dev).permute(0, 3, 1, 2); Mg = torch.from_numpy(M).to(dev)
mean = torch.tensor([0.485, 0.456, 0.406], device=dev)[None, :, None, None] * 255; std = torch.tensor([0.229, 0.224, 0.225], device=dev)[None, :, None, None] * 255

def dih(t, k):
    if k & 4: t = t.transpose(-1, -2)
    if k & 1: t = t.flip(-1)
    if k & 2: t = t.flip(-2)
    return t

net = smp.Unet("resnet34", encoder_weights="imagenet", classes=1).to(dev).to(memory_format=torch.channels_last)
opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=steps, pct_start=0.08)
fwd = lambda x: F.avg_pool2d(net(((x.float() - mean) / std).contiguous(memory_format=torch.channels_last)), 4)[:, 0]

@torch.no_grad()
def predict(ids):
    net.eval(); out = []
    for s in range(0, len(ids), 16):
        b = torch.from_numpy(ids[s:s + 16]).to(dev); acc = 0
        for k in (0, 1):
            acc = acc + dih(fwd(dih(Xg[b], k)), k)
        out.append((acc / 2).float().cpu().numpy())
    net.train(); return np.concatenate(out)

def auc(s, l):
    s = s.ravel(); l = l.ravel(); o = np.argsort(s, kind="stable"); r = np.empty(len(o)); r[o] = np.arange(1, len(o) + 1); n1 = l.sum(); n0 = len(l) - n1
    return float((r[l].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))

def show(tag, ids):
    lg = predict(ids); s = miou_vec(lg > 0, Y128[ids]); f = full[ids]; lo, hi = boot(s)
    fr = Y128[ids].reshape(len(ids), -1).mean(1); const = (np.maximum(fr, 1 - fr) / 2).mean()
    aucs = [auc(l, y) for l, y in zip(lg, Y128[ids])]
    print(f"  {tag:34s} n={len(ids):3d} mIoU {s.mean():.4f} IC95 [{lo:.3f} ; {hi:.3f}] pleines {s[f].mean():.4f} partielles {s[~f].mean():.4f} | constante oracle {const:.4f} | AUC par image, médiane {np.median(aucs):.3f}, part > 0,8 : {np.mean(np.array(aucs) > 0.8):.2f}", flush=True)
    return lg

t0 = time.time(); net.train()
for step in range(1, steps + 1):
    b = torch.from_numpy(rng.choice(tr, 12)).to(dev); k = int(rng.integers(8))
    x = dih(Xg[b], k); y = (F.avg_pool2d(dih(Mg[b], k).float()[:, None], 4)[:, 0] >= 0.5).float()
    with torch.autocast("cuda", dtype=torch.bfloat16): lg = fwd(x)
    lg = lg.float(); p = torch.sigmoid(lg); i1 = (p * y).sum((1, 2)); i0 = ((1 - p) * (1 - y)).sum((1, 2))
    dice = 1 - 0.5 * ((2 * i1 + 1) / (p.sum((1, 2)) + y.sum((1, 2)) + 1) + (2 * i0 + 1) / ((1 - p).sum((1, 2)) + (1 - y).sum((1, 2)) + 1))
    loss = F.binary_cross_entropy_with_logits(lg, y) + dice.mean()
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    if step % 500 == 0 or step == steps:
        print(f"pas {step} perte {float(loss):.3f} {round(time.time() - t0)} s", flush=True)
        lo_ = show(f"AUTRE filigrane ({held}), jamais vu", te_other); show("réserve des deux filigranes appris", te_res)
np.savez_compressed(Path(__file__).with_name(f"xmeth_{held}.npz"), names=np.array([meta[i]["image_name"] for i in te_other]), logits=lo_.astype(np.float16))
