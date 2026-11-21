"""T1 · segmenteur appris sur données synthétiques (un réseau par filigrane).
1) gen : fabrique N images (base propre COCO -> tampon réel -> forme tirée des vrais masques du train) avec la réponse exacte.
   - cas "plein"   : toute l'image est filigranée, puis les pixels d'origine sont recollés dans une forme (trou propre) ;
   - cas "partiel" : seul un rectangle (ou une forme d'objet) est filigrané, le reste est l'image propre.
2) train : U-Net ResNet-34, entrée 512, sortie 128x128, perte BCE + Dice.
3) eval : mIoU officiel sur le VRAI train (jamais vu), dont un cinquième "réserve" (index % 5 == 0) dont les formes ne servent pas.
Usage : python seg.py --method pixelseal --n 6000 --epochs 12"""
import os, sys, json, argparse, time, random
os.environ.setdefault("OMP_NUM_THREADS", "8")
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "kit"))
from wmloc.watermarks import load_watermark
from wmloc.data import LocalisationDataset
from wmloc.utils import downsample_mask, miou_image

ap = argparse.ArgumentParser()
ap.add_argument("--method", required=True); ap.add_argument("--n", type=int, default=6000)
ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--bs", type=int, default=24)
ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--tag", default="v2")
ap.add_argument("--arch", default="dec"); ap.add_argument("--real", type=int, default=6)
ap.add_argument("--predict", action="store_true")
ap.add_argument("--coco", default=str(HERE.parent / "shared_data" / "coco" / "val2017"))
a = ap.parse_args()
torch.set_num_threads(8); dev = "cuda"; method = a.method; rng = random.Random(0); nprng = np.random.default_rng(0)
OUT = HERE / "adem" / f"seg_{a.tag}_{method}"; OUT.mkdir(parents=True, exist_ok=True)
def log(*x): print(method, *x, flush=True)

# ---------- vrai train : formes (hors réserve) et évaluation
real = LocalisationDataset(HERE / "release", "train", methods=None)
meta = real.meta
holes, parts = [], []                                  # formes en 512x512 : zone PROPRE des cas pleins, zone FILIGRANÉE des cas partiels
for i, m in enumerate(meta):
    if i % 5 == 0: continue                            # réserve : ses formes ne servent pas
    mk = (np.asarray(Image.open(real.dir / "masks" / f"{m['image_name']}.png")) > 127)
    (holes if m.get("full_watermark") else parts).append(~mk if m.get("full_watermark") else mk)
log("formes", len(holes), "trous,", len(parts), "partiels")

def jitter(shape):                                     # symétries + décalage par roulement (multiples de 4 px)
    s = shape
    if rng.random() < .5: s = s[:, ::-1]
    if rng.random() < .5: s = s[::-1, :]
    if rng.random() < .5: s = s.T
    return np.roll(s, (rng.randrange(0, 512, 4), rng.randrange(0, 512, 4)), axis=(0, 1))

def rand_rect(max_ar):
    for _ in range(100):
        area = 512 * 512 * rng.uniform(0.01, 0.35); ar = np.exp(rng.uniform(-np.log(max_ar), np.log(max_ar)))
        h = int(round(np.sqrt(area / ar) / 4) * 4); w = int(round(np.sqrt(area * ar) / 4) * 4)
        if 32 <= h <= 512 and 32 <= w <= 512:
            y = rng.randrange(0, 512 - h + 1, 4); x = rng.randrange(0, 512 - w + 1, 4); return y, x, h, w
    return 128, 128, 256, 256

def load_base(p):
    im = Image.open(p).convert("RGB"); w, h = im.size; s = min(w, h)
    l = rng.randint(0, w - s); t = rng.randint(0, h - s)
    return np.asarray(im.crop((l, t, l + s, t + s)).resize((512, 512), Image.Resampling.LANCZOS))

def soft_paste(dst, src, mask, k):                     # recolle src dans dst là où mask=1, bord adouci de k px
    m = torch.from_numpy(mask.astype(np.float32))[None, None]
    if k > 0: m = F.avg_pool2d(m, 2 * k + 1, 1, k)
    m = m[0, 0].numpy()[..., None]
    return np.clip(dst * (1 - m) + src * m, 0, 255).round().astype(np.uint8)

# ---------- 1) génération
xp, yp, bp = OUT / "X.npy", OUT / "Y.npy", OUT / "B.npy"
if not xp.exists():
    wm = load_watermark(method, models_dir=HERE / "kit" / "models", device=dev)
    files = sorted(Path(a.coco).glob("*.jpg")); rng.shuffle(files)
    X = np.lib.format.open_memmap(xp, mode="w+", dtype=np.uint8, shape=(a.n, 512, 512, 3))
    Y = np.lib.format.open_memmap(yp, mode="w+", dtype=np.uint8, shape=(a.n, 128, 128))
    B = np.zeros((a.n, wm.nbits), np.uint8)
    max_ar = 2.0 if method == "trustmark" else 16.0; t0 = time.time()
    for i in range(a.n):
        base = load_base(files[i % len(files)]); bits = nprng.integers(0, 2, wm.nbits)
        r = rng.random()
        if r < 0.5:                                    # plein + trou propre
            wmk = wm.embed(base, bits); hole = jitter(holes[rng.randrange(len(holes))])
            img = soft_paste(wmk, base, hole, rng.choice([0, 0, 1, 2])); mask = ~hole
        elif r < 0.85:                                 # partiel : rectangle filigrané
            y, x, h, w = rand_rect(max_ar); img = base.copy()
            img[y:y + h, x:x + w] = wm.embed(np.ascontiguousarray(base[y:y + h, x:x + w]), bits)
            mask = np.zeros((512, 512), bool); mask[y:y + h, x:x + w] = True
        else:                                          # partiel : forme réelle (objet greffé ou rectangle du train)
            wmk = wm.embed(base, bits); shp = jitter(parts[rng.randrange(len(parts))])
            img = soft_paste(base, wmk, shp, rng.choice([0, 0, 1])); mask = shp
        X[i] = img; Y[i] = downsample_mask(mask.astype(np.uint8)); B[i] = bits
        if i % 500 == 0: log("gen", i, round(time.time() - t0), "s")
    X.flush(); Y.flush(); np.save(bp, B); del wm; torch.cuda.empty_cache()
X = np.load(xp, mmap_mode="r"); Y = np.load(yp, mmap_mode="r"); B = np.load(bp)
Image.fromarray(np.concatenate([np.concatenate([X[j], np.repeat(np.repeat(Y[j] * 255, 4, 0), 4, 1)[..., None].repeat(3, 2)], 1) for j in range(4)], 0)).save(OUT / "exemples.png")

# ---------- 2) modèle et entraînement
class FPN(nn.Module):
    def __init__(self, chans, extra=1, d=160):
        super().__init__()
        self.lat = nn.ModuleList([nn.Conv2d(c + (extra if i == len(chans) - 1 else 0), d, 1) for i, c in enumerate(chans)])
        self.smooth = nn.ModuleList([nn.Sequential(nn.Conv2d(d, d, 3, padding=1), nn.GroupNorm(8, d), nn.GELU()) for _ in chans])
        self.head = nn.Sequential(nn.Conv2d(d, d, 3, padding=1), nn.GELU(), nn.Conv2d(d, 1, 1))
    def forward(self, feats):
        h = None
        for i in range(len(feats) - 1, -1, -1):
            l = self.lat[i](feats[i]); h = l if h is None else l + F.interpolate(h, size=l.shape[-2:], mode="bilinear", align_corners=False)
            h = self.smooth[i](h)
        return F.interpolate(self.head(h), size=(128, 128), mode="bilinear", align_corners=False)[:, 0]

def freeze_bn(mod):
    for m_ in mod.modules():
        if isinstance(m_, nn.BatchNorm2d): m_.eval()

if a.arch == "unet":
    # U-Net a encodeur ResNet-34 pre-entraine ImageNet, sans segmentation_models_pytorch (absent du venv).
    import torchvision
    class _Dec(nn.Module):
        def __init__(self, ci, co):
            super().__init__()
            self.net = nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.GroupNorm(8, co), nn.GELU(),
                                     nn.Conv2d(co, co, 3, padding=1), nn.GroupNorm(8, co), nn.GELU())
        def forward(self, x): return self.net(x)
    class _UNet34(nn.Module):
        def __init__(self):
            super().__init__()
            bb = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
            self.stem = nn.Sequential(bb.conv1, bb.bn1, bb.relu)     # 64, /2
            self.p1, self.l1 = bb.maxpool, bb.layer1             # 64, /4
            self.l2, self.l3, self.l4 = bb.layer2, bb.layer3, bb.layer4   # 128/8, 256/16, 512/32
            self.u4 = _Dec(512 + 256, 256); self.u3 = _Dec(256 + 128, 128)
            self.u2 = _Dec(128 + 64, 64); self.u1 = _Dec(64 + 64, 64)
            self.head = nn.Conv2d(64, 1, 1)
        def forward(self, x):
            s = self.stem(x); p = self.l1(self.p1(s))
            a = self.l2(p); b2 = self.l3(a); c = self.l4(b2)
            u = self.u4(torch.cat([F.interpolate(c, size=b2.shape[-2:], mode="nearest"), b2], 1))
            u = self.u3(torch.cat([F.interpolate(u, size=a.shape[-2:], mode="nearest"), a], 1))
            u = self.u2(torch.cat([F.interpolate(u, size=p.shape[-2:], mode="nearest"), p], 1))
            u = self.u1(torch.cat([F.interpolate(u, size=s.shape[-2:], mode="nearest"), s], 1))
            u = F.interpolate(u, size=x.shape[-2:], mode="bilinear", align_corners=False)
            return self.head(u)
    net = _UNet34().to(dev)
    MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev)[None, :, None, None]
    STD = torch.tensor([0.229, 0.224, 0.225], device=dev)[None, :, None, None]
    params = [{"params": net.parameters(), "lr": a.lr}]
    def forward(xb, bits):
        x = (xb.permute(0, 3, 1, 2).float() / 255 - MEAN) / STD
        return F.avg_pool2d(net(x), 4)[:, 0]
else:
    wmk = load_watermark(method, models_dir=HERE / "kit" / "models", device=dev)
    if method == "pixelseal":
        det = wmk.det.float(); trunk = det; lin = det.pixel_decoder.linear; fpn = FPN([96, 192, 384, 768]).to(dev)
        def feats_of(x):
            h = det.preprocess(F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False, antialias=True)); fs = []
            for dl, st in zip(det.convnext.downsample_layers, det.convnext.stages):
                h = st(dl(h)); fs.append(h)
            emb = det.pixel_decoder.output_upscaling(fs[-1])
            pp = torch.einsum("bchw,oc->bohw", emb, lin.weight[1:]) + lin.bias[1:][None, :, None, None]
            return fs, pp
    else:
        r = wmk.net.float(); trunk = r; fpn = FPN([256, 512, 1024, 2048]).to(dev)
        def feats_of(x):
            h = F.interpolate(x, size=(wmk.resolution,) * 2, mode="bilinear", align_corners=False, antialias=True) * 2 - 1
            h = r.maxpool(r.relu(r.bn1(r.conv1(h)))); fs = []
            for k in ("layer1", "layer2", "layer3", "layer4"):
                h = getattr(r, k)(h); fs.append(h)
            pp = torch.einsum("bchw,oc->bohw", h, r.fc.weight) + r.fc.bias[None, :, None, None]
            return fs, pp
    for p in trunk.parameters(): p.requires_grad_(True)
    trunk.train(); freeze_bn(trunk)
    net = nn.ModuleList([trunk, fpn])
    params = [{"params": trunk.parameters(), "lr": a.lr / 10}, {"params": fpn.parameters(), "lr": a.lr}]
    def forward(xb, bits):
        x = xb.permute(0, 3, 1, 2).float() / 255
        fs, pp = feats_of(x); sgn = bits.float() * 2 - 1
        agree = torch.tanh((pp.float() * sgn[:, :, None, None]).mean(1, keepdim=True))
        fs = list(fs); fs[-1] = torch.cat([fs[-1], agree.to(fs[-1].dtype)], 1)
        return fpn(fs)

real_idx = [i for i, m in enumerate(meta) if m["method"] == method]
def evaluate(th=0.0, tta=True):
    net.eval(); sc, res, full = [], [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in real_idx:
            it = real[i]; xb = torch.from_numpy(it["image"])[None].to(dev); bb = torch.from_numpy(it["message"])[None].to(dev)
            lg = forward(xb, bb).float()
            sc.append(miou_image((lg[0] > th).cpu().numpy(), it["mask128"])); res.append(i % 5 == 0); full.append(bool(it.get("full_watermark")))
    sc, res, full = np.array(sc), np.array(res), np.array(full); net.train(); freeze_bn(net) if a.arch != 'unet' else None
    return {"tous": round(float(sc.mean()), 4), "reserve": round(float(sc[res].mean()), 4), "pleins": round(float(sc[full].mean()), 4), "partiels": round(float(sc[~full].mean()), 4)}

if a.predict:
    net.load_state_dict(torch.load(OUT / "best.pt", map_location=dev)); net.eval()
    print(method, "PREDICT train", json.dumps(evaluate()), flush=True); net.eval()
    for split in ("train", "test"):
        dsx = LocalisationDataset(HERE / "release", split, methods=[method]); names, lgs = [], []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for it in dsx:
                xb = torch.from_numpy(it["image"])[None].to(dev); bb = torch.from_numpy(it["message"])[None].to(dev)
                lg = (forward(xb, bb).float() + forward(xb.flip(2), bb).float().flip(2)) / 2
                names.append(it["image_name"]); lgs.append(lg[0].cpu().numpy().astype(np.float16))
        np.savez_compressed(OUT / f"pred_{split}_{method}.npz", names=np.array(names), logits=np.stack(lgs))
        print(method, "PREDICT", split, len(names), "frac>0", round(float((np.stack(lgs) > 0).mean()), 3), flush=True)
    sys.exit(0)
opt = torch.optim.AdamW(params, weight_decay=1e-4)
real_train = [i for i in real_idx if i % 5 != 0]
RX = np.stack([real[i]['image'] for i in real_train]); RY = np.stack([real[i]['mask128'] for i in real_train]); RB = np.stack([real[i]['message'] for i in real_train])
steps = a.epochs * (a.n // a.bs); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g['lr'] for g in params], total_steps=steps, pct_start=0.1)
hist = []; t0 = time.time(); best = -1
for ep in range(a.epochs):
    perm = nprng.permutation(a.n); tot = 0
    for b in range(a.n // a.bs):
        idx = np.sort(perm[b * a.bs:(b + 1) * a.bs])
        xb = torch.from_numpy(X[idx]); yb = torch.from_numpy(Y[idx]); bb = torch.from_numpy(B[idx])
        if a.real:
            ri = nprng.integers(0, len(real_train), a.real)
            xb = torch.cat([xb, torch.from_numpy(RX[ri])]); yb = torch.cat([yb, torch.from_numpy(RY[ri])]); bb = torch.cat([bb, torch.from_numpy(RB[ri])])
        xb, yb, bb = xb.to(dev), yb.to(dev).float(), bb.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = forward(xb, bb).float()
        p = torch.sigmoid(lg); inter = (p * yb).sum((1, 2)); inter0 = ((1 - p) * (1 - yb)).sum((1, 2))
        dice = 1 - 0.5 * ((2 * inter + 1) / (p.sum((1, 2)) + yb.sum((1, 2)) + 1) + (2 * inter0 + 1) / ((1 - p).sum((1, 2)) + (1 - yb).sum((1, 2)) + 1))
        loss = F.binary_cross_entropy_with_logits(lg, yb) + dice.mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); tot += float(loss)
    ev = evaluate(); ev.update(epoch=ep, loss=round(tot / (a.n // a.bs), 4), sec=round(time.time() - t0)); hist.append(ev); log("EPOCH", json.dumps(ev))
    if ev["tous"] > best: best = ev["tous"]; torch.save(net.state_dict(), OUT / "best.pt")
(OUT / "hist.json").write_text(json.dumps(hist, indent=1)); log("RESULT", json.dumps({"best_tous": best, "last": hist[-1]}))
