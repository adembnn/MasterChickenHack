"""T1 · chaîne v3 « réel » : segmenteur entraîné sur des exemples fabriqués avec la recette mesurée du hackathon + les vraies images.
Recette (vérifiée par check.py) : pleine = image filigranée en entier puis pixels propres recollés dans un trou ;
rectangle = image source entière filigranée dans son propre cadre, collée dans une image propre ; objet = image filigranée en entier (cadre 512),
dont on ne garde que la forme d'un objet. Fonds propres = images du hackathon des deux AUTRES filigranes (+ COCO).
Sous-commandes : cache | gen | train | agg.   Tout s'écrit dans adem/t1-reel/."""
import os, sys, json, argparse, time, copy
NT = int(os.environ.get("T1_THREADS", "8")); os.environ.setdefault("OMP_NUM_THREADS", str(NT))
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
ROOT = Path(__file__).resolve().parent; HERE = ROOT.parent if (ROOT.parent / "kit").exists() else ROOT.parent.parent; CACHE = ROOT / "cache"   # Jülich : adem/t1-reel ; H200 : ~/eac/t1/t1-reel
sys.path.insert(0, str(HERE / "kit"))
from wmloc.data import LocalisationDataset
from wmloc.utils import downsample_mask, miou_image
METHODS = ["pixelseal", "trustmark", "stegastamp"]

def metas():
    return json.loads((HERE / "release/train/metadata.json").read_text()), json.loads((HERE / "release/test/metadata.json").read_text())

def dih_t(x, k):                                         # une des 8 symétries du carré, sur les 2 derniers axes d'un tenseur
    if k & 4: x = x.transpose(-1, -2)
    if k & 1: x = x.flip(-1)
    if k & 2: x = x.flip(-2)
    return x
def dih_inv(x, k):
    if k & 2: x = x.flip(-2)
    if k & 1: x = x.flip(-1)
    if k & 4: x = x.transpose(-1, -2)
    return x
def dih_np(a, k):
    if k & 4: a = a.swapaxes(0, 1)
    if k & 1: a = a[:, ::-1]
    if k & 2: a = a[::-1]
    return np.ascontiguousarray(a)

def splits(trm, method):
    """réserve = index % 5 == 0 ; les autres sont rangés en 5 plis, équilibrés pleines / partielles."""
    idx = [i for i, m in enumerate(trm) if m["method"] == method]
    res = [i for i in idx if i % 5 == 0]; rest = [i for i in idx if i % 5 != 0]
    order = sorted(rest, key=lambda i: (not trm[i]["full_watermark"], i))
    fold = {i: r % 5 for r, i in enumerate(order)}
    return idx, res, rest, fold

def miou_list(pred, gt): return np.array([miou_image(p, g) for p, g in zip(pred, gt)])
def auc(score, lab):
    score = score.ravel(); lab = lab.ravel() > 0; o = np.argsort(score, kind="stable"); r = np.empty(len(o)); r[o] = np.arange(1, len(o) + 1)
    n1 = lab.sum(); n0 = len(lab) - n1
    return float((r[lab].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))

# ------------------------------------------------------------------ cache : toutes les images décodées une fois
def cmd_cache(a):
    CACHE.mkdir(parents=True, exist_ok=True)
    if (CACHE / "done").exists(): return
    tr = LocalisationDataset(HERE / "release", "train"); trm, tem = metas()
    rd = lambda p: np.asarray(Image.open(p).convert("RGB"))
    np.save(CACHE / "REAL_train.npy", np.stack([rd(HERE / "release/train/images" / f"{m['image_name']}.png") for m in trm]))
    np.save(CACHE / "MASK_train.npy", np.stack([(np.asarray(Image.open(HERE / "release/train/masks" / f"{m['image_name']}.png").convert("L")) > 127) for m in trm]).astype(np.uint8))
    np.save(CACHE / "Y128_train.npy", np.stack([tr.m128[m["image_name"]] for m in trm]).astype(np.uint8))
    np.save(CACHE / "REAL_test.npy", np.stack([rd(HERE / "release/test/images" / f"{m['image_name']}.png") for m in tem]))
    (CACHE / "done").write_text("ok"); print("cache ok", flush=True)

# ------------------------------------------------------------------ gen : ingrédients fabriqués pour un filigrane
def cmd_gen(a):
    from wmloc.watermarks import load_watermark
    method = a.method; D = ROOT / f"data_{method}{a.suffix}"; D.mkdir(parents=True, exist_ok=True)
    if (D / "done").exists(): print("gen déjà fait"); return
    rng = np.random.default_rng(100 + METHODS.index(method) + 1000 * a.gseed); t0 = time.time()
    wm = load_watermark(method, models_dir=HERE / "kit" / "models", device="cuda")
    trm, tem = metas(); RT = np.load(CACHE / "REAL_train.npy", mmap_mode="r"); MT = np.load(CACHE / "MASK_train.npy", mmap_mode="r"); RE = np.load(CACHE / "REAL_test.npy", mmap_mode="r")
    hb_tr = [i for i, m in enumerate(trm) if m["method"] != method and i % 5 != 0]; hb_te = [i for i, m in enumerate(tem) if m["method"] != method]
    if a.small: hb_tr, hb_te = hb_tr[:12], hb_te[:12]
    ent = []
    for kind, lst in ((0, hb_tr), (1, hb_te)):
        for i in lst:
            for k in [0] + list(rng.choice(np.arange(1, 8), a.nvar - 1, replace=False)): ent.append((kind, i, int(k)))
    files = sorted(Path(a.coco).glob("*.jpg")); rng.shuffle(files)
    ent += [(2, j, 0) for j in range(a.ncoco)]
    def coco(p):
        im = Image.open(p).convert("RGB"); w, h = im.size; s = min(w, h); l = int(rng.integers(0, w - s + 1)); t = int(rng.integers(0, h - s + 1))
        return np.asarray(im.crop((l, t, l + s, t + s)).resize((512, 512), Image.Resampling.LANCZOS))
    def embed_pad(img, bits, py, px, t=None, l=None):   # filigrane posé sur une toile plus grande (bords en miroir) puis rognée ; t, l = marges haut et gauche si rognage décentré
        H, W = img.shape[:2]; py, px = min(py, (H - 1) // 1), min(px, (W - 1) // 1)
        if t is None: t, b_, l, r_ = py, py, px, px
        else: b_, r_ = 2 * py - t, 2 * px - l
        t, b_, l, r_ = [int(min(v, H - 1 if k < 2 else W - 1)) for k, v in enumerate((t, b_, l, r_))]
        if t + b_ + l + r_ == 0: return wm.embed(img, bits)
        c = np.pad(img, ((t, b_), (l, r_), (0, 0)), mode="reflect"); return wm.embed(np.ascontiguousarray(c), bits)[t:t + H, l:l + W]
    def embed_full(img, bits):
        if a.canvas == "real":                           # mesure de t1-residus sur 75 vraies pleines : toile CARRÉE de côté 512 + s (s médian 32, jusqu'à ~80 ; 20 % au cadre exact), rognage placé au hasard
            sz = int(rng.integers(0, 9)) if rng.random() < 0.2 else int(rng.integers(8, a.smax + 1)); return embed_pad(img, bits, sz / 2, sz / 2, int(rng.integers(0, sz + 1)), int(rng.integers(0, sz + 1)))
        return embed_pad(img, bits, int(rng.integers(a.padlo, a.padhi + 1)), int(rng.integers(a.padlo, a.padhi + 1)))
    def embed_rect(img, bits):
        f = rng.uniform(a.rfraclo, a.rfrachi); q = int(rng.integers(a.rpadlo, a.rpadhi + 1)); return embed_pad(img, bits, int(round(f * img.shape[0])) + q, int(round(f * img.shape[1])) + q)
    NB = len(ent); mm = lambda n, shape, dt=np.uint8: np.lib.format.open_memmap(D / n, mode="w+", dtype=dt, shape=shape)
    BASE = mm("BASE.npy", (NB, 512, 512, 3)); WFULL = mm("WFULL.npy", (NB, 512, 512, 3))
    for n, (kind, i, k) in enumerate(ent):
        b = dih_np(RT[i], k) if kind == 0 else dih_np(RE[i], k) if kind == 1 else coco(files[i])
        BASE[n] = b; WFULL[n] = embed_full(b, rng.integers(0, 2, wm.nbits))
        if n % 500 == 0: print(method, "base", n, NB, round(time.time() - t0), "s", flush=True)
    np.save(D / "BINFO.npy", np.array(ent, dtype=np.int32))
    # rectangles : image source entière filigranée dans son cadre (aire 0,3 % à 45 %, loi log-uniforme, médiane ~3,7 % comme le réel)
    maxar = 1.98 if method == "trustmark" else 16.0; NR = a.nrect
    RECT = mm("RECT.npy", (NR, 512, 512, 3)); RBOX = np.zeros((NR, 4), np.int32); RSRC = np.zeros(NR, np.int32)
    for r in range(NR):
        while True:
            area = 512 * 512 * np.exp(rng.uniform(np.log(0.003), np.log(0.45)))
            ar = np.exp(rng.uniform(0, np.log(min(maxar, 3.0) if rng.random() < 0.5 else maxar))) ** rng.choice([-1, 1])
            h = int(round(np.sqrt(area / ar))); w = int(round(np.sqrt(area * ar)))
            if 10 <= h <= 512 and 10 <= w <= 512 and max(h, w) / min(h, w) <= maxar: break
        y0 = int(rng.integers(0, 512 - h + 1)); x0 = int(rng.integers(0, 512 - w + 1)); n = int(rng.integers(NB))
        img = np.array(BASE[n]); img[y0:y0 + h, x0:x0 + w] = embed_rect(np.ascontiguousarray(img[y0:y0 + h, x0:x0 + w]), rng.integers(0, 2, wm.nbits))
        RECT[r] = img; RBOX[r] = (y0, x0, h, w); RSRC[r] = n
        if r % 500 == 0: print(method, "rect", r, NR, round(time.time() - t0), "s", flush=True)
    np.save(D / "RBOX.npy", RBOX); np.save(D / "RSRC.npy", RSRC)
    # « alignés » : vraies images des autres filigranes, re-filigranées avec le nôtre exactement là où leur vrai masque le dit
    AL, ALY, ALF, ALI = [], [], [], []
    for i in hb_tr:
        img = np.array(RT[i]); M = np.array(MT[i]).astype(bool); ys, xs = np.where(M); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        isrect = M.sum() >= 0.98 * (y1 - y0) * (x1 - x0); full = bool(trm[i]["full_watermark"])
        for rep in range(a.nalign):
            bits = rng.integers(0, 2, wm.nbits)
            if full or not isrect: out = np.where(M[..., None], embed_full(img, bits), img)
            else:
                h, w = y1 - y0, x1 - x0
                if max(h, w) / min(h, w) > maxar: continue
                out = img.copy(); out[y0:y1, x0:x1] = embed_rect(np.ascontiguousarray(img[y0:y1, x0:x1]), bits)
            AL.append(out); ALY.append(downsample_mask(M.astype(np.uint8))); ALF.append(full); ALI.append(i)
    np.save(D / "AL.npy", np.stack(AL)); np.save(D / "ALY.npy", np.stack(ALY)); np.save(D / "ALF.npy", np.array(ALF)); np.save(D / "ALI.npy", np.array(ALI))
    # bits justes de l'image entière (drapeau pleine / partielle), train et test du filigrane
    def aall(X, meta_list):
        out = []
        for j in range(0, len(X), 16):
            lg = wm.logits(torch.from_numpy(np.ascontiguousarray(X[j:j + 16])).permute(0, 3, 1, 2).float() / 255).cpu().numpy() > 0
            for l, m in zip(lg, meta_list[j:j + 16]): out.append(float((l == np.array([int(c) for c in m["message"]])).mean()))
        return np.array(out, np.float32)
    own_tr = [i for i, m in enumerate(trm) if m["method"] == method]; own_te = [i for i, m in enumerate(tem) if m["method"] == method]
    np.save(D / "AALL_train.npy", aall(RT[own_tr], [trm[i] for i in own_tr])); np.save(D / "AALL_test.npy", aall(RE[own_te], [tem[i] for i in own_te]))
    BASE.flush(); WFULL.flush(); RECT.flush(); (D / "done").write_text("ok"); print(method, "gen ok", NB, "bases", NR, "rect", len(AL), "alignés", round(time.time() - t0), "s", flush=True)

def cmd_emap(a):
    from wmloc.watermarks import load_watermark
    method = a.method; wm = load_watermark(method, models_dir=HERE / "kit" / "models", device="cuda"); rng = np.random.default_rng(7); trm, tem = metas(); t0 = time.time()
    def E_of(img, bits, p=12):
        img = np.ascontiguousarray(img); c = np.pad(img, ((p, p), (p, p), (0, 0)), mode="reflect"); r = wm.embed(np.ascontiguousarray(c), bits)[p:p + 512, p:p + 512].astype(np.float32) - img
        return np.clip(np.abs(r).mean(2).reshape(128, 4, 128, 4).mean((1, 3)) * 32, 0, 255).round().astype(np.uint8)
    for k, suf in enumerate(a.suffix.split(",")):
        D = ROOT / f"data_{method}{suf}"
        if k == 0 and not (D / "EALL_test.npy").exists():
            RT = np.load(CACHE / "REAL_train.npy", mmap_mode="r"); RE = np.load(CACHE / "REAL_test.npy", mmap_mode="r"); bits = lambda m: np.array([int(c) for c in m["message"]])
            np.save(D / "EALL_train.npy", np.stack([E_of(RT[i], bits(m)) for i, m in enumerate(trm) if m["method"] == method])); np.save(D / "EALL_test.npy", np.stack([E_of(RE[i], bits(m)) for i, m in enumerate(tem) if m["method"] == method]))
        if (D / "ER.npy").exists(): continue
        for src, dst in (("BASE", "EB"), ("WFULL", "EW"), ("RECT", "ER")):
            A = np.load(D / f"{src}.npy", mmap_mode="r"); np.save(D / f"{dst}.npy", np.stack([E_of(A[n], rng.integers(0, 2, wm.nbits)) for n in range(len(A))])); print(method, suf, dst, len(A), round(time.time() - t0), "s", flush=True)
    print(method, "emap ok", flush=True)

# ------------------------------------------------------------------ modèle
class FP32(nn.Module):                                   # première convolution en précision pleine : le filigrane vaut ~1 niveau sur 255
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, x):
        with torch.autocast("cuda", enabled=False): return s.m(x.float())

class DecNet(nn.Module):
    """tronc = décodeur du filigrane (entraîné par son auteur à lire le message, donc sensible au motif) + pyramide de caractéristiques -> carte 128x128.
    TrustMark : ResNet-50 en 245 px. PixelSeal : ConvNeXt en 256 px, calculé en précision pleine (signal de 0,8 niveau)."""
    def __init__(s, method, d=160):
        super().__init__()
        from wmloc.watermarks import load_watermark
        wm = load_watermark(method, models_dir=HERE / "kit" / "models", device="cpu"); s.method = method
        if method == "trustmark": s.res = wm.resolution; r = wm.net.float(); s.trunk = r; r.conv1 = FP32(r.conv1); ch = [256, 512, 1024, 2048]
        elif method == "pixelseal": s.res = 256; s.trunk = wm.det.float(); ch = [int(st[-1].dwconv.out_channels) if hasattr(st[-1], "dwconv") else c for st, c in zip(s.trunk.convnext.stages, [96, 192, 384, 768])]
        else: raise ValueError(method)
        for q in s.trunk.parameters(): q.requires_grad_(True)
        s.lat = nn.ModuleList([nn.Conv2d(c, d, 1) for c in ch]); s.smooth = nn.ModuleList([nn.Sequential(nn.Conv2d(d, d, 3, padding=1), nn.GroupNorm(8, d), nn.GELU()) for _ in ch])
        s.head = nn.Sequential(nn.Conv2d(d, d, 3, padding=1), nn.GELU(), nn.Conv2d(d, 1, 1))
    def train(s, mode=True):
        super().train(mode)
        if s.method == "pixelseal": s.trunk.eval()                     # pas de blocs sautés au hasard dans le tronc ; les gradients passent quand même
        for m in s.trunk.modules():
            if isinstance(m, nn.BatchNorm2d): m.eval()                 # statistiques de normalisation du décodeur gardées figées
        return s
    def feats(s, x01):
        h = F.interpolate(x01, size=(s.res, s.res), mode="bilinear", align_corners=False, antialias=True); fs = []
        if s.method == "trustmark":
            r = s.trunk; h = r.maxpool(r.relu(r.bn1(r.conv1(h * 2 - 1))))
            for k in ("layer1", "layer2", "layer3", "layer4"): h = getattr(r, k)(h); fs.append(h)
        else:
            with torch.autocast("cuda", enabled=False):
                h = s.trunk.preprocess(h.float())
                for dl, st in zip(s.trunk.convnext.downsample_layers, s.trunk.convnext.stages): h = st(dl(h)); fs.append(h)
        return fs
    def forward(s, x01):                                 # x01 : B x 3 x 512 x 512 dans [0, 1]
        fs = s.feats(x01); t = None
        for i in range(3, -1, -1):
            l = s.lat[i](fs[i]); t = l if t is None else l + F.interpolate(t, size=l.shape[-2:], mode="bilinear", align_corners=False); t = s.smooth[i](t)
        return F.interpolate(s.head(t), size=(128, 128), mode="bilinear", align_corners=False)
    def groups(s, lr): return [{"params": s.trunk.parameters(), "lr": lr / 10}, {"params": [q for m in (s.lat, s.smooth, s.head) for q in m.parameters()], "lr": lr}]

class StemSt(nn.Module):
    """branche pour signal faible : les canaux passe-haut (image - flou, à 3 échelles) traversent 2 convolutions 3x3 à pleine résolution, puis rejoignent la sortie
    de la première convolution pré-entraînée. La dernière convolution part de zéro : au départ le réseau est exactement le réseau pré-entraîné."""
    def __init__(s, conv1, nhp):
        super().__init__(); s.c1 = conv1; st = conv1.stride[0]
        s.b = nn.Sequential(nn.Conv2d(nhp, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Conv2d(32, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                            nn.Conv2d(32, conv1.out_channels, 3, st, 1, bias=False)); nn.init.zeros_(s.b[-1].weight)
    def forward(s, x):
        with torch.autocast("cuda", enabled=False):
            x = x.float(); return s.c1(x[:, :3]) + s.b(x[:, 3:])

def build_net(a, cin):
    if a.arch == "dec": return DecNet(a.method)
    import segmentation_models_pytorch as smp
    kw = dict(encoder_name=a.enc, encoder_weights="imagenet" if a.pretrained else None, in_channels=3 if a.st else cin, classes=1)
    if a.arch == "unet": net = smp.Unet(**kw)
    elif a.arch == "fpn": net = smp.FPN(**kw, upsampling=1)                 # sortie directe en 128x128
    elif a.arch == "upernet": net = smp.UPerNet(**kw, upsampling=1) if "upsampling" in smp.UPerNet.__init__.__code__.co_varnames else smp.UPerNet(**kw)
    elif a.arch == "segformer": net = smp.Segformer(**kw)
    elif a.arch == "unetpp": net = smp.UnetPlusPlus(**kw)
    else: raise ValueError(a.arch)
    if a.s1: net.encoder.conv1.stride = (1, 1)
    if a.s1 >= 2: net.encoder.maxpool = nn.Identity()             # aucune réduction non plus au deuxième étage : 16 fois le calcul de la tige d'origine
    if a.stem1:                                          # tige sans sous-échantillonnage : l'image passe par un dépliage 2x2 (sans perte), conv1 de pas 1
        c = net.encoder.conv1; n = nn.Conv2d(cin * 4, c.out_channels, 3, 1, 1, bias=False)
        nn.init.kaiming_normal_(n.weight, nonlinearity="relu"); net.encoder.conv1 = n
    if a.aux: assert a.arch == "unet"; net.auxhead = nn.Conv2d(16, 3, 3, padding=1)
    if a.st: net.encoder.conv1 = StemSt(net.encoder.conv1, cin - 3); return net
    for name, m in net.encoder.named_modules():          # première convolution en fp32
        if isinstance(m, nn.Conv2d):
            par = net.encoder; parts = name.split(".")
            for p in parts[:-1]: par = getattr(par, p)
            setattr(par, parts[-1], FP32(m)); break
    return net

class Runner:
    def __init__(s, a, dev="cuda"):
        s.a = a; s.dev = dev
        s.MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev)[None, :, None, None]; s.STD = torch.tensor([0.229, 0.224, 0.225], device=dev)[None, :, None, None]
        s.sigs = {0: (), 1: (3.0,), 2: (1.0, 2.0, 4.0)}[a.hp]; s.cin = 3 + 3 * len(s.sigs) + a.flag + a.echan; assert not (a.st and (a.flag or not a.hp)), '--st demande --hp et pas --flag'
        assert a.arch != "dec" or not (a.flag or a.hp), "--flag et --hp ne sont pas pris en charge avec --arch dec"
        s.net = build_net(a, s.cin).to(dev).to(memory_format=torch.channels_last)
        s.gs = []
        for sg in s.sigs:
            r = int(3 * sg + 1); g = torch.arange(-r, r + 1, device=dev).float(); g = torch.exp(-g ** 2 / (2 * sg ** 2)); s.gs.append((g / g.sum(), r))
    def prep(s, X, fl):                                  # X : B x 3 x 512 x 512, valeurs 0..255
        Ec = X[:, 3:4] / 64 if s.a.echan else None; X = X[:, :3]      # 4e canal éventuel : force attendue du filigrane (x32 en entrée, ici ramenée à niveaux / 2)
        ch = [(X / 255 - s.MEAN) / s.STD]
        for g, r in s.gs:                                # passe-haut : image moins sa version floutée, bornée à +-8 niveaux, 1 niveau = 0,25
            b = F.conv2d(F.pad(X, (r, r, 0, 0), mode="reflect"), g.view(1, 1, 1, -1).repeat(3, 1, 1, 1), groups=3)
            b = F.conv2d(F.pad(b, (0, 0, r, r), mode="reflect"), g.view(1, 1, -1, 1).repeat(3, 1, 1, 1), groups=3); ch.append((X - b).clamp(-8, 8) / 4)
        if Ec is not None: ch.append(Ec)
        if s.a.flag: ch.append((fl.view(-1, 1, 1, 1) * 2 - 1).expand(-1, 1, X.shape[2], X.shape[3]))
        x = torch.cat(ch, 1)
        if s.a.stem1: x = F.pixel_unshuffle(x, 2)
        return x.contiguous(memory_format=torch.channels_last)
    def fwd(s, net, X, fl):
        x = X / 255 if s.a.arch == "dec" else s.prep(X, fl)
        with torch.autocast("cuda", dtype=torch.float16, enabled=s.a.prec == "fp16"):
            if hasattr(net, "auxhead"): dec = net.decoder(net.encoder(x)); o = net.segmentation_head(dec); s.last_aux = net.auxhead(dec).float()
            else: o = net(x)
        o = o.float(); s.last_full = o[:, 0]
        if o.shape[-1] < 128: o = F.interpolate(o, size=(128, 128), mode="bilinear", align_corners=False)
        k = o.shape[-1] // 128
        return (F.avg_pool2d(o, k) if k > 1 else o)[:, 0]
    @torch.no_grad()
    def predict(s, net, imgs, flags, tta=2, bs=16):      # imgs uint8 N x 512 x 512 x 3 -> logits N x 128 x 128 (moyenne sur les symétries)
        net.eval(); out = []; ks = {1: [0], 2: [0, 1], 4: [0, 1, 2, 3], 8: list(range(8))}[tta]
        for j in range(0, len(imgs), bs):
            X = torch.from_numpy(np.ascontiguousarray(imgs[j:j + bs])).to(s.dev).permute(0, 3, 1, 2).float(); fl = torch.from_numpy(np.asarray(flags[j:j + bs], np.float32)).to(s.dev)
            lg = sum(dih_inv(s.fwd(net, dih_t(X, k), fl), k) for k in ks) / len(ks); out.append(lg.cpu().numpy())
        return np.concatenate(out)

def _lov_grad(gs):
    g = gs.sum(); inter = g - gs.cumsum(0); union = g + (1 - gs).cumsum(0); j = 1 - inter / union
    j[1:] = j[1:] - j[:-1].clone(); return j
def _lov_flat(lg, y):
    err = 1 - lg * (2 * y - 1); es, perm = torch.sort(err, descending=True); return torch.dot(F.relu(es), _lov_grad(y[perm]))
def lovasz2(lg, y):                                      # moyenne par image de la perte IoU des deux classes
    return torch.stack([0.5 * (_lov_flat(l.flatten(), t.flatten()) + _lov_flat(-l.flatten(), 1 - t.flatten())) for l, t in zip(lg, y)]).mean()

# ------------------------------------------------------------------ train : un pli
def cmd_train(a):
    torch.set_num_threads(NT); dev = "cuda"; method = a.method; rng = np.random.default_rng(a.seed * 100 + a.fold + 7); torch.manual_seed(a.seed * 100 + a.fold)
    if a.prec == "fp32": torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    OUT = ROOT / "runs" / a.tag / method / f"fold{a.fold}"; OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "done").exists(): print("déjà fait", OUT); return
    def log(**k): print(a.tag, method, a.fold, json.dumps(k), flush=True); open(OUT / "log.jsonl", "a").write(json.dumps(k) + "\n")
    trm, tem = metas(); SUF = a.suffix.split(","); D = ROOT / f"data_{method}{SUF[0]}"; t0 = time.time()
    RT = np.load(CACHE / "REAL_train.npy", mmap_mode="r"); MT = np.load(CACHE / "MASK_train.npy", mmap_mode="r"); Y128 = np.load(CACHE / "Y128_train.npy"); RE = np.load(CACHE / "REAL_test.npy", mmap_mode="r")
    idx, res, rest, fold = splits(trm, method); pos = {i: p for p, i in enumerate(idx)}
    AALL = np.load(D / "AALL_train.npy"); AALLT = np.load(D / "AALL_test.npy"); own_te = [i for i, m in enumerate(tem) if m["method"] == method]
    tr_i = [i for i in rest if fold[i] != a.fold] if a.fold >= 0 else []; va_i = [i for i in rest if fold[i] == a.fold] if a.fold >= 0 else list(rest)
    if a.fold < 0: assert a.nreal == 0 and a.nmix == 0, 'pré-entraînement = fabriqué seul'
    flag_of = lambda i: float(AALL[pos[i]] > 0.75)
    if a.realpop == "full": tr_i = [i for i in tr_i if trm[i]["full_watermark"]]
    isfull = lambda i: bool(trm[i]["full_watermark"])
    # ingrédients fabriqués
    ns = a.bs - a.nreal - a.nmix - a.nal - a.npl
    kinds_ok = {"all": (0, 1, 2), "notest": (0, 2), "coco": (2,), "hack": (0, 1)}[a.bases]
    POOLNAMES = {}; RSRC_ = {}
    def load_pool(Dk):
        BINFO = np.load(Dk / "BINFO.npy"); BSEL = np.where(np.isin(BINFO[:, 0], kinds_ok))[0]; HBSEL = np.where(np.isin(BINFO[:, 0], [k for k in kinds_ok if k != 2] or [2]))[0]
        BASE = np.load(Dk / "BASE.npy", mmap_mode="r"); WFULL = np.load(Dk / "WFULL.npy", mmap_mode="r"); RECT = np.load(Dk / "RECT.npy", mmap_mode="r")
        if ns > 0 and not a.lazy: BASE = np.ascontiguousarray(BASE); WFULL = np.ascontiguousarray(WFULL); RECT = np.ascontiguousarray(RECT)
        RSRC_[id(BASE)] = np.load(Dk / "RSRC.npy"); RBOX = np.load(Dk / "RBOX.npy"); RSEL = np.where(np.isin(BINFO[np.load(Dk / "RSRC.npy"), 0], kinds_ok))[0]; RSMALL = RSEL[(RBOX[RSEL, 2] * RBOX[RSEL, 3]) < 0.03 * 512 * 512]
        EB, EW, ER = [np.load(Dk / f"{n}.npy") if a.echan else None for n in ("EB", "EW", "ER")]
        POOLNAMES[id(BASE)] = [((trm if kd == 0 else tem)[ix]["image_name"], int(kk)) if kd < 2 else (None, 0) for kd, ix, kk in BINFO]
        return BASE, WFULL, RECT, RBOX, BSEL, RSEL, HBSEL, RSMALL, EB, EW, ER
    POOLS = [load_pool(ROOT / f"data_{method}{k}") for k in SUF]; BASE, WFULL, RECT, RBOX, BSEL, RSEL, HBSEL, RSMALL, EB, EW, ER = POOLS[0]
    assert not (a.echan and (a.nal or a.arch == "dec")), "--echan : pas avec --nal ni --arch dec"
    up4 = lambda e: np.repeat(np.repeat(e, 4, -2), 4, -1)
    if a.echan: EAL = np.load(D / "EALL_train.npy"); EALT = np.load(D / "EALL_test.npy")
    W4 = lambda imgs, L, src: np.concatenate([np.ascontiguousarray(imgs), up4(src[L])[..., None]], -1) if a.echan else np.ascontiguousarray(imgs)   # image + 4e canal
    tup4 = lambda e: e.repeat_interleave(4, -2).repeat_interleave(4, -1)
    if a.nal: AL = np.load(D / "AL.npy"); ALY = np.load(D / "ALY.npy"); ALF = np.load(D / "ALF.npy"); ALI = np.load(D / "ALI.npy")
    # vraies images du filigrane (plis d'entraînement) sur GPU
    G = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(dev)
    RXg = G(W4(RT[tr_i], [pos[i] for i in tr_i], EAL if a.echan else None)).permute(0, 3, 1, 2); RMg = G(MT[tr_i]).bool(); RFL = [flag_of(i) for i in tr_i]
    if a.npl:   # images de test du filigrane, étiquetées par la prédiction du modèle du même pli (aucune étiquette du pli tenu à l'écart ni de la réserve n'entre)
        assert a.fold >= 0 and not a.echan; zt = np.load(ROOT / 'runs' / a.pl / method / f'fold{a.fold}' / 'test.npz'); PLX = G(RE[own_te]).permute(0, 3, 1, 2); PLM = tup4(G(zt['logits'].astype(np.float32) > 0)); assert len(PLX) == len(PLM), (len(PLX), len(PLM))
        print('pseudo-étiquettes', a.pl, len(PLX), 'part filigranée', round(float(PLM.float().mean()), 3), flush=True)
    # banque de formes : trous (zone propre des pleines) et objets (partielles non rectangulaires), hors réserve et hors pli tenu à l'écart
    holes, objs = [], []
    for i, m in enumerate(trm):
        if i % 5 == 0 or (m["method"] == method and (a.fold < 0 or fold[i] == a.fold)): continue
        M = np.array(MT[i]).astype(bool)
        if m["full_watermark"]: holes.append(~M)
        else:
            ys, xs = np.where(M)
            if M.sum() < 0.98 * (ys.max() + 1 - ys.min()) * (xs.max() + 1 - xs.min()): objs.append(M)
    bank = {"hole": G(np.stack(holes)).bool(), "obj": G(np.stack(objs)).bool()}
    log(info="données", real_train=len(tr_i), val=len(va_i), reserves=len(POOLS), bases=sum(len(q[4]) for q in POOLS), rect=sum(len(q[5]) for q in POOLS), trous=len(holes), objets=len(objs), sec=round(time.time() - t0))

    def blob(lo, hi):                                    # tache aléatoire lisse d'aire tirée entre lo et hi (part de l'image)
        area = float(np.exp(rng.uniform(np.log(lo), np.log(hi)))); g = int(rng.integers(3, 11))
        up = F.interpolate(torch.randn(1, 1, g, g, device=dev), size=(512, 512), mode="bicubic", align_corners=False)[0, 0]
        return up > torch.quantile(up.flatten(), 1 - area)
    def small_hole():                                    # petit trou : vraie forme de trou ramenée à une aire tirée entre hlo et hhi (part de l'image), posée n'importe où ; 30 % de taches lisses
        t = float(np.exp(rng.uniform(np.log(a.hlo), np.log(a.hhi))))
        if rng.random() < 0.3: return blob(t, t * 1.0001)
        s0 = dih_t(bank["hole"][int(rng.integers(len(bank["hole"])))], int(rng.integers(8))); ar = float(s0.float().mean())
        if ar <= 0: return blob(t, t * 1.0001)
        ys, xs = torch.where(s0); crop = s0[int(ys.min()):int(ys.max()) + 1, int(xs.min()):int(xs.max()) + 1].float(); f = (t / ar) ** 0.5
        h = max(4, min(512, int(round(crop.shape[0] * f)))); w = max(4, min(512, int(round(crop.shape[1] * f))))
        c = F.interpolate(crop[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0] > 0.5
        out = torch.zeros(512, 512, dtype=torch.bool, device=dev); y0 = int(rng.integers(0, 512 - h + 1)); x0 = int(rng.integers(0, 512 - w + 1)); out[y0:y0 + h, x0:x0 + w] = c
        return out if out.sum() > 200 else blob(t, t * 1.0001)
    def rand_shape(kind):
        if rng.random() < a.pblob: return blob(0.004, 0.5 if kind == "hole" else 0.45)
        s0 = dih_t(bank[kind][int(rng.integers(len(bank[kind])))], int(rng.integers(8)))
        if rng.random() >= a.pshift: return s0                # forme réelle gardée à sa place (les vrais trous touchent souvent 3 ou 4 côtés ; le décalage détruit cela)
        dy, dx = int(rng.integers(-160, 161)), int(rng.integers(-160, 161)); s = torch.roll(s0, (dy, dx), (0, 1)).clone()
        if dy > 0: s[:dy] = False
        elif dy < 0: s[dy:] = False
        if dx > 0: s[:, :dx] = False
        elif dx < 0: s[:, dx:] = False
        return s if s.sum() > 600 else s0
    def paste(clean, wmk, m):                            # m vrai = filigrané ; 30 % de bords adoucis (1 ou 2 px)
        if rng.random() < a.psoft:
            k = int(rng.integers(1, 3)); mf = F.avg_pool2d(m.float()[None, None], 2 * k + 1, 1, k)[0]
            return (clean * (1 - mf) + wmk * mf).round()
        return torch.where(m[None], wmk, clean)
    SAM = {}                                             # greffe d'objet réaliste : masques d'objets (SAM) des images qui servent de fonds
    if a.objsam:
        for fz in sorted(Path(a.objsam).glob("sam_t*_*.npz")):
            if fz.stem.count("_") != 2: continue
            z = np.load(fz)
            for kz in z.files:
                if not kz.endswith("__n"): SAM[kz] = z[kz]
        log(info="masques SAM", images=len(SAM))
    def sam_object(name, k):                             # un objet de l'image source (0,3 % à 45 % de l'image), dans l'orientation du fond, en 512 x 512
        if name not in SAM: return None
        mk = np.unpackbits(SAM[name].reshape(-1, 2048), axis=1).reshape(-1, 128, 128); ar = mk.mean((1, 2)); ok = np.where((ar > 0.003) & (ar < 0.45))[0]
        if len(ok) == 0: return None
        m = dih_t(G(mk[int(rng.choice(ok))]).float(), k); return F.interpolate(m[None, None], size=(512, 512), mode="bilinear", align_corners=False)[0, 0] > 0.5
    def moved(t, dy, dx):                                # déplace un tenseur (..., H, W) de (dy, dx), le vide laissé vaut 0
        t = torch.roll(t, (dy, dx), (-2, -1)).clone()
        if dy > 0: t[..., :dy, :] = 0
        elif dy < 0: t[..., dy:, :] = 0
        if dx > 0: t[..., :, :dx] = 0
        elif dx < 0: t[..., :, dx:] = 0
        return t
    Ms = []                                              # masques 512 x 512 du lot en cours (pour --fullsup)
    Rs = []                                              # résidu vrai par exemple fabriqué (3 x 512 x 512, en niveaux), None pour une vraie image
    def to128(m): Ms.append(m.float()); return (F.avg_pool2d(m.float()[None, None], 4)[0, 0] >= 0.5).float()

    def make_batch():
        Ms.clear(); Rs.clear(); Xs, Ys, FL = [], [], []; BASE, WFULL, RECT, RBOX, BSEL, RSEL, HBSEL, RSMALL, EB, EW, ER = POOLS[int(rng.integers(len(POOLS)))]
        if ns > 0:
            u = rng.random(ns); nfull = int((u < a.pfull).sum()); nrect = int(((u >= a.pfull) & (u < a.pfull + (1 - a.pfull) / 2)).sum()); nobj = ns - nfull - nrect
            bi = np.sort(rng.choice(BSEL, nfull + nobj)); base = G(BASE[bi]).permute(0, 3, 1, 2).float(); wf = G(WFULL[bi]).permute(0, 3, 1, 2).float()
            pm = torch.from_numpy(rng.permutation(len(bi))).to(dev); base, wf = base[pm], wf[pm]
            if a.echan: eb = G(EB[bi]).float()[pm]; ew = G(EW[bi]).float()[pm]
            bip = bi[pm.cpu().numpy()]; names = POOLNAMES[id(BASE)]
            for j in range(nfull + nobj):
                full = j < nfull; s_ = small_hole() if (a.hsmall > 0 and full and rng.random() < a.hsmall) else blob(0.003, 0.03) if rng.random() < a.psmall else rand_shape("hole" if full else "obj"); m = ~s_ if full else s_
                xj = None; rj = None
                if full and a.holesam and rng.random() < a.holesam:   # trou aligné sur le contenu : une ou deux régions (masques SAM) de l'image elle-même, rendues propres
                    mo = sam_object(*names[int(bip[j])])
                    if mo is not None:
                        if rng.random() < 0.3:
                            m2 = sam_object(*names[int(bip[j])]); mo = mo | m2 if m2 is not None else mo
                        if 0.002 < float(mo.float().mean()) < 0.5: m = ~mo
                if not full and a.objsam:                # greffe : un vrai objet d'une image filigranée sur SA toile, collé ailleurs (déplacement médian ~40 px, jusqu'à 130) sur une AUTRE image propre
                    mo = sam_object(*names[int(bip[j])])
                    if mo is not None:
                        dy, dx = (0, 0) if rng.random() < 0.25 else (int(np.clip(rng.normal(0, 60), -130, 130)), int(np.clip(rng.normal(0, 60), -130, 130))); mm = moved(mo, dy, dx)
                        if mm.sum() < 300: mm, dy, dx = mo, 0, 0
                        m = mm; xj = paste(base[(j + 1) % (nfull + nobj)], moved(wf[j], dy, dx), m); rj = torch.where(m[None], moved(wf[j] - base[j], dy, dx), torch.zeros_like(wf[j]))
                if xj is None: xj = paste(base[j], wf[j], m)
                Rs.append(rj if rj is not None else torch.where(m[None], wf[j] - base[j], torch.zeros_like(wf[j])))
                if a.echan: xj = torch.cat([xj, tup4(torch.where(to128(m) > 0, ew[j], eb[j]))[None]])
                Xs.append(xj); Ys.append(to128(m)); FL.append(float(full) if rng.random() > 0.02 else 1 - float(full))
            if nrect:
                ri = np.sort(np.array([rng.choice(RSMALL if (rng.random() < a.psmall and len(RSMALL)) else RSEL) for _ in range(nrect)])); rx = G(RECT[ri]).permute(0, 3, 1, 2).float()
                if a.echan: rx = torch.cat([rx, tup4(G(ER[ri]).float())[:, None]], 1)
                for j, r in enumerate(ri):
                    y0, x0, h, w = RBOX[r]; m = torch.zeros(512, 512, dtype=torch.bool, device=dev); m[y0:y0 + h, x0:x0 + w] = True
                    Rs.append(rx[j][:3] - G(BASE[int(RSRC_[id(BASE)][r])]).permute(2, 0, 1).float() if a.aux else None)
                    Xs.append(rx[j]); Ys.append(to128(m)); FL.append(0.0 if rng.random() > 0.02 else 1.0)
        for j in rng.integers(0, len(tr_i), a.nreal):   # vraies images, 8 symétries
            k = int(rng.integers(8)) if a.raug else 0
            Xs.append(dih_t(RXg[j].float(), k)); Ys.append(to128(dih_t(RMg[j], k))); FL.append(RFL[j])
        for j in (rng.integers(0, len(PLX), a.npl) if a.npl else []):   # images de test pseudo-étiquetées
            k = int(rng.integers(8)) if a.raug else 0
            Xs.append(dih_t(PLX[j].float(), k)); Ys.append(to128(dih_t(PLM[j], k))); FL.append(0.0)
        for _ in range(a.nmix):                          # mélange de deux vraies images selon une tache : les pixels restent vrais, les positions aussi
            ja = int(rng.integers(len(tr_i))); S = blob(0.03, 0.6)
            if rng.random() < 0.5: jb = int(rng.integers(len(tr_i))); xb = RXg[jb].float(); mb = RMg[jb]
            else:
                ib = int(rng.choice(HBSEL)); xb = G(BASE[ib]).permute(2, 0, 1).float(); mb = torch.zeros(512, 512, dtype=torch.bool, device=dev)
                if a.echan: xb = torch.cat([xb, tup4(G(EB[ib]).float())[None]])
            x = torch.where(S[None], xb, RXg[ja].float()); m = torch.where(S, mb, RMg[ja]); k = int(rng.integers(8)) if a.raug else 0
            Xs.append(dih_t(x, k)); Ys.append(to128(dih_t(m, k))); FL.append(float(m.float().mean() > 0.5))
        for j in (rng.integers(0, len(AL), a.nal) if a.nal else []):
            if rng.random() < 0.5: Xs.append(G(RT[int(ALI[j])]).permute(2, 0, 1).float()); Ys.append(torch.zeros(128, 128, device=dev)); FL.append(0.0)
            else: Xs.append(G(AL[j]).permute(2, 0, 1).float()); Ys.append(G(ALY[j]).float()); FL.append(float(ALF[j]))
        while len(Rs) < len(Xs): Rs.append(None)
        return torch.stack(Xs), torch.stack(Ys), torch.tensor(FL, device=dev)

    R = Runner(a, dev); net = R.net
    FV = ROOT / f"fabval_{method}{SUF[0]}.npz"
    if FV.exists(): z = np.load(FV); SX, SY, SF = z["X"], z["Y"], z["F"]
    else:
        rng_main = rng; rng = np.random.default_rng(4242); keep = (a.nreal, a.nmix, a.nal, ns, a.pfull, a.psmall, a.pshift, a.pblob, a.objsam, a.holesam, a.aux)
        a.nreal = a.nmix = a.nal = 0; ns = a.bs; a.pfull, a.psmall, a.pshift, a.pblob, a.objsam, a.holesam, a.aux = 0.5, 0.0, 0.15, 0.15, "", 0.0, 0; SX, SY, SF = [], [], []
        for _ in range(max(1, 256 // a.bs)):
            Xb, Yb, Fb = make_batch(); SX.append(Xb[:, :3].round().byte().permute(0, 2, 3, 1).cpu().numpy()); SY.append(Yb.cpu().numpy()); SF.append(Fb.cpu().numpy())
        SX, SY, SF = np.concatenate(SX), np.concatenate(SY), np.concatenate(SF); a.nreal, a.nmix, a.nal, ns, a.pfull, a.psmall, a.pshift, a.pblob, a.objsam, a.holesam, a.aux = keep; rng = rng_main
        np.savez(str(FV) + f".{os.getpid()}.tmp.npz", X=SX, Y=SY, F=SF); os.replace(str(FV) + f".{os.getpid()}.tmp.npz", FV)
    if a.init:
        PI = ROOT / 'runs' / a.init / method / 'fold-1'; sd = torch.load(PI / 'ema.pt', map_location=dev)['state'] if (PI / 'ema.pt').exists() else torch.load(PI / 'ckpt.pt', map_location=dev)['ema']   # à défaut de poids finaux : dernier point de contrôle
        print('init', a.init, net.load_state_dict(sd), flush=True)
    ema = copy.deepcopy(net)
    for p in ema.parameters(): p.requires_grad_(False)
    pg = net.groups(a.lr) if hasattr(net, 'groups') else [{'params': net.parameters(), 'lr': a.lr}]
    opt = torch.optim.AdamW(pg, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g['lr'] for g in pg], total_steps=a.steps, pct_start=a.warm); scaler = torch.amp.GradScaler("cuda", enabled=a.prec == "fp16")
    RXr = W4(RT[res], [pos[i] for i in res], EAL if a.echan else None); RFr = [flag_of(i) for i in res]; rfull = np.array([isfull(i) for i in res])
    VX = W4(RT[va_i], [pos[i] for i in va_i], EAL if a.echan else None); VY = Y128[va_i]; VF = [flag_of(i) for i in va_i]; vfull = np.array([isfull(i) for i in va_i])
    def hole(P, Gt, sel):                                # IoU moyen du trou sur les images sélectionnées
        v = [1.0 if ((~p) | (~g)).sum() == 0 else ((~p) & (~g)).sum() / ((~p) | (~g)).sum() for p, g in zip(P[sel], Gt[sel].astype(bool))]; return round(float(np.mean(v)), 4) if len(v) else -1
    def evaluate(step, tot):
        lg = R.predict(ema, VX, VF, tta=1); sc = miou_list(lg > 0, VY); lgV = lg
        ths = np.linspace(-4, 4, 17); best = max((float(miou_list(lg > t, VY).mean()), float(t)) for t in ths)
        lR = R.predict(ema, RXr, RFr, tta=1); lS = R.predict(ema, SX, SF, tta=1); sr = miou_list(lR > 0, Y128[res]); sf = miou_list(lS > 0, SY); sfu = SY.mean((1, 2)) > 0.5
        log(step=step, loss=round(tot, 4), trou_fab=hole(lS > 0, SY > 0.5, sfu), trou_res=hole(lR > 0, Y128[res], rfull), trou_val=hole(lgV > 0, VY, vfull), fab=round(float(sf.mean()), 4), fab_pleins=round(float(sf[sfu].mean()), 4), fab_partiels=round(float(sf[~sfu].mean()), 4), reserve=round(float(sr.mean()), 4), res_pleins=round(float(sr[rfull].mean()), 4), res_partiels=round(float(sr[~rfull].mean()), 4), val=round(float(sc.mean()), 4), pleins=round(float(sc[vfull].mean()), 4), partiels=round(float(sc[~vfull].mean()), 4),
            auc=round(auc(lg, VY), 4), val_seuil_oracle=round(best[0], 4), seuil=best[1], sec=round(time.time() - t0))
    step0 = 0; CK = OUT / "ckpt.pt"
    if CK.exists():
        ck = torch.load(CK, map_location=dev); net.load_state_dict(ck["net"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"]); step0 = ck["step"]; rng = np.random.default_rng(a.seed * 100 + a.fold + 7 + 7919 * step0); torch.manual_seed(a.seed * 100 + a.fold + 7919 * step0); log(info="reprise", step=step0)
    net.train(); tot = 0; n = 0
    for step in range(step0 + 1, a.steps + 1):
        X, Y, FLt = make_batch()
        if a.dump and step == step0 + 1:               # contrôle visuel du premier lot : image, puis masque (blanc = filigrané)
            im = X[:, :3].round().byte().permute(0, 2, 3, 1).cpu().numpy(); mk = (F.interpolate(Y[:, None], scale_factor=4)[:, 0] * 255).byte().cpu().numpy()
            Image.fromarray(np.concatenate([np.concatenate([np.concatenate([im[j], np.repeat(mk[j][..., None], 3, 2)], 1) for j in range(r, min(r + 4, len(im)))], 1) for r in range(0, len(im), 4)], 0)[::2, ::2]).save(a.dump)
        lg = R.fwd(net, X, FLt)
        p = torch.sigmoid(lg); i1 = (p * Y).sum((1, 2)); i0 = ((1 - p) * (1 - Y)).sum((1, 2))
        dice = 1 - 0.5 * ((2 * i1 + 1) / (p.sum((1, 2)) + Y.sum((1, 2)) + 1) + (2 * i0 + 1) / ((1 - p).sum((1, 2)) + (1 - Y).sum((1, 2)) + 1))
        loss = F.binary_cross_entropy_with_logits(lg, Y) + dice.mean()
        if a.lovasz: loss = loss + a.lovasz * lovasz2(lg, Y)
        if a.aux:                                        # erreur absolue moyenne entre résidu prédit et résidu vrai (en demi-niveaux), sur les exemples fabriqués seulement
            wk = torch.tensor([r is not None for r in Rs], device=dev).float(); T = torch.stack([r if r is not None else torch.zeros(3, 512, 512, device=dev) for r in Rs]) / 2
            loss = loss + a.aux * (((R.last_aux - T).abs().mean((1, 2, 3)) * wk).sum() / wk.sum().clamp_min(1))
        if a.fullsup:
            M5 = torch.stack(Ms); lf = R.last_full; assert lf.shape == M5.shape, (lf.shape, M5.shape)
            loss = loss + a.fullsup * F.binary_cross_entropy_with_logits(lf, M5)
        if a.wclean:
            fu = Y.mean((1, 2)) > 0.5
            if fu.any(): loss = loss + a.wclean * torch.stack([_lov_flat(-l.flatten(), 1 - t.flatten()) for l, t in zip(lg[fu], Y[fu])]).mean()
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step(); tot += float(loss); n += 1
        with torch.no_grad():
            d = min(a.ema, (1 + step) / (10 + step))
            for pe, pn in zip(ema.parameters(), net.parameters()): pe.lerp_(pn, 1 - d)
            for be, bn in zip(ema.buffers(), net.buffers()):
                if be.dtype.is_floating_point: be.lerp_(bn, 1 - d)   # statistiques de normalisation moyennées elles aussi
                else: be.copy_(bn)
        if step % a.evalevery == 0 or step == a.steps: evaluate(step, tot / n); tot = 0; n = 0; net.train()
        if a.maxmin and time.time() - t0 > a.maxmin * 60: log(info="arrêt au temps", step=step); break
        if a.ckevery and step % a.ckevery == 0 and step < a.steps:
            torch.save({"net": net.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(), "step": step}, str(CK) + ".tmp"); os.replace(str(CK) + ".tmp", CK)
    # prédictions finales (poids moyennés, moyenne sur les symétries) : pli tenu à l'écart, réserve, test
    names = lambda L, M: np.array([M[i]["image_name"] for i in L])
    np.savez_compressed(OUT / "oof.npz", names=names(va_i, trm), idx=np.array(va_i), logits=R.predict(ema, VX, VF, tta=a.tta).astype(np.float16), logits1=R.predict(ema, VX, VF, tta=1).astype(np.float16))
    np.savez_compressed(OUT / "res.npz", names=names(res, trm), idx=np.array(res), logits=R.predict(ema, RXr, RFr, tta=a.tta).astype(np.float16), logits1=R.predict(ema, RXr, RFr, tta=1).astype(np.float16))
    if a.test: np.savez_compressed(OUT / "test.npz", names=names(own_te, tem), logits=R.predict(ema, W4(RE[own_te], np.arange(len(own_te)), EALT if a.echan else None), (AALLT > 0.75).astype(np.float32), tta=a.tta).astype(np.float16))
    if a.save: torch.save({"state": ema.state_dict(), "args": vars(a)}, OUT / "ema.pt")
    CK.unlink(missing_ok=True); (OUT / "args.json").write_text(json.dumps(vars(a))); (OUT / "done").write_text("ok"); log(info="fini", sec=round(time.time() - t0), mem_mo=int(torch.cuda.max_memory_allocated() // 2 ** 20), conv1=type(net.encoder.conv1).__name__ if hasattr(net, "encoder") and hasattr(net.encoder, "conv1") else "?")

# ------------------------------------------------------------------ agg : score en validation croisée et sur la réserve
def cmd_agg(a):
    trm, tem = metas(); Y128 = np.load(CACHE / "Y128_train.npy"); rows = []
    tags = a.tag.split(",") if a.tag != "all" else sorted(p.name for p in (ROOT / "runs").iterdir() if "+" not in p.name)
    for tag in tags:
        for method in METHODS:
            subs = tag.split("+"); Ps = [ROOT / "runs" / t / method for t in subs]; P = ROOT / "runs" / tag / method
            fs = [f for f in range(5) if all((q / f"fold{f}" / "done").exists() for q in Ps)]
            if not fs: continue
            oi, ol, rl, ol1 = [], [], [], []; ld = lambda f, n, k: np.mean([np.load(q / f"fold{f}" / n)[k].astype(np.float32) for q in Ps], 0)
            for f in fs:
                oi += list(np.load(Ps[0] / f"fold{f}" / "oof.npz")["idx"]); ol.append(ld(f, "oof.npz", "logits")); ol1.append(ld(f, "oof.npz", "logits1"))
                ri = np.load(Ps[0] / f"fold{f}" / "res.npz")["idx"]; rl.append(ld(f, "res.npz", "logits"))
            oi = np.array(oi); ol = np.concatenate(ol); ol1 = np.concatenate(ol1); rl = np.mean(rl, 0)
            full = lambda L: np.array([bool(trm[i]["full_watermark"]) for i in L])
            ths = np.linspace(-4, 4, 33); th = float(ths[int(np.argmax([miou_list(ol > t, Y128[oi]).mean() for t in ths]))])
            so = miou_list(ol > 0, Y128[oi]); sr = miou_list(rl > 0, Y128[ri]); srt = miou_list(rl > th, Y128[ri]); fo, fr = full(oi), full(ri)
            r = dict(tag=tag, methode=method, plis=len(fs), cv_sans_tta=miou_list(ol1 > 0, Y128[oi]).mean(), cv=so.mean(), cv_pleines=so[fo].mean(), cv_partielles=so[~fo].mean(), cv_auc=auc(ol, Y128[oi]),
                     seuil_cv=th, cv_au_seuil=miou_list(ol > th, Y128[oi]).mean(), reserve=sr.mean(), res_pleines=sr[fr].mean(), res_partielles=sr[~fr].mean(), reserve_au_seuil_cv=srt.mean())
            rows.append(r)
            if a.write and len(fs) == 5:
                P.mkdir(parents=True, exist_ok=True); idx, res, rest, fold = splits(trm, method); lg = {int(i): l for i, l in zip(oi, ol)}; lg.update({int(i): l for i, l in zip(ri, rl)})
                np.savez_compressed(P / f"pred_train_{method}.npz", names=np.array([trm[i]["image_name"] for i in idx]), logits=np.stack([lg[i] for i in idx]).astype(np.float16))
                tl = [np.load(q / f"fold{f}" / "test.npz") for f in fs for q in Ps if (q / f"fold{f}" / "test.npz").exists()]
                if tl: np.savez_compressed(P / f"pred_test_{method}.npz", names=tl[0]["names"], logits=np.mean([t["logits"].astype(np.float32) for t in tl], 0).astype(np.float16))
    cols = ["tag", "methode", "plis", "cv_sans_tta", "cv", "cv_pleines", "cv_partielles", "cv_auc", "seuil_cv", "cv_au_seuil", "reserve", "res_pleines", "res_partielles", "reserve_au_seuil_cv"]
    print("| " + " | ".join(cols) + " |"); print("|" + "---|" * len(cols))
    for r in sorted(rows, key=lambda r: (r["methode"], r["tag"])): print("| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--method"); ap.add_argument("--suffix", default="")
    ap.add_argument("--coco", default=str(HERE.parent / "shared_data" / "coco" / "val2017"))
    ap.add_argument("--nvar", type=int, default=2); ap.add_argument("--ncoco", type=int, default=1200); ap.add_argument("--nrect", type=int, default=3000); ap.add_argument("--nalign", type=int, default=2); ap.add_argument("--small", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0); ap.add_argument("--tag", default="base"); ap.add_argument("--steps", type=int, default=2500); ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--hsmall", type=float, default=0.0); ap.add_argument("--hlo", type=float, default=0.005); ap.add_argument("--hhi", type=float, default=0.04)
    ap.add_argument("--npl", type=int, default=0); ap.add_argument("--pl", default="");
    ap.add_argument("--nreal", type=int, default=6); ap.add_argument("--nmix", type=int, default=0); ap.add_argument("--nal", type=int, default=0); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--enc", default="resnet34"); ap.add_argument("--arch", default="unet"); ap.add_argument("--pretrained", type=int, default=1); ap.add_argument("--stem1", type=int, default=0)
    ap.add_argument("--flag", type=int, default=0); ap.add_argument("--hp", type=int, default=0); ap.add_argument("--prec", default="fp16"); ap.add_argument("--raug", type=int, default=1)
    ap.add_argument("--bases", default="all"); ap.add_argument("--pblob", type=float, default=0.3); ap.add_argument("--psoft", type=float, default=0.3); ap.add_argument("--ema", type=float, default=0.995)
    ap.add_argument("--tta", type=int, default=2); ap.add_argument("--save", type=int, default=1); ap.add_argument("--test", type=int, default=1); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--evalevery", type=int, default=250); ap.add_argument("--lazy", type=int, default=0); ap.add_argument("--write", type=int, default=0)
    ap.add_argument("--aux", type=float, default=0.0); ap.add_argument("--fullsup", type=float, default=0.0); ap.add_argument("--pfull", type=float, default=0.5); ap.add_argument("--pshift", type=float, default=1.0); ap.add_argument("--realpop", default="all"); ap.add_argument("--wclean", type=float, default=0.0); ap.add_argument("--echan", type=int, default=0); ap.add_argument("--st", type=int, default=0); ap.add_argument("--canvas", default="fixe"); ap.add_argument("--dump", default=""); ap.add_argument("--smax", type=int, default=70); ap.add_argument("--objsam", default=""); ap.add_argument("--holesam", type=float, default=0.0); ap.add_argument("--gseed", type=int, default=0); ap.add_argument("--maxmin", type=float, default=0); ap.add_argument("--s1", type=int, default=0); ap.add_argument("--psmall", type=float, default=0.0); ap.add_argument("--padlo", type=int, default=0); ap.add_argument("--padhi", type=int, default=0); ap.add_argument("--rfraclo", type=float, default=0.0); ap.add_argument("--rfrachi", type=float, default=0.0); ap.add_argument("--rpadlo", type=int, default=0); ap.add_argument("--rpadhi", type=int, default=0)
    ap.add_argument("--ckevery", type=int, default=0); ap.add_argument("--init", default=""); ap.add_argument("--lovasz", type=float, default=0.0); ap.add_argument("--warm", type=float, default=0.08)
    a = ap.parse_args(); {"cache": cmd_cache, "gen": cmd_gen, "emap": cmd_emap, "train": cmd_train, "agg": cmd_agg}[a.cmd](a)
