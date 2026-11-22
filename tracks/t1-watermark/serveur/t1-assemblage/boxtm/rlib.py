"""Bibliothèque commune de la session t1-residus : gabarits en flottant (sans arrondi), débruiteurs, mesures."""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "8")
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
HERE = Path(os.environ.get("T1_ROOT", "/p/scratch/training2666/destors1/MasterChickenHack/Hackathon_Watermark_Localisation_Task"))
sys.path.insert(0, str(HERE / "kit"))
from wmloc.watermarks import load_watermark
from wmloc.utils import miou_image, downsample_mask
OUT = Path(os.environ["T1_OUT"]) if "T1_OUT" in os.environ else HERE / "adem" / "t1-residus"
torch.set_num_threads(4)
METHODS = ("pixelseal", "trustmark", "stegastamp")


def load(method, dev="cuda"):
    return load_watermark(method, models_dir=HERE / "kit" / "models", device=dev)


def load_remover(dev="cuda"):
    from trustmark import TrustMark as TM
    return TM(use_ECC=False, verbose=False, secret_len=100, model_type="Q", device=dev).removal


def meta(split):
    """Liste des entrées avec leur index GLOBAL dans metadata.json (la réserve = index % 5 == 0)."""
    m = json.loads((HERE / "release" / split / "metadata.json").read_text())
    for i, e in enumerate(m): e["gidx"] = i
    return m


def read_img(split, name):
    return np.asarray(Image.open(HERE / "release" / split / "images" / f"{name}.png").convert("RGB"))


def read_mask(name):
    return (np.asarray(Image.open(HERE / "release" / "train" / "masks" / f"{name}.png")) > 127).astype(np.uint8)


def bits_of(e):
    return np.array([int(c) for c in e["message"]], dtype=np.float32)


def to_t(img, dev="cuda"):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None].float().div(255).to(dev)


@torch.no_grad()
def tmpl(wm, img, bits):
    """Gabarit en flottant : ce que le tampon ajouterait à `img` (uint8 HxWx3, n'importe quelle taille) pour chacun
    des K messages `bits` (K x nbits). Retour K x 3 x H x W, unités [0,1], AUCUN arrondi 8 bits."""
    bits = np.atleast_2d(np.asarray(bits, np.float32)); K = len(bits); dev = wm.device
    ist = isinstance(img, torch.Tensor)                  # tenseur 1x3xHxW en [0,1] : chemin torch, sans arrondi
    xt = img.to(dev) if ist else to_t(img, dev); h, w = xt.shape[-2:]
    if wm.name == "pixelseal":
        x = xt.expand(K, -1, -1, -1)
        out = wm.model.embed(x, msgs=torch.from_numpy(bits).to(dev), is_video=False)
        return (out["imgs_w"].clamp(0, 1) - x).float()
    if wm.name == "trustmark":
        if ist: cover = F.interpolate(xt, size=(256, 256), mode="bilinear", antialias=True) * 2 - 1
        else:
            from torchvision import transforms
            cover = transforms.ToTensor()(Image.fromarray(img).resize((256, 256), Image.BILINEAR))[None].to(dev) * 2 - 1
        cover = cover.expand(K, -1, -1, -1)
        stego, _ = wm.tm.encoder(cover, torch.from_numpy(bits).to(dev))
        res = stego.clamp(-1, 1) - cover
        res = res - res.mean(dim=(2, 3), keepdim=True)
        return (F.interpolate(res, size=(h, w), mode="bilinear") / 2).float()
    if ist: small = F.interpolate(xt, size=(400, 400), mode="bicubic", antialias=True).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    else: small = np.asarray(Image.fromarray(img).resize((400, 400), Image.Resampling.BICUBIC), np.float32) / 255
    res = np.concatenate([wm._run(small[None], ["residual"], bits[k:k + 1])[0] for k in range(K)])
    res = torch.from_numpy(res).to(dev).permute(0, 3, 1, 2)
    res = res - res.mean(dim=(2, 3), keepdim=True)
    return F.interpolate(res, size=(h, w), mode="bilinear", antialias=True).float()


def gauss(x, sigma):
    if sigma <= 0: return x
    k = int(4 * sigma + 0.5) * 2 + 1; t = torch.arange(k, device=x.device) - k // 2
    w = torch.exp(-t.float() ** 2 / (2 * sigma ** 2)); w = w / w.sum(); c = x.shape[1]; p = k // 2
    mode = "reflect" if p < min(x.shape[-2:]) else "replicate"
    x = F.conv2d(F.pad(x, (p, p, 0, 0), mode=mode), w.view(1, 1, 1, k).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, p, p), mode=mode), w.view(1, 1, k, 1).repeat(c, 1, 1, 1), groups=c)


LINEAR = lambda n: n[0] == "g" or n.startswith("dog") or n == "ud"


@torch.no_grad()
def hp(name, x, rm=None):
    """Résidu estimé h = x - débruité(x). x : B x 3 x H x W en [0,1]."""
    if name.startswith("dog"):
        a, b = name[3:].split("_"); return gauss(x, float(a)) - gauss(x, float(b))
    if name[0] == "g": return x - gauss(x, float(name[1:]))
    if name.startswith("med"):
        k = int(name[3:]); p = k // 2; B, C, H, W = x.shape; out = []
        for i in range(B):
            u = F.unfold(F.pad(x[i:i + 1], (p,) * 4, mode="reflect"), k).view(1, C, k * k, H * W)
            out.append(x[i:i + 1] - u.median(2).values.view(1, C, H, W))
        return torch.cat(out)
    if name == "ud":
        d = F.interpolate(x, scale_factor=0.5, mode="bilinear", antialias=True)
        return x - F.interpolate(d, size=x.shape[-2:], mode="bilinear")
    if name == "tmrm":                                   # réseau de RETRAIT de TrustMark (KBNet, cadre 256)
        out = []
        for i in range(0, len(x), 4):
            s = F.interpolate(x[i:i + 4], size=(256, 256), mode="bilinear", antialias=True) * 2 - 1
            res = rm(s).clamp(-1, 1) - s
            out.append(-F.interpolate(res, size=x.shape[-2:], mode="bilinear") / 2)
        return torch.cat(out)
    raise ValueError(name)


def expected(name, x, t, rm=None):
    """Ce que le gabarit t deviendrait après le même filtre : F(t) si le filtre est linéaire, sinon F(x) - F(x - t)."""
    if LINEAR(name): return hp(name, t)
    return hp(name, x, rm) - hp(name, x - t, rm)


def cos(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-20))


def auc(pos, neg):
    """Chance qu'une valeur de `pos` dépasse une valeur de `neg` (0,5 = hasard)."""
    pos = np.asarray(pos, np.float32).ravel(); neg = np.asarray(neg, np.float32).ravel()
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    v = np.concatenate([pos, neg]); order = np.argsort(v, kind="stable"); r = np.empty(len(v), np.float64)
    r[order] = np.arange(1, len(v) + 1)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def fit_threshold(M, G, sel, n=41):
    """Seuil global réglé sur les images `sel` (hors réserve)."""
    M = M.astype(np.float32); qs = np.quantile(M[sel], np.linspace(0.02, 0.98, n)); best = (-1, 0)
    for t in qs:
        s = np.mean([miou_image(M[j] > t, G[j]) for j in np.where(sel)[0]])
        if s > best[0]: best = (s, float(t))
    return best[1], best[0]


def score(M, G, t, idx):
    return np.array([miou_image(M[j] > t, G[j]) for j in idx])


def ss_decoder_only(wm):
    """StegaStamp : le graphe ONNX du kit recalcule l'encodeur à chaque appel du décodeur. On écrit un graphe « décodeur seul »
    (sorties de l'encodeur retirées, donc élaguées) et on remplace la session du wrapper : mêmes logits, bien plus rapide."""
    import onnx, onnxruntime as ort
    src = HERE / "kit" / "models" / "stegastamp" / "stega_stamp_logits.onnx"; dst = OUT / "stega_stamp_dec.onnx"
    if not dst.exists():
        m = onnx.load(str(src)); keep = [o for o in m.graph.output if o.name == wm.LOGITS]
        del m.graph.output[:]; m.graph.output.extend(keep); onnx.save(m, str(dst))
    so = ort.SessionOptions(); so.intra_op_num_threads = 4
    prov = [("CUDAExecutionProvider", {"device_id": 0, "arena_extend_strategy": "kSameAsRequested"}), "CPUExecutionProvider"]
    full = wm.sess; wm.sess_full = full; wm.sess = ort.InferenceSession(str(dst), so, providers=prov)
    names = [i.name for i in wm.sess.get_inputs()]
    def _run(x_nhwc, outputs, secret=None):
        if outputs != [wm.LOGITS]:
            secret = np.zeros((len(x_nhwc), 100), np.float32) if secret is None else secret
            return full.run(outputs, {"image": x_nhwc.astype(np.float32), "secret": secret})
        feed = {"image": x_nhwc.astype(np.float32)}
        if "secret" in names: feed["secret"] = np.zeros((len(x_nhwc), 100), np.float32)
        return wm.sess.run(outputs, feed)
    wm._run = _run
    return wm
