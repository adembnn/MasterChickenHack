"""Où, dans l'image, le réseau fabriqué-seul rate-t-il le vrai filigrane ? Moyenne du logit par case d'une grille 8x8 (zones filigranées seulement), réel contre fabriqué.
Et force locale du motif StegaStamp : pente de la régression du détail de l'image sur le motif attendu, par case 8x8."""
import sys, argparse, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); import reel2 as R
from wmloc.watermarks import load_watermark
from scipy import ndimage as ndi
method, tag, N = sys.argv[1], sys.argv[2], int(sys.argv[3]); dev = "cuda"; rng = np.random.default_rng(1); np.set_printoptions(precision=1, suppress=True, linewidth=200)
trm, tem = R.metas(); RT = np.load(R.CACHE / "REAL_train.npy", mmap_mode="r"); MT = np.load(R.CACHE / "MASK_train.npy", mmap_mode="r"); Y = np.load(R.CACHE / "Y128_train.npy")
run = R.Runner(argparse.Namespace(enc="resnet34", arch="unet", pretrained=0, stem1=0, flag=0, hp=0, prec="fp16"), dev)
P = R.ROOT / "runs" / tag / method / "fold-1"; run.net.load_state_dict(torch.load(P / "ema.pt", map_location=dev)["state"] if (P / "ema.pt").exists() else torch.load(P / "ckpt.pt", map_location=dev)["ema"])
wm = load_watermark(method, models_dir=R.HERE / "kit" / "models", device=dev)
fulls = [i for i, m in enumerate(trm) if m["method"] == method and i % 5 != 0 and m["full_watermark"]][:N]; others = [i for i, m in enumerate(trm) if m["method"] != method and i % 5 != 0]
real = [np.ascontiguousarray(RT[i]) for i in fulls]; bases = [np.ascontiguousarray(RT[int(rng.choice(others))]) for _ in fulls]; msgs = [np.array([int(c) for c in trm[i]["message"]]) for i in fulls]
W = [wm.embed(b, m) for b, m in zip(bases, msgs)]; syn = [np.where(~np.array(MT[i]).astype(bool)[..., None], b, w) for i, b, w in zip(fulls, bases, W)]
def grid(vals, masks):                                   # moyenne par case 8x8 des valeurs là où masque = 1
    num = np.zeros((8, 8)); den = np.zeros((8, 8))
    for v, m in zip(vals, masks):
        k = v.shape[0] // 8; num += (v * m).reshape(8, k, 8, k).sum((1, 3)); den += m.reshape(8, k, 8, k).sum((1, 3))
    return num / np.maximum(den, 1)
Ym = [Y[i].astype(np.float32) for i in fulls]
lr = run.predict(run.net, np.stack(real), np.ones(N), tta=1); ls = run.predict(run.net, np.stack(syn), np.ones(N), tta=1)
print(f"## {method} : logit moyen du réseau fabriqué-seul par case 8x8, zones filigranées de {N} pleines"); print("RÉEL"); print(grid(lr, Ym)); print("FABRIQUÉ (mêmes masques)"); print(grid(ls, Ym))
def beta(imgs):                                          # pente locale : détail de l'image expliqué par le motif attendu (1 = motif présent à pleine force)
    out = []
    for x, m in zip(imgs, msgs):
        T = wm.embed(x, m).astype(np.float32) - x; xf = x.astype(np.float32); hp = lambda a: a - ndi.gaussian_filter(a, (3, 3, 0))
        a, t = hp(xf), hp(T); num = (a * t).sum(2); den = (t * t).sum(2); out.append((num, den))
    return out
M512 = [np.array(MT[i]).astype(np.float32) for i in fulls]
for nm, imgs in (("RÉEL", real), ("FABRIQUÉ", syn)):
    b = beta(imgs); print(f"pente du motif attendu par case 8x8 ({nm})"); print(np.round(grid([n for n, d in b], M512) / np.maximum(grid([d for n, d in b], M512), 1e-6), 2))
