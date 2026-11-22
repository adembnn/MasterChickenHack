"""Planche : vraies pleines (image | vrai masque | carte du réseau fabriqué-seul) puis pleines fabriquées avec les mêmes masques."""
import sys, argparse, numpy as np, torch
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent)); import reel2 as R
from wmloc.watermarks import load_watermark
method, tag, N = sys.argv[1], sys.argv[2], int(sys.argv[3]); dev = "cuda"; rng = np.random.default_rng(1)
trm, tem = R.metas(); RT = np.load(R.CACHE / "REAL_train.npy", mmap_mode="r"); MT = np.load(R.CACHE / "MASK_train.npy", mmap_mode="r"); Y = np.load(R.CACHE / "Y128_train.npy")
run = R.Runner(argparse.Namespace(enc="resnet34", arch="unet", pretrained=0, stem1=0, flag=0, hp=0, prec="fp16"), dev)
P = R.ROOT / "runs" / tag / method / "fold-1"; run.net.load_state_dict(torch.load(P / "ema.pt", map_location=dev)["state"] if (P / "ema.pt").exists() else torch.load(P / "ckpt.pt", map_location=dev)["ema"])
wm = load_watermark(method, models_dir=R.HERE / "kit" / "models", device=dev)
fulls = [i for i, m in enumerate(trm) if m["method"] == method and i % 5 != 0 and m["full_watermark"]][:N]; others = [i for i, m in enumerate(trm) if m["method"] != method and i % 5 != 0]
real = [np.ascontiguousarray(RT[i]) for i in fulls]; bases = [np.ascontiguousarray(RT[int(rng.choice(others))]) for _ in fulls]
syn = [np.where(~np.array(MT[i]).astype(bool)[..., None], b, wm.embed(b, rng.integers(0, 2, wm.nbits))) for i, b in zip(fulls, bases)]
# troisième bloc : la VRAIE image re-filigranée par nous à l'intérieur de son propre trou seulement n'a pas de sens ; à la place : vraie image dont on re-filigrane tout (double filigrane)
def heat(l): p = 1 / (1 + np.exp(-np.clip(l, -8, 8))); g = (p * 255).astype(np.uint8); return np.asarray(Image.fromarray(np.stack([g, g, 255 - g], -1)).resize((256, 256), Image.Resampling.NEAREST))
def cell(im): return np.asarray(Image.fromarray(im).resize((256, 256), Image.Resampling.LANCZOS))
def gt(i): g = (Y[i] * 255).astype(np.uint8); return np.asarray(Image.fromarray(np.stack([g, g, 255 - g], -1)).resize((256, 256), Image.Resampling.NEAREST))
lr = run.predict(run.net, np.stack(real), np.ones(N), tta=1); ls = run.predict(run.net, np.stack(syn), np.ones(N), tta=1)
rows = [np.concatenate([np.concatenate([cell(real[j]), gt(fulls[j]), heat(lr[j]), cell(syn[j]), heat(ls[j])], 1) for j in range(k, k + 2)], 1) for k in range(0, N, 2)]
Image.fromarray(np.concatenate(rows, 0)).save(R.ROOT / f"planche_gap_{method}.png"); print("ok")
