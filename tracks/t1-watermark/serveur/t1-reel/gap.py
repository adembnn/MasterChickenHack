"""Pourquoi le fabriqué ne se transfère-t-il pas aux vraies images PLEINES ? On prend un réseau entraîné sur fabriqué seul comme instrument :
logit moyen dans la zone filigranée / dans le trou, sur les vraies pleines, puis sur des pleines fabriquées avec les MÊMES masques selon plusieurs hypothèses
(force du filigrane, filigranage en 1024 puis réduction...). L'hypothèse qui reproduit les chiffres du réel (logits ET bits justes) est la bonne recette."""
import sys, json, argparse, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent)); import reel2 as R
from wmloc.watermarks import load_watermark
method, tag = sys.argv[1], sys.argv[2]; N = int(sys.argv[3]); torch.set_num_threads(4); dev = "cuda"; rng = np.random.default_rng(0)
trm, tem = R.metas(); RT = np.load(R.CACHE / "REAL_train.npy", mmap_mode="r"); MT = np.load(R.CACHE / "MASK_train.npy", mmap_mode="r"); Y = np.load(R.CACHE / "Y128_train.npy")
a = argparse.Namespace(enc="resnet34", arch="unet", pretrained=0, stem1=0, flag=0, hp=0, prec="fp16"); run = R.Runner(a, dev)
P = R.ROOT / "runs" / tag / method / "fold-1"; sd = torch.load(P / "ema.pt", map_location=dev)["state"] if (P / "ema.pt").exists() else torch.load(P / "ckpt.pt", map_location=dev)["ema"]
run.net.load_state_dict(sd); wm = load_watermark(method, models_dir=R.HERE / "kit" / "models", device=dev)
fulls = [i for i, m in enumerate(trm) if m["method"] == method and i % 5 != 0 and m["full_watermark"]][:N]
others = [i for i, m in enumerate(trm) if m["method"] != method and i % 5 != 0]
def measure(imgs, masks128, msgs):
    lg = run.predict(run.net, np.stack(imgs), np.ones(len(imgs)), tta=1); w = np.array([l[m > 0].mean() for l, m in zip(lg, masks128)]); h = np.array([l[m == 0].mean() for l, m in zip(lg, masks128) if (m == 0).any()])
    X = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float() / 255; acc = np.mean([((wm.logits(X[j:j + 1])[0].cpu().numpy() > 0) == b).mean() for j, b in enumerate(msgs)])
    mar = np.mean([np.abs(wm.logits(X[j:j + 1])[0].cpu().numpy()).mean() for j in range(len(imgs))])
    sc = np.mean([R.miou_image(l > 0, m) for l, m in zip(lg, masks128)])
    return f"logit zone filigranée {w.mean():+.2f} | trou {h.mean():+.2f} | mIoU seuil 0 {sc:.3f} | bits justes {acc:.3f} | |logit décodeur| {mar:.2f}"
bits_of = lambda i: np.array([int(c) for c in trm[i]["message"]])
print(f"## {method} (instrument : {tag}, fabriqué seul) sur {len(fulls)} pleines hors réserve")
print(f"{'RÉEL':34s}", measure([np.ascontiguousarray(RT[i]) for i in fulls], [Y[i] for i in fulls], [bits_of(i) for i in fulls]))
bases = [np.ascontiguousarray(RT[int(rng.choice(others))]) for _ in fulls]; msgs = [rng.integers(0, 2, wm.nbits) for _ in fulls]; holes = [~np.array(MT[i]).astype(bool) for i in fulls]
W512 = [wm.embed(b, m) for b, m in zip(bases, msgs)]
def up_embed_down(b, m, S, filt):
    big = np.asarray(Image.fromarray(b).resize((S, S), Image.Resampling.BICUBIC)); return np.asarray(Image.fromarray(wm.embed(big, m)).resize((512, 512), filt))
import io
from scipy import ndimage as ndi
def jpeg(w, q):
    b = io.BytesIO(); Image.fromarray(w).save(b, "JPEG", quality=q, subsampling=0); return np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"))
def jpeg420(w, q):
    b = io.BytesIO(); Image.fromarray(w).save(b, "JPEG", quality=q); return np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"))
def webp(w, q):
    b = io.BytesIO(); Image.fromarray(w).save(b, "WEBP", quality=q); return np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"))
def downup(w, S, f1=Image.Resampling.LANCZOS, f2=Image.Resampling.LANCZOS): return np.asarray(Image.fromarray(w).resize((S, S), f1).resize((512, 512), f2))
def blur(w, sg): return np.clip(np.round(ndi.gaussian_filter(w.astype(np.float32), (sg, sg, 0))), 0, 255).astype(np.uint8)
def noise(w, sg): return np.clip(np.round(w + rng.normal(0, sg, w.shape)), 0, 255).astype(np.uint8)
V = {"kit en 512 (recette actuelle)": lambda w: w}
for q in (98, 95, 90, 80, 70): V[f"puis JPEG {q} (4:4:4)"] = lambda w, q=q: jpeg(w, q)
for q in (95, 85): V[f"puis JPEG {q} (4:2:0)"] = lambda w, q=q: jpeg420(w, q)
for q in (95, 80): V[f"puis WebP {q}"] = lambda w, q=q: webp(w, q)
for S in (448, 384, 256): V[f"puis réduit à {S} et ré-agrandi"] = lambda w, S=S: downup(w, S)
V["puis agrandi à 1024 et réduit"] = lambda w: downup(w, 1024, Image.Resampling.BICUBIC, Image.Resampling.BICUBIC)
for sg in (0.5, 1.0): V[f"puis flou sigma {sg}"] = lambda w, sg=sg: blur(w, sg)
for sg in (1.0, 2.0, 4.0): V[f"puis bruit sigma {sg}"] = lambda w, sg=sg: noise(w, sg)
for nm, f in V.items():
    imgs = [np.where(h[..., None], b, f(w)) for h, b, w in zip(holes, bases, W512)]
    print(f"{nm:34s}", measure(imgs, [Y[i] for i in fulls], msgs))
