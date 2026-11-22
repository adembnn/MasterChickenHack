"""Quelle transformation, appliquée après le tampon, fait que (a) le décodeur relit encore le message mais (b) le tampon ne reconnaît plus
son filigrane (R3 proche de 1), comme sur les vraies images pleines ? Images fabriquées, tampon sur l'image entière."""
import argparse, io, time
import numpy as np, torch
from PIL import Image
from rlib import *
ap = argparse.ArgumentParser(); ap.add_argument("--method", required=True); ap.add_argument("--n", type=int, default=6); a = ap.parse_args()
dev = "cuda"; wm = load(a.method, dev); rng = np.random.default_rng(0)
def jpeg(q):
    def f(img, m):
        b = io.BytesIO(); Image.fromarray(wm.embed(img, m)).save(b, "JPEG", quality=q); return np.asarray(Image.open(io.BytesIO(b.getvalue())).convert("RGB"))
    return f
def hires(scale, down):
    def f(img, m):
        big = np.asarray(Image.fromarray(img).resize((512 * scale, 512 * scale), Image.BICUBIC)); return np.asarray(Image.fromarray(wm.embed(big, m)).resize((512, 512), down))
    return f
def updown(up, down):
    def f(img, m): return np.asarray(Image.fromarray(wm.embed(img, m)).resize((1024, 1024), up).resize((512, 512), down))
    return f
def noise(s):
    def f(img, m): return np.clip(wm.embed(img, m).astype(np.float32) + rng.normal(0, s, img.shape), 0, 255).round().astype(np.uint8)
    return f
def lowres(sz):
    def f(img, m):
        small = np.asarray(Image.fromarray(img).resize((sz, sz), Image.BICUBIC)); return np.asarray(Image.fromarray(wm.embed(small, m)).resize((512, 512), Image.BICUBIC))
    return f
T = {"aucune": lambda img, m: wm.embed(img, m), "JPEG 95": jpeg(95), "JPEG 85": jpeg(85), "JPEG 70": jpeg(70),
     "tampon en 1024 puis réduit (bicubique)": hires(2, Image.BICUBIC), "tampon en 1024 puis réduit (lanczos)": hires(2, Image.LANCZOS), "tampon en 768 puis réduit": hires(1, Image.BICUBIC) if False else (lambda img, m: np.asarray(Image.fromarray(wm.embed(np.asarray(Image.fromarray(img).resize((768, 768), Image.BICUBIC)), m)).resize((512, 512), Image.BICUBIC))),
     "tampon en 256 puis agrandi": lowres(256), "agrandi x2 puis réduit": updown(Image.BICUBIC, Image.BICUBIC), "bruit 1 niveau": noise(1.0), "bruit 3 niveaux": noise(3.0)}
others = [e for e in meta("train") if e["method"] != a.method and e["gidx"] % 5 != 0][:a.n]; acc = {k: {"r3": [], "bits": [], "psnr": []} for k in T}
for e in others:
    img = read_img("train", e["image_name"]); m = rng.integers(0, 2, wm.nbits).astype(np.float32); msgs = np.concatenate([m[None], rng.integers(0, 2, (2, wm.nbits)).astype(np.float32)])
    for k, f in T.items():
        y = np.ascontiguousarray(f(img, m)); en = tmpl(wm, y, msgs).pow(2).sum((1, 2, 3)); lg = wm.logits(to_t(y, dev))[0].cpu().numpy()
        acc[k]["r3"].append(float(en[0] / en[1:].mean())); acc[k]["bits"].append(float(((lg > 0) == (m > 0)).mean())); acc[k]["psnr"].append(10 * np.log10(255 ** 2 / np.mean((y.astype(np.float32) - img) ** 2)))
print("TRANSF", a.method, "n=%d images fabriquées. Vraies pleines : R3 ~0,95-1,0 (TrustMark, PixelSeal), ~0,75 (StegaStamp), bits 0,97-0,98" % len(others))
print("  transformation après le tampon              R3      bits justes   PSNR (dB, image finale contre image propre)")
for k, v in acc.items(): print("  %-42s  %.2f    %.3f         %.1f" % (k, np.mean(v["r3"]), np.mean(v["bits"]), np.mean(v["psnr"])))
