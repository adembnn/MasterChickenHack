"""Test physique : que reste-t-il du filigrane PixelSeal dans un rectangle selon sa taille ? Pour chaque vrai rectangle du train (boîte de la vérité),
(a) réel : bits justes relus par le décodeur sur la boîte exacte ; (b) fabriqué avec le vrai encodeur du kit, même boîte, sur 3 images propres pour
PixelSeal (images TrustMark du train) : part des pixels modifiés, écart RMS, bits justes sur la boîte exacte."""
import sys, json
import numpy as np, torch
from PIL import Image
from verif import HERE, load_truth
sys.path.insert(0, str(HERE / "kit")); from wmloc.watermarks import load_watermark
ME = sys.argv[1] if len(sys.argv) > 1 else "pixelseal"; OTHER = "trustmark" if ME != "trustmark" else "stegastamp"
wm = load_watermark(ME, models_dir=HERE / "kit" / "models", device="cuda"); rng = np.random.default_rng(0)
meta, g = load_truth(check_png=False); img = lambda n: np.asarray(Image.open(HERE / "release" / "train" / "images" / f"{n}.png").convert("RGB"))
def T(a): return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)[None].float().div(255)
def acc(a, bits): return float((wm.decode(T(a))[0].cpu().numpy() == np.array(list(map(int, bits)))).mean())
clean = [m["image_name"] for m in meta if m["method"] == OTHER]; rows = []
for m in meta:
    if m["method"] != ME or m["full_watermark"]: continue
    G = g[m["image_name"]]; ys, xs = np.where(G)
    if len(ys) == 0: continue
    y0, y1, x0, x1 = ys.min() * 4, (ys.max() + 1) * 4, xs.min() * 4, (xs.max() + 1) * 4
    if G.sum() < 0.95 * (y1 - y0) * (x1 - x0) / 16: continue                      # rectangles seulement
    real = acc(img(m["image_name"])[y0:y1, x0:x1], m["message"]); ch, rms, fa = [], [], []
    for _ in range(3):
        b = img(clean[int(rng.integers(len(clean)))]); bits = rng.integers(0, 2, wm.nbits); c = np.ascontiguousarray(b[y0:y1, x0:x1]); w = wm.embed(c, bits); d = w.astype(np.float32) - c
        ch.append(float((np.abs(d).max(-1) > 0).mean())); rms.append(float(np.sqrt((d ** 2).mean()))); fa.append(acc(w, bits))
    rows.append((min(y1 - y0, x1 - x0), y1 - y0, x1 - x0, real, np.mean(ch), np.mean(rms), np.mean(fa)))
R = np.array(rows); print(f"{ME} : {len(R)} vrais rectangles du train")
for lo, hi in ((0, 32), (32, 64), (64, 128), (128, 513)):
    k = (R[:, 0] >= lo) & (R[:, 0] < hi)
    if k.sum(): print(f"  petit côté {lo:3d}-{hi:3d} px n={k.sum():2d} | RÉEL bits justes sur la boîte exacte {R[k, 3].mean():.3f} | FABRIQUÉ : pixels modifiés {R[k, 4].mean():.1%}, écart RMS {R[k, 5].mean():.2f} niveau, bits justes {R[k, 6].mean():.3f}")
b = img(clean[0]); bits = rng.integers(0, 2, wm.nbits); w = wm.embed(b, bits); d = w.astype(np.float32) - b
print(f"  image entière 512 px (référence) : pixels modifiés {(np.abs(d).max(-1) > 0).mean():.1%}, écart RMS {np.sqrt((d ** 2).mean()):.2f}, bits justes {acc(w, bits):.3f}")
print("  détail par rectangle (petit côté, h x l, réel, modifiés, RMS, fabriqué) : " + " | ".join(f"{int(r[1])}x{int(r[2])} {r[3]:.2f} {r[4]:.0%} {r[5]:.2f} {r[6]:.2f}" for r in R[np.argsort(R[:, 1] * R[:, 2])]))
