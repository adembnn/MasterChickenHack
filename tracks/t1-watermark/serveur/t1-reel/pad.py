"""Quelle marge reproduit le réel ? Pleines fabriquées en filigranant une toile plus grande (image + marge p en miroir de chaque côté) puis en rognant.
On compare au réel : bits justes et marge du décodeur, logit du réseau fabriqué-seul au centre et aux bords. Idem pour les rectangles (bits dans la boîte)."""
import sys, json, argparse, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); import reel2 as R
from wmloc.watermarks import load_watermark
method, tag, N = sys.argv[1], sys.argv[2], int(sys.argv[3]); dev = "cuda"; rng = np.random.default_rng(0)
trm, tem = R.metas(); RT = np.load(R.CACHE / "REAL_train.npy", mmap_mode="r"); MT = np.load(R.CACHE / "MASK_train.npy", mmap_mode="r"); Y = np.load(R.CACHE / "Y128_train.npy")
run = R.Runner(argparse.Namespace(enc="resnet34", arch="unet", pretrained=0, stem1=0, flag=0, hp=0, prec="fp16"), dev)
P = R.ROOT / "runs" / tag / method / "fold-1"; run.net.load_state_dict(torch.load(P / "ema.pt", map_location=dev)["state"] if (P / "ema.pt").exists() else torch.load(P / "ckpt.pt", map_location=dev)["ema"])
wm = load_watermark(method, models_dir=R.HERE / "kit" / "models", device=dev); st = {r["i"]: r for r in json.load(open(R.ROOT / f"stats_{method}.json"))} if (R.ROOT / f"stats_{method}.json").exists() else {}
ids = [i for i, m in enumerate(trm) if m["method"] == method and i % 5 != 0]; fulls = [i for i in ids if trm[i]["full_watermark"]][:N]; others = [i for i, m in enumerate(trm) if m["method"] != method and i % 5 != 0]
edge = np.zeros((128, 128), bool); edge[:16] = edge[-16:] = True; edge[:, :16] = edge[:, -16:] = True
def dec(img, bits):
    l = wm.logits(torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None].float() / 255)[0].cpu().numpy() * (bits * 2 - 1); return (l > 0).mean(), l.mean()
def report(nm, imgs, msgs, masks):
    lg = run.predict(run.net, np.stack(imgs), np.ones(len(imgs)), tta=1, bs=8); d = np.array([dec(x, b) for x, b in zip(imgs, msgs)])
    c = np.mean([l[(m > 0) & ~edge].mean() for l, m in zip(lg, masks)]); e = np.mean([l[(m > 0) & edge].mean() for l, m in zip(lg, masks)])
    print(f"{nm:26s} bits justes {d[:, 0].mean():.3f} | marge décodeur {d[:, 1].mean():.2f} | logit réseau centre {c:+.2f} | bords {e:+.2f}")
def embed_pad(b, bits, p, oy=None, ox=None):
    if p == 0: return wm.embed(b, bits)
    oy = p if oy is None else oy; ox = p if ox is None else ox
    c = np.pad(b, ((oy, 2 * p - oy), (ox, 2 * p - ox), (0, 0)), mode="reflect"); return wm.embed(np.ascontiguousarray(c), bits)[oy:oy + b.shape[0], ox:ox + b.shape[1]]
bits_of = lambda i: np.array([int(c) for c in trm[i]["message"]])
print(f"## {method} : pleines ({len(fulls)})")
report("RÉEL", [np.ascontiguousarray(RT[i]) for i in fulls], [bits_of(i) for i in fulls], [Y[i] for i in fulls])
bases = [np.ascontiguousarray(RT[int(rng.choice(others))]) for _ in fulls]; msgs = [rng.integers(0, 2, wm.nbits) for _ in fulls]; holes = [~np.array(MT[i]).astype(bool) for i in fulls]
for p in (0, 2, 4, 8, 16, 32, 64, 128):
    report(f"fabriqué, marge {p} px", [np.where(h[..., None], b, embed_pad(b, m, p)) for h, b, m in zip(holes, bases, msgs)], msgs, [Y[i] for i in fulls])
report("fabriqué, marge 0 à 64 décentrée", [np.where(h[..., None], b, embed_pad(b, m, 32, int(rng.integers(0, 65)), int(rng.integers(0, 65)))) for h, b, m in zip(holes, bases, msgs)], msgs, [Y[i] for i in fulls])
if st:
    rects = [st[i] for i in ids if not st[i]["full"] and st[i]["rect"] > 0.98][:N]; print(f"## {method} : rectangles ({len(rects)}) : bits justes en rognant la vraie boîte ; réel = {np.mean([r['a_box'] for r in rects]):.3f}")
    for p in (0, 2, 4, 8, 16, 32):
        a = []
        for r in rects:
            y0, x0, h, w = r["y0"], r["x0"], r["h"], r["w"]; b = np.ascontiguousarray(RT[int(rng.choice(others))])[y0:y0 + h, x0:x0 + w]; bits = rng.integers(0, 2, wm.nbits)
            try: a.append(dec(embed_pad(np.ascontiguousarray(b), bits, min(p, h - 1, w - 1)), bits)[0])
            except Exception as e: pass
        print(f"  marge {p:2d} px : {np.mean(a):.3f}  (n={len(a)})")
