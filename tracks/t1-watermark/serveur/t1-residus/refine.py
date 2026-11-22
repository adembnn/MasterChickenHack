"""Affinage au pixel des boîtes trouvées par le décodeur, avec le critère « re-tamponnage » (R3) :
R3(boîte) = énergie de ce que le tampon ajouterait au rognage avec le VRAI message / avec des messages au hasard.
Très pointu (fabriqué : 0,37 à la boîte exacte, 0,64 à 1 px, 1,0 à 4 px). TrustMark, lots sur GPU.
Entrée : boxes_<methode>_<split><tag>.json ; sortie : même fichier + clés frame (cadre affiné), r3, et sur le train l'écart au masque."""
import argparse, json, time, itertools
import numpy as np, torch, torch.nn.functional as F
from torchvision.ops import roi_align
from rlib import *
ap = argparse.ArgumentParser(); ap.add_argument("--method", default="trustmark"); ap.add_argument("--split", default="train"); ap.add_argument("--tag", default="_g13k")
ap.add_argument("--zmin", type=float, default=6.5); ap.add_argument("--n", type=int, default=0); ap.add_argument("--K", type=int, default=2); ap.add_argument("--bs", type=int, default=128); a = ap.parse_args()
dev = "cuda"; wm = load(a.method, dev); rng = np.random.default_rng(0); assert a.method == "trustmark"

@torch.no_grad()
def r3_batch(x, boxes, msgs):
    """x 1x3x512x512 ; boxes liste (y0,x0,y1,x1) ; msgs (1+K) x nbits, le vrai d'abord. Retour R3 par boîte."""
    out = []; M = torch.from_numpy(msgs).to(dev); k1 = len(msgs)
    for i in range(0, len(boxes), a.bs):
        bb = boxes[i:i + a.bs]; b = torch.tensor([(0, t[1], t[0], t[3], t[2]) for t in bb], dtype=torch.float32, device=dev)
        cover = roi_align(x, b, output_size=(256, 256), spatial_scale=1.0, sampling_ratio=2, aligned=True).clamp(0, 1) * 2 - 1
        cov = cover.repeat_interleave(k1, 0); sec = M.repeat(len(bb), 1)
        with torch.autocast("cuda", dtype=torch.float16):
            stego, _ = wm.tm.encoder(cov, sec)
        res = stego.float().clamp(-1, 1) - cov; res = res - res.mean((2, 3), keepdim=True)
        en = res.pow(2).sum((1, 2, 3)).view(len(bb), k1); out.append(en[:, 0] / en[:, 1:].mean(1))
    return torch.cat(out).cpu().numpy()

def clipbox(b):
    y0, x0, y1, x1 = b; y0 = max(0, min(y0, 500)); x0 = max(0, min(x0, 500)); return (y0, x0, max(y0 + 8, min(y1, 512)), max(x0 + 8, min(x1, 512)))

def refine(x, b, msgs):
    best = (b, float(r3_batch(x, [b], msgs)[0])); n = 1
    for step, rad in ((3, 9), (1, 2), (1, 2)):
        offs = range(-rad, rad + 1, step); b0 = best[0]
        c = list({clipbox((b0[0] + d[0], b0[1] + d[1], b0[2] + d[2], b0[3] + d[3])) for d in itertools.product(offs, repeat=4)})
        v = r3_batch(x, c, msgs); n += len(c); j = int(np.argmin(v))
        if v[j] < best[1]: best = (c[j], float(v[j]))
    return best[0], best[1], n

R = json.load(open(OUT / f"boxes_{a.method}_{a.split}{a.tag}.json")); t0 = time.time(); done = 0; meta_by = {e["image_name"]: e for e in meta(a.split)}
for r in R:
    if "box" not in r or r["z"] < a.zmin: continue
    e = meta_by[r["name"]]; x = to_t(read_img(a.split, r["name"]), dev); m = bits_of(e)
    msgs = np.concatenate([m[None], rng.integers(0, 2, (a.K, wm.nbits)).astype(np.float32)])
    f, v, n = refine(x, tuple(int(t) for t in r["box"]), msgs); r["frame"] = f; r["r3"] = v; r["r3_box"] = float(r3_batch(x, [tuple(int(t) for t in r["box"])], msgs)[0]); done += 1
    if "gt_box" in r: r["r3_gt"] = float(r3_batch(x, [tuple(r["gt_box"])], msgs)[0])
    if done % 5 == 0: print("  %d boîtes affinées, %d s ; dernière : R3 %.2f -> %.2f, cadre %s%s" % (done, time.time() - t0, r["r3_box"], v, f, (" masque %s" % (r["gt_box"],)) if "gt_box" in r else ""), flush=True)
    if a.n and done >= a.n: break
json.dump(R, open(OUT / f"boxes_{a.method}_{a.split}{a.tag}_r3.json", "w"), default=float)
print("REFINE", a.method, a.split, "n=%d sec=%d" % (done, time.time() - t0))
S = [r for r in R if "frame" in r and r.get("rect")]
if S:
    d = np.array([[r["frame"][k] - r["gt_box"][k] for k in range(4)] for r in S]); v = np.array([r["r3"] for r in S]); ok = v < 0.7
    print("rectangles vrais n=%d | R3 au cadre affiné : médiane %.2f [q10 %.2f ; q90 %.2f] ; R3 < 0,7 : %d | R3 à la boîte du décodeur : médiane %.2f | R3 au masque : médiane %.2f" % (len(S), np.median(v), np.quantile(v, .1), np.quantile(v, .9), ok.sum(), np.median([r["r3_box"] for r in S]), np.median([r["r3_gt"] for r in S])))
    print("écart cadre affiné - boîte du masque (y0,x0,y1,x1), sur R3 < 0,7 : médiane %s ; écart-type %s" % (np.median(d[ok], 0), d[ok].std(0).round(1)))
    for row, r in zip(d[ok][:12], [r for r, k in zip(S, ok) if k][:12]): print("     ", row, "R3=%.2f" % r["r3"], "taille masque %dx%d" % (r["gt_box"][2] - r["gt_box"][0], r["gt_box"][3] - r["gt_box"][1]))
    res_ = np.array([r["gidx"] % 5 == 0 for r in S]); off = np.round(np.median(d[ok & ~res_], 0)).astype(int) if (ok & ~res_).sum() else np.zeros(4, int)
    def pm(r, use):
        b = r["box"] if not use else [r["frame"][k] - off[k] for k in range(4)]
        mk = np.zeros((512, 512), np.uint8); mk[max(0, int(b[0])):int(b[2]), max(0, int(b[1])):int(b[3])] = 1
        return miou_image(downsample_mask(mk), downsample_mask(read_mask(r["name"])))
    a0 = np.array([pm(r, False) for r in S]); a1 = np.array([pm(r, bool(k)) for r, k in zip(S, ok)])
    print("mIoU officiel des rectangles (masque = boîte) : décodeur seul %.3f -> affiné R3 puis rétréci de %s (réglé hors réserve) %.3f | réserve : %.3f -> %.3f (n=%d)" % (a0.mean(), off, a1.mean(), a0[res_].mean(), a1[res_].mean(), res_.sum()))
