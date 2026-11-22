"""Rattrapage de l'affinage : pour les boîtes dont le cadre n'a pas été trouvé au premier passage (R3 >= 0,7), recherche plus large
(décalage ±24 px par pas de 2, taille ±12 px par pas de 6), puis descente au pixel. Entrée : boxes_<m>_<split><tag>_r3.json ; sortie : ..._r3c.json."""
import argparse, json, time, itertools
import numpy as np, torch
from torchvision.ops import roi_align
from rlib import *
ap = argparse.ArgumentParser(); ap.add_argument("--method", default="trustmark"); ap.add_argument("--split", default="train"); ap.add_argument("--tag", default="_g13k")
ap.add_argument("--K", type=int, default=2); ap.add_argument("--bs", type=int, default=32); ap.add_argument("--n", type=int, default=0); ap.add_argument("--src", default="_r3"); ap.add_argument("--thr", type=float, default=0.6); a = ap.parse_args()
dev = "cuda"; wm = load(a.method, dev); rng = np.random.default_rng(1)

@torch.no_grad()
def r3_batch(x, boxes, msgs):
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
def wide(x, b, msgs):
    """Recherche proportionnelle à la taille : décalage jusqu'à ±25 % du côté (au moins 24 px), taille ±12 %, pas = côté/64 (au moins 2 px)."""
    h, w = b[2] - b[0], b[3] - b[1]; ry, rx = max(24, int(0.25 * h)), max(24, int(0.25 * w)); sy, sx = max(2, round(h / 64)), max(2, round(w / 64)); c = set()
    for fh, fw in itertools.product((-0.12, -0.06, 0, 0.06, 0.12), repeat=2):
        dh, dw = int(round(fh * h)), int(round(fw * w))
        for dy in range(-ry, ry + 1, sy):
            for dx in range(-rx, rx + 1, sx): c.add(clipbox((b[0] + dy - dh // 2, b[1] + dx - dw // 2, b[2] + dy + dh - dh // 2, b[3] + dx + dw - dw // 2)))
    c = list(c); v = r3_batch(x, c, msgs); j = int(np.argmin(v)); best = (c[j], float(v[j])); n = len(c)
    for step, rad in ((max(1, sy // 2), 2 * max(1, sy // 2)), (1, 2), (1, 2), (1, 1)):
        b0 = best[0]; c2 = list({clipbox((b0[0] + d[0], b0[1] + d[1], b0[2] + d[2], b0[3] + d[3])) for d in itertools.product(range(-rad, rad + 1, step), repeat=4)})
        v2 = r3_batch(x, c2, msgs); n += len(c2); j2 = int(np.argmin(v2))
        if v2[j2] < best[1]: best = (c2[j2], float(v2[j2]))
    return best[0], best[1], float(np.median(v)), n
R = json.load(open(OUT / f"boxes_{a.method}_{a.split}{a.tag}{a.src}.json")); mb = {e["image_name"]: e for e in meta(a.split)}; t0 = time.time(); done = 0
todo = [r for r in R if "frame" in r and r["r3"] >= a.thr]; print("REFINE2", a.split, "à rattraper : %d" % len(todo), flush=True)
for r in todo[:a.n or None]:
    x = to_t(read_img(a.split, r["name"]), dev); m = bits_of(mb[r["name"]]); msgs = np.concatenate([m[None], rng.integers(0, 2, (a.K, wm.nbits)).astype(np.float32)])
    f, v, med, n = wide(x, tuple(int(t) for t in r["box"]), msgs); done += 1
    print("  %d/%d, %d s : R3 %.2f -> %.2f (médiane de la grille %.2f) cadre %s%s" % (done, len(todo), time.time() - t0, r["r3"], v, med, f, (" masque %s" % (r["gt_box"],)) if "gt_box" in r else ""), flush=True)
    if v < r["r3"]: r["frame"] = f; r["r3"] = v; r["wide"] = True
    json.dump(R, open(OUT / f"boxes_{a.method}_{a.split}{a.tag}_r3c.json", "w"), default=float)
json.dump(R, open(OUT / f"boxes_{a.method}_{a.split}{a.tag}_r3c.json", "w"), default=float)
S = [r for r in R if "frame" in r and r.get("rect")]
if S:
    ok = np.array([r["r3"] < 0.7 for r in S]); d = np.array([[r["frame"][k] - r["gt_box"][k] for k in range(4)] for r in S])
    print("FIN REFINE2 : rectangles vrais %d, cadres trouvés (R3 < 0,7) %d ; écart au masque sur ceux-là : médiane %s, |écart| moyen %s" % (len(S), ok.sum(), np.median(d[ok], 0), np.abs(d[ok] - np.array([-3.5, -3.5, 3.5, 3.5])).mean(0).round(1)))
