"""Quel est le vrai cadre du filigrane dans les images PLEINES ? Recherche par le critère « re-tamponnage » (R3) sur des cadres
qui peuvent dépasser l'image (l'image est prolongée par recopie du bord). TrustMark. R3 bas = le tampon reconnaît son filigrane."""
import argparse, json, time, itertools
import numpy as np, torch, torch.nn.functional as F
from torchvision.ops import roi_align
from rlib import *
ap = argparse.ArgumentParser(); ap.add_argument("--method", default="trustmark"); ap.add_argument("--n", type=int, default=4); ap.add_argument("--K", type=int, default=2)
ap.add_argument("--rad", type=int, default=12); ap.add_argument("--step", type=int, default=3); ap.add_argument("--bs", type=int, default=96); ap.add_argument("--pop", default="pleines"); ap.add_argument("--hi", type=int, default=28); a = ap.parse_args()
dev = "cuda"; wm = load(a.method, dev); rng = np.random.default_rng(0); PAD = 32

@torch.no_grad()
def r3_batch(xp, boxes, msgs):
    """xp : image prolongée de PAD px ; boxes en coordonnées de l'image d'origine (peuvent sortir de [0,512])."""
    out = []; M = torch.from_numpy(msgs).to(dev); k1 = len(msgs)
    for i in range(0, len(boxes), a.bs):
        bb = boxes[i:i + a.bs]; b = torch.tensor([(0, t[1] + PAD, t[0] + PAD, t[3] + PAD, t[2] + PAD) for t in bb], dtype=torch.float32, device=dev)
        cover = roi_align(xp, b, output_size=(256, 256), spatial_scale=1.0, sampling_ratio=2, aligned=True).clamp(0, 1) * 2 - 1
        cov = cover.repeat_interleave(k1, 0); sec = M.repeat(len(bb), 1)
        with torch.autocast("cuda", dtype=torch.float16):
            stego, _ = wm.tm.encoder(cov, sec)
        res = stego.float().clamp(-1, 1) - cov; res = res - res.mean((2, 3), keepdim=True)
        en = res.pow(2).sum((1, 2, 3)).view(len(bb), k1); out.append(en[:, 0] / en[:, 1:].mean(1))
    return torch.cat(out).cpu().numpy()

M_ = [e for e in meta("train") if e["method"] == a.method and e["gidx"] % 5 != 0 and bool(e.get("full_watermark")) == (a.pop == "pleines")][:a.n]
offs = list(range(0, a.hi + 1, a.step)); t0 = time.time()
print("FRAME", a.method, a.pop, "n=%d, cadres testés : marge de 0 à %d px par côté, pas %d (%d cadres)" % (len(M_), a.hi, a.step, len(offs) ** 4), flush=True)
for e in M_:
    x = to_t(read_img("train", e["image_name"]), dev); xp = F.pad(x, (PAD,) * 4, mode="replicate"); m = bits_of(e)
    msgs = np.concatenate([m[None], rng.integers(0, 2, (a.K, wm.nbits)).astype(np.float32)])
    c = [(-d[0], -d[1], 512 + d[2], 512 + d[3]) for d in itertools.product(offs, repeat=4)]
    v = r3_batch(xp, c, msgs); j = int(np.argmin(v)); b = c[j]; best = float(v[j])
    for _ in range(4):                                                # affinage au pixel autour du meilleur
        c2 = list({(b[0] + d[0], b[1] + d[1], b[2] + d[2], b[3] + d[3]) for d in itertools.product((-1, 0, 1), repeat=4)}); v2 = r3_batch(xp, c2, msgs); j2 = int(np.argmin(v2))
        if v2[j2] < best: best = float(v2[j2]); b = c2[j2]
    r0 = float(r3_batch(xp, [(0, 0, 512, 512)], msgs)[0]); mk = read_mask(e["image_name"])
    print("  %s part filigranée %.2f | R3 au cadre image entière %.3f | meilleur cadre %s R3 %.3f | médiane des R3 de la grille %.3f, 2e meilleur éloigné %.3f | %d s" % (
        e["image_name"][:10], mk.mean(), r0, b, best, float(np.median(v)), float(np.sort(v)[min(20, len(v) - 1)]), time.time() - t0), flush=True)
