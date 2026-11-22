"""Diagnostic : le gabarit « cadre image entière » colle-t-il aux vraies images pleines ? Même mesure sur fabriqué (témoin).
Par image : corrélation normalisée rho entre le résidu estimé h et le gabarit e, dans la zone filigranée (dedans) et propre (dehors),
z = écart de rho(vrai message) aux rho de K messages au hasard ; pic de corrélation croisée (décalage en px)."""
import argparse, time
import numpy as np, torch
from rlib import *
ap = argparse.ArgumentParser(); ap.add_argument("--method", required=True); ap.add_argument("--n", type=int, default=8); ap.add_argument("--K", type=int, default=8)
ap.add_argument("--den", default="g2"); a = ap.parse_args()
dev = "cuda"; wm = load(a.method, dev); RM = load_remover(dev) if a.den == "tmrm" else None; rng = np.random.default_rng(0)
tr = meta("train"); mine = [e for e in tr if e["method"] == a.method and e["gidx"] % 5 != 0]
pleines = [e for e in mine if e.get("full_watermark")][:a.n]; part = [e for e in mine if not e.get("full_watermark")][:a.n]
others = [e for e in tr if e["method"] != a.method and e["gidx"] % 5 != 0][:a.n]

def stats(img, m, mask, label):
    x = to_t(img, dev); msgs = np.stack([m] + [rng.integers(0, 2, wm.nbits) for _ in range(a.K)])
    T = tmpl(wm, img, msgs); h = hp(a.den, x, RM); E = expected(a.den, x, T, RM)
    mk = torch.from_numpy(mask).to(dev).float()[None, None]
    out = []
    en = (T * T).sum(1, keepdim=True)                       # R3 : énergie de ce que le tampon ajouterait (vrai message / faux messages)
    for nm, w in (("dedans", mk), ("dehors", 1 - mk)):
        if w.sum() < 500: out.append("%s: -" % nm); continue
        num = (h * E * w).sum((1, 2, 3)); den = ((E * E * w).sum((1, 2, 3)) * (h * h * w).sum()).sqrt() + 1e-20
        rho = (num / den).cpu().numpy(); gain = float(num[0] / ((E[0] * E[0] * w[0]).sum() + 1e-20))
        z = (rho[0] - rho[1:].mean()) / (rho[1:].std() + 1e-12)
        er = float((en[0] * w[0]).sum() / ((en[1:] * w).sum((1, 2, 3)).mean() + 1e-20))
        out.append("%s: rho=%+.4f z=%+6.1f gain=%+.3f R3=%.3f" % (nm, rho[0], z, gain, er))
    # corrélation croisée (somme des canaux) : où est le pic ?
    hh = h.sum(1)[0]; ee = E[0].sum(0); cc = torch.fft.irfft2(torch.fft.rfft2(hh) * torch.fft.rfft2(ee).conj(), s=hh.shape)
    pk = int(cc.abs().argmax()); py, px = pk // cc.shape[1], pk % cc.shape[1]; py = py - 512 if py > 256 else py; px = px - 512 if px > 256 else px
    lg = wm.logits(x)[0].cpu().numpy(); s = 2 * m - 1; zdec = float((s * lg).sum() / np.sqrt((lg ** 2).sum())); acc = float(((lg > 0) == (m > 0)).mean())
    print("  %-10s part=%.2f bits=%.2f zdec=%+5.1f | %s | %s | pic croisé (dy,dx)=(%d,%d) force=%.1f sigma" % (label, mask.mean(), acc, zdec, out[0], out[1], py, px, float(cc.abs().max() / cc.std())), flush=True)

print("DIAG", a.method, "débruiteur", a.den)
print(" fabriqué (fond = vraie image d'un autre filigrane, tampon sur l'image entière, moitié gauche remise propre)")
for e in others[:a.n]:
    img = read_img("train", e["image_name"]); m = rng.integers(0, 2, wm.nbits).astype(np.float32); xw = wm.embed(img, m).copy()
    mask = np.ones((512, 512), np.uint8); mask[:, :256] = 0; xw[:, :256] = img[:, :256]
    stats(xw, m, mask, "fabriqué")
print(" réel, pleines")
for e in pleines: stats(read_img("train", e["image_name"]), bits_of(e), read_mask(e["image_name"]), "pleine")
print(" réel, partielles (cadre image entière = mauvais cadre attendu)")
for e in part: stats(read_img("train", e["image_name"]), bits_of(e), read_mask(e["image_name"]), "partielle")
