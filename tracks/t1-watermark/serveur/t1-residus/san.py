"""Étape 1 : contrôle de bon sens sur images FABRIQUÉES (original connu).
Fonds = vraies images du hackathon des deux AUTRES filigranes (propres pour celui-ci).
Questions : (a) mon gabarit flottant reproduit-il wm.embed ? (b) le gabarit calculé depuis l'image DÉJÀ filigranée
ressemble-t-il au vrai résidu ? (c) dans quelle bande de fréquence est le résidu, et y est-il plus fort que l'image ?"""
import sys, argparse, json, time
import numpy as np, torch
from rlib import *

ap = argparse.ArgumentParser(); ap.add_argument("--method", required=True); ap.add_argument("--n", type=int, default=30); a = ap.parse_args()
dev = "cuda"; wm = load(a.method, dev); rng = np.random.default_rng(1)
others = [e for e in meta("train") if e["method"] != a.method and e["gidx"] % 5 != 0]
others = [others[i] for i in rng.permutation(len(others))[:a.n]]
BANDS = [("<1px", 0, 1), ("1-2", 1, 2), ("2-4", 2, 4), ("4-8", 4, 8), ("8-16", 8, 16), (">16", 16, 0)]
def band(x, lo, hi):
    a_ = gauss(x, lo) if lo else x; b_ = gauss(x, hi) if hi else torch.zeros_like(x); return a_ - b_
acc = {k: [] for k in ("cos_t0_vrai", "cos_tw_vrai", "cos_twfaux_vrai", "cos_tw_t0", "cos_t0_t0faux", "rms_vrai", "rms_t0", "rms_tw", "frac_nul_uint8", "bits_xw")}
bacc = {b[0]: {k: [] for k in ("part_energie", "cos_tw_vrai", "cos_t0_vrai", "rms_res", "rms_img", "rapport")} for b in BANDS}
t0_ = time.time()
for e in others:
    img = read_img("train", e["image_name"]); m = rng.integers(0, 2, wm.nbits); m2 = rng.integers(0, 2, wm.nbits)
    xw = wm.embed(img, m); x0t = to_t(img, dev); xwt = to_t(xw, dev)
    r = xwt - x0t                                                    # vrai résidu (arrondi 8 bits compris)
    t0 = tmpl(wm, img, np.stack([m, m2])); tw = tmpl(wm, xw, np.stack([m, m2]))
    acc["cos_t0_vrai"].append(cos(t0[0], r)); acc["cos_tw_vrai"].append(cos(tw[0], r)); acc["cos_twfaux_vrai"].append(cos(tw[1], r))
    acc["cos_tw_t0"].append(cos(tw[0], t0[0])); acc["cos_t0_t0faux"].append(cos(t0[0], t0[1]))
    acc["rms_vrai"].append(float(r.pow(2).mean().sqrt()) * 255); acc["rms_t0"].append(float(t0[0].pow(2).mean().sqrt()) * 255)
    acc["rms_tw"].append(float(tw[0].pow(2).mean().sqrt()) * 255); acc["frac_nul_uint8"].append(float((r == 0).float().mean()))
    lg = wm.logits(xwt); acc["bits_xw"].append(float(((lg[0].cpu() > 0).numpy() == m).mean()))
    etot = float(r.pow(2).sum())
    for nm, lo, hi in BANDS:
        rb = band(r, lo, hi); xb = band(x0t, lo, hi); d = bacc[nm]
        d["part_energie"].append(float(rb.pow(2).sum()) / etot); d["cos_tw_vrai"].append(cos(band(tw[:1], lo, hi), rb)); d["cos_t0_vrai"].append(cos(band(t0[:1], lo, hi), rb))
        d["rms_res"].append(float(rb.pow(2).mean().sqrt()) * 255); d["rms_img"].append(float(xb.pow(2).mean().sqrt()) * 255)
        d["rapport"].append(d["rms_res"][-1] / (d["rms_img"][-1] + 1e-9))
res = {"method": a.method, "n": len(others), "sec": round(time.time() - t0_), "global": {k: round(float(np.median(v)), 4) for k, v in acc.items()},
       "bandes": {b: {k: round(float(np.median(v)), 4) for k, v in d.items()} for b, d in bacc.items()}}
(OUT / f"san_{a.method}.json").write_text(json.dumps(res, indent=1))
print("SANITY", a.method, "n=%d" % len(others), "(médianes ; rms en niveaux /255)")
for k, v in res["global"].items(): print(f"  {k:18s} {v}")
print("  bande(px)  part_energie  cos(tw,vrai)  cos(t0,vrai)  rms_residu  rms_image  rapport")
for b, d in res["bandes"].items(): print(f"  {b:9s}  {d['part_energie']:.3f}         {d['cos_tw_vrai']:.3f}         {d['cos_t0_vrai']:.3f}         {d['rms_res']:.3f}       {d['rms_img']:.3f}      {d['rapport']:.4f}")
