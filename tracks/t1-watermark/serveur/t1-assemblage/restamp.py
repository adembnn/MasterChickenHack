"""Carte de re-tamponnage pour les images PLEINES (TrustMark). Fait mesure par t1-residus sur les rectangles : quand on re-tamponne, avec le vrai
message, une image qui porte deja ce tampon dans le MEME cadre, l'encodeur ajoute 2 a 3 fois moins d'energie qu'avec un message au hasard.
Idee : sur une image pleine, le cadre est l'image agrandie en miroir de p px par cote (p inconnu). On cherche p (au pixel) qui minimise le rapport
energie(vrai message) / energie(messages au hasard), puis on regarde ce rapport LOCALEMENT : bas la ou le filigrane est present, proche de 1 dans
les trous propres. Mesure : AUC par image entre cases propres et filigranees. Usage : python restamp.py [n] [--save]"""
import sys, json, time, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); from t1lib import *
sys.path.insert(0, str(HERE / "kit")); from wmloc.watermarks import load_watermark
N = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 16; dev = "cuda"; torch.set_num_threads(4); K = 3; rng = np.random.default_rng(0)
wm = load_watermark("trustmark", models_dir=HERE / "kit" / "models", device=dev)
@torch.no_grad()
def energies(x, pads, msgs):
    """x : 1x3x512x512 dans [0,1] ; pads : liste de (haut, gauche, bas, droite). Renvoie pour chaque toile la carte d'energie 256x256 par message : P x (1+K) x 256 x 256."""
    cov = torch.cat([F.interpolate(F.pad(x, (l, r, t, b), mode="reflect"), size=(256, 256), mode="bilinear", align_corners=False, antialias=True) for (t, l, b, r) in pads]) * 2 - 1
    k1 = len(msgs); out = []
    for i in range(0, len(cov), 16):
        c = cov[i:i + 16].repeat_interleave(k1, 0); sec = msgs.repeat(len(cov[i:i + 16]), 1)
        with torch.autocast("cuda", dtype=torch.float16): stego, _ = wm.tm.encoder(c, sec)
        res = stego.float().clamp(-1, 1) - c; res = res - res.mean((2, 3), keepdim=True); out.append(res.pow(2).sum(1).view(-1, k1, 256, 256))
    return torch.cat(out)
def inner(pad):  # zone de l'image dans la carte 256 de la toile
    t, l, b, r = pad; H, W = 512 + t + b, 512 + l + r; return int(round(t / H * 256)), int(round((t + 512) / H * 256)), int(round(l / W * 256)), int(round((l + 512) / W * 256))
def ratio_global(E, pads):
    out = []
    for e, pad in zip(E, pads): y0, y1, x0, x1 = inner(pad); out.append(float(e[0, y0:y1, x0:x1].sum() / e[1:, y0:y1, x0:x1].mean(0).sum()))
    return np.array(out)
def local_map(e, pad, k=9):
    y0, y1, x0, x1 = inner(pad); a = F.avg_pool2d(e[None, :1], k, 1, k // 2); b = F.avg_pool2d(e[None, 1:].mean(1, keepdim=True), k, 1, k // 2)
    return F.interpolate((a / (b + 1e-12))[:, :, y0:y1, x0:x1], size=(128, 128), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
def auc(s, y):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))
rows = [r for r in load_train("trustmark") if r["full"] and 0.005 < (~r["gt"]).mean()][:N]; res = []; t0 = time.time(); maps = {}
for r in rows:
    x = torch.from_numpy(load_img("train", r["image_name"]).copy()).permute(2, 0, 1)[None].float().div(255).to(dev)
    m = np.array([int(c) for c in r["message"]], np.float32); msgs = torch.from_numpy(np.concatenate([m[None], rng.integers(0, 2, (K, 100)).astype(np.float32)])).to(dev)
    pads = [(p, p, p, p) for p in range(0, 25)]; E = energies(x, pads, msgs); rg = ratio_global(E, pads); p0 = int(np.argmin(rg)); best = (pads[p0], float(rg[p0]))
    for _ in range(6):                                                    # descente au pixel, cote par cote
        t, l, b, rr = best[0]; c = list({(max(0, t + d[0]), max(0, l + d[1]), max(0, b + d[2]), max(0, rr + d[3])) for d in [(a_, 0, 0, 0) for a_ in (-2, -1, 1, 2)] + [(0, a_, 0, 0) for a_ in (-2, -1, 1, 2)] + [(0, 0, a_, 0) for a_ in (-2, -1, 1, 2)] + [(0, 0, 0, a_) for a_ in (-2, -1, 1, 2)]})
        v = ratio_global(energies(x, c, msgs), c); j = int(np.argmin(v))
        if v[j] < best[1] - 1e-4: best = (c[j], float(v[j]))
        else: break
    Eb = energies(x, [best[0], (0, 0, 0, 0)], msgs); clean = ~r["gt"]; mp = local_map(Eb[0], best[0]); m0 = local_map(Eb[1], (0, 0, 0, 0))
    a1, a0 = auc(mp.ravel(), clean.ravel()), auc(m0.ravel(), clean.ravel()); maps[r["image_name"]] = mp.astype(np.float16)
    res.append(dict(name=r["image_name"], pad=best[0], ratio=best[1], ratio_p0=float(rg[0]), ratio_sym=float(rg[p0]), p_sym=p0, auc=a1, auc_p0=a0, trou=float(clean.mean()), r_propre=float(mp[clean].mean()), r_filigrane=float(mp[~clean].mean())))
    print(f"{r['image_name']} trou {100 * clean.mean():4.1f} % | marge symetrique {p0:2d} px, affinee {best[0]} | rapport global {rg[0]:.2f} (sans marge) -> {best[1]:.2f} | rapport local : filigrane {mp[~clean].mean():.2f}, trou {mp[clean].mean():.2f} | AUC {a0:.2f} (sans marge) -> {a1:.2f} | {round(time.time() - t0)} s", flush=True)
A = np.array([r["auc"] for r in res]); A0 = np.array([r["auc_p0"] for r in res])
print(f"\nTrustMark pleines n={len(res)} : AUC moyenne sans marge {A0.mean():.3f} -> avec cadre cale {A.mean():.3f} (mediane {np.median(A):.3f}, part > 0,9 : {(A > .9).mean():.2f}) | marge symetrique mediane {np.median([r['p_sym'] for r in res]):.0f} px | rapport global median {np.median([r['ratio'] for r in res]):.2f}")
if "--save" in sys.argv: np.savez_compressed(ME / "restamp_train_essai.npz", names=np.array(list(maps)), maps=np.stack(list(maps.values()))); json.dump(res, open(ME / "restamp_train_essai.json", "w"))
