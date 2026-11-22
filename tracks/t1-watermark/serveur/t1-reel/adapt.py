"""Piste 3 du vérificateur : adaptation au test, image par image (PixelSeal, partielles prédites).
Pour une image : on part du modèle f_st du pli qui ne l'a pas vue ; on fabrique K exemples À PARTIR DE CETTE IMAGE en ré-incrustant PixelSeal avec son message connu
dans une zone tirée au hasard (rectangle de petit côté 12 à 128 px, bandes comprises, ou tache lisse), filigrane calculé sur une toile élargie comme dans le réel ;
étiquette = zone ajoutée OU cases que le modèle de départ juge déjà filigranées (logit > 0) ; les cases incertaines (-2 < logit <= 0) hors de la zone ajoutée
ne comptent pas dans la perte (on ne sait pas si la vraie zone y est). Puis N pas d'affinage, et prédiction de l'image. Comparaison au même modèle sans adaptation.
usage : adapt.py <nb réserve> <nb hors réserve> [K] [pas] [lr]"""
import sys, json, copy, argparse, time, numpy as np, torch, torch.nn.functional as F
NR, NO = int(sys.argv[1]), int(sys.argv[2]); K = int(sys.argv[3]) if len(sys.argv) > 3 else 64; NS = int(sys.argv[4]) if len(sys.argv) > 4 else 150; LR = float(sys.argv[5]) if len(sys.argv) > 5 else 5e-5
sys.argv = sys.argv[:1]
import reel12 as R9
from wmloc.watermarks import load_watermark
me, tag = "pixelseal", "f_st"; ROOT = R9.ROOT; dev = "cuda"; rng = np.random.default_rng(0)
trm, tem = R9.metas(); RT = np.load(R9.CACHE / "REAL_train.npy", mmap_mode="r"); Y128 = np.load(R9.CACHE / "Y128_train.npy")
idx, res, rest, fold = R9.splits(trm, me)
a = argparse.Namespace(**json.loads((ROOT / "runs" / tag / me / "fold0" / "args.json").read_text()))
for k, v in dict(npl=0, pl="", hsmall=0.0, hlo=0.005, hhi=0.04, cf=0.0, rsmall=0.0, rband=0.25, wedge=0.0).items(): setattr(a, k, getattr(a, k, v))
wm = load_watermark(me, models_dir=R9.HERE / "kit" / "models", device="cuda")
def embed_pad(img, bits, py, px):
    H, W = img.shape[:2]; py, px = int(min(py, H - 1)), int(min(px, W - 1))
    if py + px == 0: return wm.embed(np.ascontiguousarray(img), bits)
    c = np.pad(img, ((py, py), (px, px), (0, 0)), mode="reflect"); return wm.embed(np.ascontiguousarray(c), bits)[py:py + H, px:px + W]
def blob(lo, hi):
    area = float(np.exp(rng.uniform(np.log(lo), np.log(hi)))); g = int(rng.integers(3, 11))
    up = F.interpolate(torch.randn(1, 1, g, g), size=(512, 512), mode="bicubic", align_corners=False)[0, 0].numpy(); return up > np.quantile(up, 1 - area)
def make(img, bits):
    X, M = [], []
    full = None
    for k in range(K):
        if rng.random() < 0.7:                            # rectangle collé, filigrane calculé sur le rectangle et une marge de 2 à 6 px
            s = int(np.exp(rng.uniform(np.log(12), np.log(128)))); r = np.exp(rng.uniform(0, np.log(16))) if rng.random() < 0.3 else np.exp(rng.uniform(0, np.log(3)))
            h, w = (s, int(min(512, s * r))) if rng.random() < 0.5 else (int(min(512, s * r)), s); y0 = int(rng.integers(0, 512 - h + 1)); x0 = int(rng.integers(0, 512 - w + 1))
            q = int(rng.integers(2, 7)); x = img.copy(); x[y0:y0 + h, x0:x0 + w] = embed_pad(img[y0:y0 + h, x0:x0 + w], bits, q, q); m = np.zeros((512, 512), bool); m[y0:y0 + h, x0:x0 + w] = True
        else:                                             # objet : image entière filigranée sur une toile élargie, gardée sur une tache
            if full is None or rng.random() < 0.3: sz = int(rng.integers(8, 100)); full = embed_pad(img, bits, sz // 2, sz // 2)
            m = blob(0.005, 0.2); x = np.where(m[..., None], full, img)
        X.append(x); M.append(m)
    return np.stack(X), np.stack(M)
def load(f):
    R = R9.Runner(a, dev); R.net.load_state_dict(torch.load(ROOT / "runs" / tag / me / f"fold{f}" / "ema.pt", map_location=dev)["state"]); R.net.eval(); return R
def predict(R, net, x):
    with torch.no_grad(): return R.predict(net, x[None], [0.0], tta=2)[0]
sel_o = [i for i in rest if not trm[i]["full_watermark"]]; sel_r = [i for i in res if not trm[i]["full_watermark"]]
sel = [(i, fold[i], "hors réserve") for i in list(rng.permutation(sel_o))[:NO]] + [(i, 0, "réserve") for i in list(rng.permutation(sel_r))[:NR]]
base = {f: load(f) for f in sorted({f for _, f, _ in sel})}; out = []; t0 = time.time()
for n, (i, f, grp) in enumerate(sel):
    img = np.asarray(RT[i]); bits = np.array([int(c) for c in trm[i]["message"]]); R = base[f]; L0 = predict(R, R.net, img)
    X, M = make(img, bits); M128 = torch.from_numpy(M).float().reshape(K, 128, 4, 128, 4).mean((2, 4)).to(dev) >= 0.5
    pos0 = torch.from_numpy(L0 > 0).to(dev); unc = torch.from_numpy((L0 > -2) & (L0 <= 0)).to(dev)
    Yt = (M128 | pos0[None]).float(); W = (~(unc[None] & ~M128)).float()
    net = copy.deepcopy(R.net); net.train(); opt = torch.optim.AdamW(net.parameters(), LR, weight_decay=1e-4); Xg = torch.from_numpy(X).to(dev).permute(0, 3, 1, 2).float()
    for st in range(NS):
        b = torch.from_numpy(rng.choice(K, 8)).to(dev); x = Xg[b]; y = Yt[b]; w = W[b]
        if rng.random() < 0.5: x, y, w = x.flip(-1), y.flip(-1), w.flip(-1)
        lg = R.fwd(net, x, torch.zeros(len(b), device=dev)); loss = (F.binary_cross_entropy_with_logits(lg, y, reduction="none") * w).sum() / w.sum().clamp_min(1)
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); L1 = predict(R, net, img); y = Y128[i].astype(bool)
    s0, s1 = R9.miou_image(L0 > 0, y), R9.miou_image(L1 > 0, y); out.append((grp, s0, s1))
    print(f"{n + 1:3d}/{len(sel)} {grp:12s} {trm[i]['image_name']} zone {y.mean():.3f} | sans adaptation {s0:.3f} | adapté {s1:.3f} | {time.time() - t0:.0f} s", flush=True)
for grp in ("hors réserve", "réserve"):
    v = np.array([(s0, s1) for g, s0, s1 in out if g == grp])
    if len(v): d = v[:, 1] - v[:, 0]; bt = d[rng.integers(0, len(d), (10000, len(d)))].mean(1); print(f"## {grp} : {len(v)} partielles | sans {v[:, 0].mean():.4f} | adapté {v[:, 1].mean():.4f} | écart {d.mean():+.4f} IC95 [{np.percentile(bt, 2.5):+.4f} ; {np.percentile(bt, 97.5):+.4f}] | {int((d > .001).sum())} mieux, {int((d < -.001).sum())} pires")
