"""T1 PixelSeal · décodeur local dense (session t1-pixelseal).
Idée : le tronc ConvNeXt du détecteur PixelSeal (déjà accordé au motif) + pyramide ; à chaque case on prédit les 256 bits
(cible = vrai bit là où c'est filigrané, 0,5 là où c'est propre : ça force la localité), puis l'accord avec le message CONNU
devient un canal qui nourrit la tête de masque 128x128.
Exemples fabriqués à la volée sur GPU, sur fonds réels du hackathon (images TrustMark / StegaStamp, propres pour PixelSeal) :
  plein  : tout filigraner puis recoller l'original dans un trou (formes des vrais trous, déformées) ;
  rect   : un rectangle filigrané dans SON cadre (image source entière), tailles tirées des vrais rectangles ;
  objet  : filigrane au cadre image, gardé dans la forme d'un vrai objet (masque aligné d'une image partielle TM/SS) ;
  forme  : idem avec une forme d'objet posée sur un autre fond.
Réserve = index % 5 == 0 (jamais vue). --fold f (1..4) : ce pli de vraies images PixelSeal est tenu à l'écart aussi (hors-pli)."""
import os, sys, json, argparse, time, copy, math
os.environ.setdefault("OMP_NUM_THREADS", "8")
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
HERE = Path(os.environ.get("WMLOC_ROOT", "/p/scratch/training2666/destors1/MasterChickenHack/Hackathon_Watermark_Localisation_Task"))
sys.path.insert(0, str(HERE / "kit"))
from wmloc.watermarks import load_watermark
from wmloc.data import LocalisationDataset
from wmloc.utils import miou_image

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True); ap.add_argument("--fold", type=int, default=1); ap.add_argument("--real", type=int, default=4)
ap.add_argument("--bitw", type=float, default=1.0); ap.add_argument("--res", type=int, default=256); ap.add_argument("--steps", type=int, default=5000)
ap.add_argument("--bs", type=int, default=28); ap.add_argument("--lr", type=float, default=4e-4); ap.add_argument("--trunk_lr", type=float, default=0.15)
ap.add_argument("--coco_n", type=int, default=1500); ap.add_argument("--eval_every", type=int, default=500); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--agree", type=int, default=1); ap.add_argument("--margin", type=int, default=16); ap.add_argument("--lovasz", type=float, default=0.0); ap.add_argument("--fpn", type=int, default=160); ap.add_argument("--hr", type=int, default=0); ap.add_argument("--kinds", default="tout"); ap.add_argument("--gain", type=float, default=1.0); ap.add_argument("--eval_only", type=int, default=0); ap.add_argument("--ppinit", type=int, default=1); ap.add_argument("--dev", default="cuda"); ap.add_argument("--amp", type=int, default=1)
ap.add_argument("--coco", default=str(HERE.parent / "shared_data" / "coco" / "val2017")); ap.add_argument("--out", default=str(HERE / "adem" / "t1-pixelseal"))
a = ap.parse_args()
torch.set_num_threads(8); dev = a.dev; rng = np.random.default_rng(a.seed); torch.manual_seed(a.seed)
OUT = Path(a.out) / f"run_{a.tag}"; OUT.mkdir(parents=True, exist_ok=True)
def log(*x): print(a.tag, *x, flush=True)

# ---------------------------------------------------------------- données réelles
ds = LocalisationDataset(HERE / "release", "train"); meta = ds.meta
def load_img(i): return np.asarray(Image.open(ds.dir / "images" / f"{meta[i]['image_name']}.png").convert("RGB"))
def load_mask(i): return np.asarray(Image.open(ds.dir / "masks" / f"{meta[i]['image_name']}.png")) > 127
ps = [i for i, m in enumerate(meta) if m["method"] == "pixelseal"]
role = np.array([0 if i % 5 == 0 else (1 if i % 5 == a.fold else 2) for i in ps])        # 0 réserve, 1 pli tenu à l'écart, 2 entraînement
PX = torch.from_numpy(np.stack([load_img(i) for i in ps]))                               # 170 x 512 x 512 x 3 uint8 (CPU)
PM = torch.from_numpy(np.stack([load_mask(i) for i in ps]))                              # masques 512
PB = torch.tensor([[int(c) for c in meta[i]["message"]] for i in ps])
PG = np.load(ds.dir / "masks_128.npz"); PG = dict(zip(PG["names"].tolist(), PG["masks"])); PG = np.stack([PG[meta[i]["image_name"]] for i in ps]).astype(bool)
pfull = np.array([bool(meta[i].get("full_watermark")) for i in ps])
tr_idx = np.where((role == 2) & (pfull if a.kinds == 'plein' else ~pfull if a.kinds == 'partiel' else True))[0]
BG, HOLES, SHAPES, OBJ, RECTS = [], [], [], [], []
def is_rect(mk):
    ys, xs = np.where(mk); return mk.sum() / ((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)) >= 0.98, (ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)
for i, m in enumerate(meta):
    if i % 5 == 0 or (m["method"] == "pixelseal" and i % 5 == a.fold): continue             # ni la réserve, ni le pli tenu à l'écart
    mk = load_mask(i)
    if m["method"] != "pixelseal": BG.append(load_img(i))
    if m.get("full_watermark"): HOLES.append(~mk)
    else:
        r, hw = is_rect(mk)
        if r:
            if m["method"] != "trustmark": RECTS.append(hw)
        else:
            SHAPES.append(mk)
            if m["method"] == "trustmark": OBJ.append((len(BG) - 1, len(SHAPES) - 1))
n_real_bg = len(BG)
if a.coco_n and Path(a.coco).exists():
    files = sorted(Path(a.coco).glob("*.jpg")); r2 = np.random.default_rng(1); r2.shuffle(files)
    for p in files[:a.coco_n]:
        im = Image.open(p).convert("RGB"); w, h = im.size; s = min(w, h); l = int(r2.integers(0, w - s + 1)); t = int(r2.integers(0, h - s + 1))
        BG.append(np.asarray(im.crop((l, t, l + s, t + s)).resize((512, 512), Image.Resampling.LANCZOS)))
BG = torch.from_numpy(np.stack(BG)).to(dev); HOLES = torch.from_numpy(np.stack(HOLES)).to(dev); SHAPES = torch.from_numpy(np.stack(SHAPES)).to(dev)
log("fonds", len(BG), "dont réels", n_real_bg, "| trous", len(HOLES), "| formes", len(SHAPES), "| objets alignés", len(OBJ), "| rectangles", len(RECTS),
    "| vraies PS : entraînement", len(tr_idx), "pli", int((role == 1).sum()), "réserve", int((role == 0).sum()))

# ---------------------------------------------------------------- fabrication à la volée
wm = load_watermark("pixelseal", models_dir=HERE / "kit" / "models", device=dev)
@torch.no_grad()
def embed(x, bits):                                   # x B x 3 x H x W dans [0,1] ; arrondi 8 bits comme wm.embed ; GAIN > 1 = résidu amplifié (début d'entraînement)
    o = wm.model.embed(x, msgs=bits.float(), is_video=False)["imgs_w"]
    if GAIN != 1.0: o = x + GAIN * (o - x)
    return (o.clamp(0, 1) * 255).round() / 255
GAIN = 1.0
PK = {"tout": (0.50, 0.70, 0.85), "plein": (1.0, 1.0, 1.0), "partiel": (0.0, 0.45, 0.75)}[a.kinds]
def jit(m, big):                                      # déformation affine d'une forme 512x512 (bool)
    ang = rng.uniform(0, 2 * math.pi) if big else rng.uniform(-0.3, 0.3); s = math.exp(rng.uniform(-0.5, 0.4) if big else rng.uniform(-0.3, 0.3))
    fl = -1.0 if rng.random() < 0.5 else 1.0; t = 0.6 if big else 0.4
    th = torch.tensor([[fl * math.cos(ang) / s, -math.sin(ang) / s, rng.uniform(-t, t)], [fl * math.sin(ang) / s, math.cos(ang) / s, rng.uniform(-t, t)]], device=dev, dtype=torch.float32)
    g = F.affine_grid(th[None], (1, 1, 512, 512), align_corners=False)
    return F.grid_sample(m[None, None].float(), g, mode="nearest", padding_mode="zeros", align_corners=False)[0, 0] > 0.5
@torch.no_grad()
def gen(n, force=None):
    bits = torch.randint(0, 2, (n, 256), device=dev); x0 = torch.empty(n, 3, 512, 512, device=dev); msk = torch.zeros(n, 512, 512, dtype=torch.bool, device=dev); kind = []
    for j in range(n):
        r = rng.random(); k = force or ("plein" if r < PK[0] else "rect" if r < PK[1] else "objet" if r < PK[2] else "forme"); m = None
        if k == "objet":
            bi, si = OBJ[int(rng.integers(len(OBJ)))]; b = BG[bi]; m = SHAPES[si].float()[None]
        else:
            b = BG[int(rng.integers(n_real_bg)) if (rng.random() < 0.7 or len(BG) == n_real_bg) else int(rng.integers(n_real_bg, len(BG)))]
        b = b.permute(2, 0, 1).float()
        if rng.random() < 0.5:
            b = b.flip(-1); m = m.flip(-1) if m is not None else None
        if rng.random() < 0.5:
            s = int(512 * math.sqrt(rng.uniform(0.5, 1))); y = int(rng.integers(0, 512 - s + 1)); x = int(rng.integers(0, 512 - s + 1))
            b = F.interpolate(b[None, :, y:y + s, x:x + s], size=512, mode="bilinear", align_corners=False)[0].round()
            if m is not None: m = F.interpolate(m[None, :, y:y + s, x:x + s], size=512, mode="nearest")[0]
        x0[j] = b / 255
        if k == "objet":
            msk[j] = m[0] > 0.5
            if msk[j].float().mean() < 0.003: k = "forme"
        kind.append(k)
    p = int(rng.integers(0, a.margin + 1))               # toile plus grande de p px par côté (miroir), filigranée puis rognée (découverte t1-reel)
    xw = embed(F.pad(x0, (p, p, p, p), mode="reflect"), bits)[:, :, p:p + 512, p:p + 512] if p else embed(x0, bits); x = x0.clone()
    for j, k in enumerate(kind):
        if k == "plein":
            h = jit(HOLES[int(rng.integers(len(HOLES)))], True)
            if rng.random() < 0.2: h = h | jit(HOLES[int(rng.integers(len(HOLES)))], True)
            if h.float().mean() > 0.5: h = jit(HOLES[int(rng.integers(len(HOLES)))], True) & h
            msk[j] = ~h; x[j] = torch.where(h[None], x0[j], xw[j])
        elif k == "rect":
            h, w = RECTS[int(rng.integers(len(RECTS)))]
            h = int(min(512, max(8, h * math.exp(rng.uniform(-0.35, 0.35))))); w = int(min(512, max(8, w * math.exp(rng.uniform(-0.35, 0.35)))))
            y = int(rng.integers(0, 512 - h + 1)); xx = int(rng.integers(0, 512 - w + 1))
            q = int(round(rng.uniform(0, 0.03) * min(h, w))) if a.margin else 0     # rectangle : marge proportionnelle (0 à 3 % du petit côté)
            c = x0[j:j + 1, :, y:y + h, xx:xx + w]
            x[j, :, y:y + h, xx:xx + w] = (embed(F.pad(c, (q, q, q, q), mode="reflect"), bits[j:j + 1])[:, :, q:q + h, q:q + w] if q else embed(c, bits[j:j + 1]))[0]; msk[j, y:y + h, xx:xx + w] = True
        else:
            if k == "forme": msk[j] = jit(SHAPES[int(rng.integers(len(SHAPES)))], False)
            x[j] = torch.where(msk[j][None], xw[j], x0[j])
    gen.kinds = kind; return x, msk, bits
def real_batch(n):
    idx = tr_idx[rng.integers(0, len(tr_idx), n)]; x = PX[idx].to(dev).permute(0, 3, 1, 2).float() / 255; m = PM[idx].to(dev); b = PB[idx].to(dev)
    xs, ms = [], []
    for j in range(n):                                # décalage de -16..16 px par pas de 8 (bords répétés) ; pas de retournement : le motif n'y survit pas
        dy, dx = (int(rng.integers(-2, 3)) * 8 for _ in range(2))
        xs.append(F.pad(x[j:j + 1], (16, 16, 16, 16), mode="replicate")[:, :, 16 + dy:528 + dy, 16 + dx:528 + dx])
        ms.append(F.pad(m[j:j + 1, None].float(), (16, 16, 16, 16), mode="replicate")[:, 0, 16 + dy:528 + dy, 16 + dx:528 + dx] > 0.5)
    return torch.cat(xs), torch.cat(ms), b

# ---------------------------------------------------------------- modèle
sys.path.insert(0, str(Path(__file__).resolve().parent))
from locnet import Loc
det = copy.deepcopy(wm.det).float()
for p in det.parameters(): p.requires_grad_(True)
net = Loc(det, d=a.fpn, res=a.res, agree=a.agree, ppinit=a.ppinit, hr=a.hr).to(dev); net.train(); net.det.eval()               # tronc en eval : pas de blocs sautés au hasard
head_params = [p for n_, p in net.named_parameters() if not n_.startswith("det.")]
opt = torch.optim.AdamW([{"params": net.det.parameters(), "lr": a.lr * a.trunk_lr}, {"params": head_params, "lr": a.lr}], weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr * a.trunk_lr, a.lr], total_steps=a.steps, pct_start=0.06)
AMP = dict(device_type="cuda", dtype=torch.bfloat16, enabled=bool(a.amp) and "cuda" in dev)

def lovasz(m, y):                                      # approximation dérivable de l'IoU (Lovász hinge), par image, pour les deux classes
    tot = 0.0
    for sgn_, yy in ((1.0, y), (-1.0, 1 - y)):
        lg = (sgn_ * m).flatten(1); t = yy.flatten(1); err = 1 - lg * (2 * t - 1); es, perm = torch.sort(err, dim=1, descending=True); ts = torch.gather(t, 1, perm)
        inter = ts.sum(1, keepdim=True) - ts.cumsum(1); union = ts.sum(1, keepdim=True) + (1 - ts).cumsum(1); jac = 1 - inter / union
        jac = torch.cat([jac[:, :1], jac[:, 1:] - jac[:, :-1]], 1); tot = tot + (F.relu(es) * jac).sum(1).mean()
    return tot / 2
def losses(m, L, mask512, bits):
    y = (F.avg_pool2d(mask512[:, None].float(), 4)[:, 0] >= 0.5).float()                   # règle officielle 512 -> 128
    p = torch.sigmoid(m); i1 = (p * y).sum((1, 2)); i0 = ((1 - p) * (1 - y)).sum((1, 2))
    dice = 1 - 0.5 * ((2 * i1 + 1) / (p.sum((1, 2)) + y.sum((1, 2)) + 1) + (2 * i0 + 1) / ((1 - p).sum((1, 2)) + (1 - y).sum((1, 2)) + 1))
    lm = F.binary_cross_entropy_with_logits(m, y) + dice.mean()
    if a.lovasz: lm = lm + a.lovasz * lovasz(m, y)
    g = L.shape[-1]; cell = (F.avg_pool2d(mask512[:, None].float(), 512 // g) > 0.5).float()   # B x 1 x g x g
    tgt = 0.5 + (bits[:, :, None, None].float() - 0.5) * cell
    lb = F.binary_cross_entropy_with_logits(L, tgt.expand_as(L))
    return lm, lb, y, cell
def auc(s, g):
    s = s.ravel(); g = g.ravel(); n1 = g.sum(); n0 = g.size - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = np.empty(s.size); r[np.argsort(s, kind="stable")] = np.arange(1, s.size + 1); return (r[g].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
@torch.no_grad()
def predict(X, B, bs=17):                             # X N x 512 x 512 x 3 uint8 (CPU), B N x 256
    net.eval(); out, agr = [], []
    for k in range(0, len(X), bs):
        x = X[k:k + bs].to(dev).permute(0, 3, 1, 2).float() / 255; sg = B[k:k + bs].to(dev).float() * 2 - 1
        with torch.autocast(**AMP): m, L, a1 = net(x, sg)
        out.append(m.float().cpu()); agr.append(a1.float().cpu())
    net.train(); net.det.eval(); return torch.cat(out).numpy(), torch.cat(agr).numpy()
def evaluate():
    lg, agr = predict(PX, PB); res = {}
    sc = np.array([miou_image(lg[j] > 0, PG[j]) for j in range(len(lg))]); au = np.array([auc(lg[j], PG[j]) for j in range(len(lg))])
    g8 = torch.from_numpy(PG)[:, None].float(); g = agr.shape[-1]; cell = (F.avg_pool2d(g8, 128 // g) > 0.5)[:, 0].numpy() if g < 128 else PG
    aa = np.array([auc(agr[j], cell[j]) for j in range(len(lg))])
    for nm, sel in (("reserve", role == 0), ("pli", role == 1), ("appris", role == 2)):
        if sel.sum() == 0: continue
        res[nm] = {"miou": round(float(sc[sel].mean()), 4), "pleines": round(float(sc[sel & pfull].mean()), 4), "partielles": round(float(sc[sel & ~pfull].mean()), 4),
                   "auc": round(float(np.nanmean(au[sel])), 3), "auc_accord": round(float(np.nanmean(aa[sel])), 3)}
    return res, lg

(OUT / "args.json").write_text(json.dumps(vars(a))) if not a.eval_only else None
if a.eval_only:                                       # mesures par type d'exemple fabriqué + test du raccourci 'objet'
    net.load_state_dict(torch.load(OUT / "net.pt", map_location=dev)); net.eval()
    def run(x, bits):
        with torch.no_grad(), torch.autocast(**AMP): return net(x, bits.float() * 2 - 1)[0].float()
    for kind in ("plein", "rect", "objet", "forme"):
        sc, au, fr = [], [], []
        for _ in range(6):
            x, mk, bits = gen(16, force=kind); y = (F.avg_pool2d(mk[:, None].float(), 4)[:, 0] >= 0.5).cpu().numpy(); lg = run(x, bits).cpu().numpy()
            sc += [miou_image(lg[j] > 0, y[j]) for j in range(16)]; au += [auc(lg[j], y[j]) for j in range(16)]; fr += [y[j].mean() for j in range(16)]
        fr = np.array(fr); sc = np.array(sc); base = np.maximum(fr, 1 - fr) / 2
        log(f"FAB {kind:6s} mIoU {np.mean(sc):.4f} (prédiction constante : {base.mean():.4f}) AUC {np.nanmean(au):.3f} part filigranée médiane {np.median(fr):.3f}")
        if kind == "rect":
            for nm, sel in (("petits rect (<3 %)", fr < 0.03), ("grands rect (>=3 %)", fr >= 0.03)): log(f"   {nm} n={sel.sum()} mIoU {sc[sel].mean():.4f} constante {base[sel].mean():.4f}")
        if kind == "plein":
            for nm, sel in (("petit trou (<6 %)", fr > 0.94), ("grand trou", fr <= 0.94)): log(f"   {nm} n={sel.sum()} mIoU {sc[sel].mean():.4f} constante {base[sel].mean():.4f}")
    ious, fracs = [], []                                # raccourci : objets TM/SS SANS PixelSeal ; la prédiction recouvre-t-elle la forme ?
    for bi, si in OBJ[:40]:
        x = BG[bi].permute(2, 0, 1)[None].float() / 255; lg = run(x, torch.randint(0, 2, (1, 256), device=dev))[0] > 0
        y = F.avg_pool2d(SHAPES[si][None, None].float(), 4)[0, 0] >= 0.5; ious.append(float((lg & y).sum() / (lg | y).sum().clamp(min=1))); fracs.append(float(lg.float().mean()))
    log(f"RACCOURCI objets TM/SS sans PixelSeal : IoU prédiction/forme moyen {np.mean(ious):.3f}, part prédite filigranée {np.mean(fracs):.3f}")
    lg, _ = predict(PX, PB); sc = np.array([miou_image(lg[j] > 0, PG[j]) for j in range(len(lg))]); prect = np.array([(not pfull[j]) and is_rect(PM[j].numpy())[0] for j in range(len(lg))])
    for nm, sel in (("pleines", pfull), ("rectangles", prect), ("objets", ~pfull & ~prect)):
        s2 = sel & (role != 2); fr = PG[s2].reshape(s2.sum(), -1).mean(1); log(f"RÉEL non appris {nm:10s} n={s2.sum():3d} mIoU {sc[s2].mean():.4f} (constante : {(np.maximum(fr, 1 - fr) / 2).mean():.4f})")
    sys.exit(0)
# ---------------------------------------------------------------- entraînement
t0 = time.time(); acc = np.zeros(4); hist = []; ns = a.bs - a.real; bykind = {}
for step in range(1, a.steps + 1):
    GAIN = 1.0 + (a.gain - 1.0) * max(0.0, 1 - step / (0.35 * a.steps))
    x, mk, bits = gen(ns)
    if a.real:
        xr, mr, br = real_batch(a.real); x = torch.cat([x, xr]); mk = torch.cat([mk, mr]); bits = torch.cat([bits, br])
    sg = bits.float() * 2 - 1
    with torch.autocast(**AMP): m, L, a1 = net(x, sg)
    lm, lb, y, cell = losses(m, L, mk, bits); loss = lm + a.bitw * lb
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0); opt.step(); sched.step()
    with torch.no_grad():
        okm = ((L[:ns] > 0) == bits[:ns, :, None, None].bool()).float().mean(1); c = cell[:ns, 0]; okb = (okm * c).sum() / c.sum().clamp(min=1)
        inter = ((m[:ns] > 0) & (y[:ns] > 0.5)).sum((1, 2)).float(); uni = ((m[:ns] > 0) | (y[:ns] > 0.5)).sum((1, 2)).float().clamp(min=1)
        i0 = ((m[:ns] <= 0) & (y[:ns] <= 0.5)).sum((1, 2)).float(); u0 = ((m[:ns] <= 0) | (y[:ns] <= 0.5)).sum((1, 2)).float().clamp(min=1)
        mi = 0.5 * (inter / uni + i0 / u0); acc += np.array([float(lm), float(lb), float(okb), float(mi.mean())])
        for kk, vv in zip(gen.kinds, mi.tolist()): bykind.setdefault(kk, []).append(vv)
    if step % a.eval_every == 0 or step == a.steps:
        res, lg = evaluate(); k = a.eval_every if step % a.eval_every == 0 else step % a.eval_every
        ev = {"step": step, "sec": round(time.time() - t0), "perte_masque": round(acc[0] / k, 4), "perte_bits": round(acc[1] / k, 4),
              "fab_bits_locaux": round(acc[2] / k, 4), "fab_miou": round(acc[3] / k, 4), "gain": round(GAIN, 2),
              "fab": {kk: round(float(np.mean(vv)), 3) for kk, vv in bykind.items()}, **res}
        acc[:] = 0; bykind = {}; hist.append(ev); log("EVAL", json.dumps(ev))
(OUT / "hist.json").write_text(json.dumps(hist, indent=1))
torch.save({k: v for k, v in net.state_dict().items()}, OUT / "net.pt")
names = np.array([meta[i]["image_name"] for i in ps])
np.savez_compressed(OUT / "pred_train.npz", names=names, logits=lg.astype(np.float16), role=role)
if (HERE / "release" / "test" / "metadata.json").exists():
    tm = [m for m in json.loads((HERE / "release" / "test" / "metadata.json").read_text()) if m["method"] == "pixelseal"]
    TX = torch.from_numpy(np.stack([np.asarray(Image.open(HERE / "release" / "test" / "images" / f"{m['image_name']}.png").convert("RGB")) for m in tm]))
    TB = torch.tensor([[int(c) for c in m["message"]] for m in tm]); tl, _ = predict(TX, TB)
    np.savez_compressed(OUT / "pred_test.npz", names=np.array([m["image_name"] for m in tm]), logits=tl.astype(np.float16))
    log("test", tl.shape, "fraction prédite filigranée", round(float((tl > 0).mean()), 3))
log("RESULT", json.dumps(hist[-1]))
