"""P5 (critique PixelSeal), étape 1 : un détecteur local VOIT-il un petit rectangle filigrané ? Classifieur de vignettes 64 x 64 px (ResNet-34, première couche en pleine
résolution), entraîné sur fabriqué seul : vignette centrée sur un rectangle fabriqué (positif) contre vignette de la même image sans filigrane (négatif).
Mesure : AUC sur des rectangles fabriqués tenus à l'écart, par petit côté, puis sur les VRAIS rectangles (jamais vus), contre des vignettes propres des mêmes images.
Si l'AUC sur fabriqué reste près de 0,6 pour les petits côtés : limite de signal, inutile de construire le détecteur glissant.
usage : p5.py <filigrane> <suffixe de la réserve d'exemples> [pas]"""
import sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
me, suf = sys.argv[1], sys.argv[2]; NST = int(sys.argv[3]) if len(sys.argv) > 3 else 3000; sys.argv = sys.argv[:1]
import reel11 as R9, segmentation_models_pytorch as smp
dev = "cuda"; rng = np.random.default_rng(0); S = 64; PR = [np.load(R9.ROOT / f"data_{me}{k}" / "RECT.npy", mmap_mode="r") for k in suf.split(",")]; RBOX = np.concatenate([np.load(R9.ROOT / f"data_{me}{k}" / "RBOX.npy") for k in suf.split(",")]); N = len(RBOX); OFF = np.cumsum([0] + [len(p) for p in PR])
class _R:                                               # plusieurs réserves d'exemples vues comme une seule
    def __getitem__(s, r): k = int(np.searchsorted(OFF, r, side="right") - 1); return PR[k][r - OFF[k]]
RECT = _R()
small = np.minimum(RBOX[:, 2], RBOX[:, 3]); te = np.arange(N) % 5 == 0; tr = np.where(~te)[0]
def pos_patch(img, box):                                 # vignette qui recouvre le rectangle au mieux : centre tiré dans le rectangle
    y0, x0, h, w = box; cy = int(rng.integers(y0, y0 + h)); cx = int(rng.integers(x0, x0 + w)); py = int(np.clip(cy - S // 2, 0, 512 - S)); px = int(np.clip(cx - S // 2, 0, 512 - S)); return img[py:py + S, px:px + S]
def neg_patch(img, box):                                 # vignette de la même image sans aucun pixel du rectangle
    y0, x0, h, w = box
    for _ in range(50):
        py = int(rng.integers(0, 512 - S + 1)); px = int(rng.integers(0, 512 - S + 1))
        if py + S <= y0 or py >= y0 + h or px + S <= x0 or px >= x0 + w: return img[py:py + S, px:px + S]
    return None
class Net(nn.Module):
    def __init__(s): super().__init__(); s.e = smp.encoders.get_encoder("resnet34", weights="imagenet"); s.h = nn.Linear(512, 1); s.e.conv1.stride = (1, 1)
    def forward(s, x): return s.h(s.e(x)[-1].mean((2, 3)))[:, 0]
M = torch.tensor([0.485, 0.456, 0.406], device=dev)[None, :, None, None]; Dv = torch.tensor([0.229, 0.224, 0.225], device=dev)[None, :, None, None]
prep = lambda p: (torch.from_numpy(np.ascontiguousarray(p)).to(dev).permute(0, 3, 1, 2).float() / 255 - M) / Dv
def auc(s, t):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1); n1 = t.sum(); n0 = len(t) - n1; return (r[t == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)
torch.manual_seed(0); net = Net().to(dev); opt = torch.optim.AdamW(net.parameters(), 2e-4, weight_decay=1e-4); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-4, total_steps=NST, pct_start=0.1); net.train()
trs = tr[small[tr] < 128]                                # on entraîne sur les petits et moyens rectangles
for step in range(1, NST + 1):
    X, T = [], []
    for r in rng.choice(trs, 32):
        img = np.asarray(RECT[r]); X.append(pos_patch(img, RBOX[r])); T.append(1.0); n = neg_patch(img, RBOX[r])
        if n is not None: X.append(n); T.append(0.0)
    x = prep(np.stack(X))
    if rng.random() < 0.5: x = x.flip(-1)
    loss = F.binary_cross_entropy_with_logits(net(x), torch.tensor(T, device=dev)); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if step % 500 == 0: print("pas", step, "perte", round(float(loss), 4), flush=True)
net.eval()
def score(P):
    with torch.no_grad(): return np.concatenate([net(prep(np.stack(P[j:j + 256]))).cpu().numpy() for j in range(0, len(P), 256)])
print(f"## {me}, rectangles FABRIQUÉS tenus à l'écart (réserve {suf})")
for lo, hi in ((10, 32), (32, 64), (64, 128)):
    sel = np.where(te & (small >= lo) & (small < hi))[0][:300]; P, T = [], []
    for r in sel:
        img = np.asarray(RECT[r]); P.append(pos_patch(img, RBOX[r])); T.append(1); n = neg_patch(img, RBOX[r])
        if n is not None: P.append(n); T.append(0)
    s = score(P); T = np.array(T); print(f"   petit côté {lo:3d} à {hi:3d} px : {len(sel):3d} rectangles | AUC vignette avec rectangle contre vignette propre {auc(s, T):.3f} | détectés à 5 % de fausses alertes {np.mean(s[T == 1] > np.quantile(s[T == 0], 0.95)):.2f}", flush=True)
trm, tem = R9.metas(); RT = np.load(R9.CACHE / "REAL_train.npy", mmap_mode="r"); MT = np.load(R9.CACHE / "MASK_train.npy", mmap_mode="r"); print(f"## {me}, VRAIS rectangles (toutes les partielles dont le masque remplit sa boîte)")
real = []
for i, m in enumerate(trm):
    if m["method"] != me or m["full_watermark"]: continue
    Mk = np.asarray(MT[i]).astype(bool)
    if not Mk.any(): continue
    ys, xs = np.where(Mk); h, w = ys.max() + 1 - ys.min(), xs.max() + 1 - xs.min()
    if Mk.sum() >= 0.98 * h * w: real.append((i, (int(ys.min()), int(xs.min()), int(h), int(w))))
for lo, hi in ((0, 32), (32, 64), (64, 128)):
    P, T = [], []
    for i, box in real:
        if not (lo <= min(box[2], box[3]) < hi): continue
        img = np.asarray(RT[i])
        for _ in range(8):
            P.append(pos_patch(img, box)); T.append(1); n = neg_patch(img, box)
            if n is not None: P.append(n); T.append(0)
    if not P: continue
    s = score(P); T = np.array(T); print(f"   petit côté {lo:3d} à {hi:3d} px : {sum(lo <= min(b[2], b[3]) < hi for _, b in real):3d} vrais rectangles | AUC {auc(s, T):.3f} | détectés à 5 % de fausses alertes {np.mean(s[T == 1] > np.quantile(s[T == 0], 0.95)):.2f}", flush=True)
