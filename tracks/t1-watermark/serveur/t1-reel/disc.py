"""Piste 3 : un petit classifieur sépare-t-il les VRAIES pleines des pleines FABRIQUÉES, et surtout au niveau du trou ?
Trois types de vignettes 64 x 64 px, mêmes règles des deux côtés : « bord » (centrée sur la frontière du trou), « dedans » (au moins 90 % dans le trou),
« loin » (entièrement en zone filigranée : témoin, mesure ce qui sépare réel et fabriqué SANS rapport avec le trou : fond d'image, toile).
Lecture : si « bord » et « dedans » ne dépassent pas « loin », rien de propre au trou ne trahit le fabriqué. AUC par vignette et par image, 5 plis par image.
usage : disc.py <filigrane> <suffixe du jeu fabriqué>"""
import sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
me, suf = sys.argv[1], sys.argv[2]; S1 = len(sys.argv) > 3 and sys.argv[3] == "s1"; NST = 1000 if S1 else 400; sys.argv = sys.argv[:1]
import reel9 as R9, segmentation_models_pytorch as smp
dev = "cuda"; rng = np.random.default_rng(0); K = 16; S = 64
z = np.load(R9.ROOT / f"fabval_{me}{suf}.npz"); fu = z["Y"].reshape(len(z["Y"]), -1).mean(1) > 0.5; FX, FY = z["X"][fu], z["Y"][fu] > 0.5
trm, tem = R9.metas(); RT = np.load(R9.CACHE / "REAL_train.npy", mmap_mode="r"); Y128 = np.load(R9.CACHE / "Y128_train.npy")
idx = [i for i, m in enumerate(trm) if m["method"] == me and m["full_watermark"] and i % 5 != 0]; RX = np.stack([RT[i] for i in idx]); RY = Y128[idx] > 0.5
print(f"{me} : {len(RX)} vraies pleines, {len(FX)} pleines fabriquées ; première couche pleine résolution : {S1} ; {NST} pas", flush=True)
def centres(y):                                         # y : masque 128 (vrai = filigrané). Renvoie les centres possibles (en cases de 4 px) par type
    c = torch.from_numpy(~y).float()[None, None]; fr = F.avg_pool2d(F.pad(c, (8, 7, 8, 7)), 16, 1)[0, 0].numpy()   # part propre de la fenêtre 64 px centrée sur la case
    ok = np.zeros_like(y); ok[8:120, 8:120] = True
    edge = np.zeros_like(y); edge[:-1] |= y[:-1] ^ y[1:]; edge[:, :-1] |= y[:, :-1] ^ y[:, 1:]
    return {"bord": np.argwhere(edge & ok), "dedans": np.argwhere((fr >= 0.9) & ok), "loin": np.argwhere((fr == 0) & ok)}
def patches(X, Y, typ):
    P, G = [], []
    for g, (x, y) in enumerate(zip(X, Y)):
        c = centres(y)[typ]
        if len(c) == 0: continue
        for cy, cx in c[rng.integers(0, len(c), K)]:
            y0, x0 = int(cy) * 4 + 2 - S // 2, int(cx) * 4 + 2 - S // 2; P.append(x[y0:y0 + S, x0:x0 + S]); G.append(g)
    return np.stack(P), np.array(G)
def auc(s, t):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1); n1 = t.sum(); n0 = len(t) - n1; return (r[t == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
class Net(nn.Module):
    def __init__(s): super().__init__(); s.e = smp.encoders.get_encoder("resnet34", weights="imagenet"); s.h = nn.Linear(512, 1); s.e.conv1.stride = (1, 1) if S1 else (2, 2)   # S1 : première couche en pleine résolution, comme le segmenteur
    def forward(s, x): return s.h(s.e(x)[-1].mean((2, 3)))[:, 0]
M = torch.tensor([0.485, 0.456, 0.406], device=dev)[None, :, None, None]; D = torch.tensor([0.229, 0.224, 0.225], device=dev)[None, :, None, None]
prep = lambda p: (torch.from_numpy(p).to(dev).permute(0, 3, 1, 2).float() / 255 - M) / D
for typ in ("loin", "bord", "dedans"):
    PR, GR = patches(RX, RY, typ); PF, GF = patches(FX, FY, typ); X = np.concatenate([PR, PF]); T = np.concatenate([np.ones(len(PR)), np.zeros(len(PF))]); Gi = np.concatenate([GR, 10000 + GF])
    fold = {g: k % 5 for k, g in enumerate(rng.permutation(np.unique(GR)))}; fold.update({g: k % 5 for k, g in enumerate(rng.permutation(np.unique(10000 + GF)))}); fo = np.array([fold[g] for g in Gi]); sc = np.zeros(len(X))
    for f in range(5):
        torch.manual_seed(f); net = Net().to(dev); opt = torch.optim.AdamW(net.parameters(), 2e-4, weight_decay=1e-4); tr = np.where(fo != f)[0]; te = np.where(fo == f)[0]; net.train()
        for step in range(NST):
            b = np.concatenate([rng.choice(tr[T[tr] == 1], 32), rng.choice(tr[T[tr] == 0], 32)]); x = prep(X[b])
            if rng.random() < 0.5: x = x.flip(-1)
            if rng.random() < 0.5: x = x.flip(-2)
            loss = F.binary_cross_entropy_with_logits(net(x), torch.from_numpy(T[b]).float().to(dev)); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): sc[te] = np.concatenate([net(prep(X[te[j:j + 256]])).cpu().numpy() for j in range(0, len(te), 256)])
    gi = np.unique(Gi); si = np.array([sc[Gi == g].mean() for g in gi]); ti = (gi < 10000).astype(int)
    print(f"{typ:7s} : {len(np.unique(GR)):3d} vraies images, {len(np.unique(GF)):3d} fabriquées | AUC par vignette {auc(sc, T):.3f} | AUC par image {auc(si, ti):.3f}", flush=True)
