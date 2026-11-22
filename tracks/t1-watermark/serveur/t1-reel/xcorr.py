"""TrustMark : retrouver le cadre exact de chaque vraie image, puis localiser. La moitié du motif TrustMark ne dépend que du message (corrélation 0,53 entre le
résidu d'une image et le résidu moyen de 8 autres images pour le même message). On calcule donc ce motif moyen sur une toile de côté 512 + 2p, et on cherche,
pour chaque p et chaque position de rognage, où le détail de la vraie image lui ressemble le plus (corrélation croisée normalisée). Un pic net = cadre trouvé.
Puis carte locale de corrélation avec le motif aligné : AUC par image contre le vrai masque."""
import sys, json, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
ROOT = Path(__file__).resolve().parent; HERE = ROOT.parent if (ROOT.parent / "kit").exists() else ROOT.parent.parent; CACHE = ROOT / "cache"; sys.path.insert(0, str(HERE / "kit"))
from wmloc.watermarks import load_watermark
dev = "cuda"; method = sys.argv[1]; N = int(sys.argv[2]); K = int(sys.argv[3]); wm = load_watermark(method, models_dir=HERE / "kit" / "models", device=dev); rng = np.random.default_rng(0)
trm = json.loads((HERE / "release/train/metadata.json").read_text()); RT = np.load(CACHE / "REAL_train.npy", mmap_mode="r"); Y = np.load(CACHE / "Y128_train.npy"); MT = np.load(CACHE / "MASK_train.npy", mmap_mode="r")
others = [i for i, m in enumerate(trm) if m["method"] != method and i % 5 != 0]; clean = [np.ascontiguousarray(RT[int(rng.choice(others))]) for _ in range(K)]
def gk(sig):
    r = int(3 * sig + 1); g = torch.arange(-r, r + 1, device=dev).float(); g = torch.exp(-g ** 2 / (2 * sig ** 2)); return g / g.sum(), r
def hp(x, sig=2.0):                                      # x : 1 x 3 x H x W
    g, r = gk(sig); b = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), g.view(1, 1, 1, -1).repeat(3, 1, 1, 1), groups=3); b = F.conv2d(F.pad(b, (0, 0, r, r), mode="reflect"), g.view(1, 1, -1, 1).repeat(3, 1, 1, 1), groups=3); return x - b
T_ = lambda a: torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)[None].float().to(dev)
def template(bits, p):                                   # motif moyen du message sur la toile de côté 512 + 2p : 1 x 3 x S x S
    acc = 0
    for b in clean:
        c = np.pad(b, ((p, p), (p, p), (0, 0)), mode="reflect"); acc = acc + (wm.embed(np.ascontiguousarray(c), bits).astype(np.float32) - c)
    return hp(T_(acc / len(clean)))
def search(x, bits, P=(0, 4, 6, 8, 10, 12, 14, 16, 20)):
    xh = hp(T_(x)); nx = xh.pow(2).sum().sqrt(); out = []
    for p in P:
        t = template(bits, p); num = F.conv2d(t, xh); den = F.conv2d(t.pow(2).sum(1, keepdim=True), torch.ones(1, 1, 512, 512, device=dev)).clamp_min(1e-6).sqrt() * nx; ncc = (num / den)[0, 0]
        k = int(ncc.argmax()); oy, ox = divmod(k, ncc.shape[1]); out.append((p, float(ncc.max()), oy, ox, float(ncc[p, p]), float(ncc.median()), float(ncc.std()) if ncc.numel() > 1 else 0.0, t))
    return xh, out
def auc(s, l):
    s = s.ravel(); l = l.ravel() > 0; o = np.argsort(s, kind="stable"); r_ = np.empty(len(o)); r_[o] = np.arange(1, len(o) + 1); n1 = l.sum(); n0 = len(l) - n1; return (r_[l].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def local(xh, t, k):
    box = lambda a: F.avg_pool2d(a.sum(1, keepdim=True), k, 1, k // 2); return F.avg_pool2d(box(xh * t) / (box(xh * xh) * box(t * t)).sqrt().clamp_min(1e-9), 4)[0, 0].cpu().numpy()
# témoin : image fabriquée à marge connue (12 px, rognage décentré de (+3, -2))
b = np.ascontiguousarray(RT[int(rng.choice(others))]); m = rng.integers(0, 2, wm.nbits); c = np.pad(b, ((15, 9), (10, 14), (0, 0)), mode="reflect"); w = wm.embed(np.ascontiguousarray(c), m)[15:527, 10:522]
xh, o = search(w, m); print(f"## {method} témoin fabriqué (vraie marge 12, rognage en (15, 10)) :", " | ".join(f"p={p}: pic {v:.3f} en ({oy},{ox})" for p, v, oy, ox, *_ in o))
ids = [i for i, mm in enumerate(trm) if mm["method"] == method and i % 5 != 0 and mm["full_watermark"]][:N]; A = {}
for i in ids:
    x = np.ascontiguousarray(RT[i]); bits = np.array([int(ch) for ch in trm[i]["message"]]); xh, o = search(x, bits); best = max(o, key=lambda r: r[1]); p, v, oy, ox, vc, med, sd, t = best
    print(f"image {i} (filigranée à {MT[i].mean():.2f}) : " + " ".join(f"p{q}:{vv:.3f}" for q, vv, *_ in o) + f" -> meilleur p={p}, rognage ({oy},{ox}), pic {v:.3f}, bruit {sd:.3f}", flush=True)
    if Y[i].min() != Y[i].max():
        ta = t[..., oy:oy + 512, ox:ox + 512]; t0 = o[0][7]
        for k in (17, 33, 65): A.setdefault(("aligné", k), []).append(auc(local(xh, ta, k), Y[i])); A.setdefault(("cadre 512", k), []).append(auc(local(xh, t0, k), Y[i]))
print("AUC moyenne par image de la carte locale :", " ; ".join(f"{a} k={k} {np.mean(v):.3f} (part > 0,9 : {np.mean(np.array(v) > .9):.2f})" for (a, k), v in A.items()))
