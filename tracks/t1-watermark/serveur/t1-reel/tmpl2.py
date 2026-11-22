"""Le motif recalculé à partir du message connu devient-il une carte utile une fois posé dans le BON cadre (toile agrandie de p px) ?
T_p = embed(toile(x, p), msg) rogné - x ; carte = corrélation locale normalisée entre le détail de l'image et T_p ; AUC par image (pixel filigrané > propre)."""
import sys, json, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
ROOT = Path(__file__).resolve().parent; HERE = ROOT.parent if (ROOT.parent / "kit").exists() else ROOT.parent.parent; CACHE = ROOT / "cache"; sys.path.insert(0, str(HERE / "kit"))
from wmloc.watermarks import load_watermark
dev = "cuda"; trm = json.loads((HERE / "release/train/metadata.json").read_text()); RT = np.load(CACHE / "REAL_train.npy", mmap_mode="r"); Y = np.load(CACHE / "Y128_train.npy"); N = int(sys.argv[1])
def gk(sig):
    r = int(3 * sig + 1); g = torch.arange(-r, r + 1, device=dev).float(); g = torch.exp(-g ** 2 / (2 * sig ** 2)); return g / g.sum(), r
def blur(x, sig):
    g, r = gk(sig); c = x.shape[1]; x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), g.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), g.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)
def auc(s, l):
    s = s.ravel(); l = l.ravel() > 0; o = np.argsort(s, kind="stable"); r = np.empty(len(o)); r[o] = np.arange(1, len(o) + 1); n1 = l.sum(); n0 = len(l) - n1; return (r[l].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def cmap(x, T, sig, k):
    xh = x - blur(x, sig); th = T - blur(T, sig); box = lambda a: F.avg_pool2d(a.sum(1, keepdim=True), k, 1, k // 2)
    return F.avg_pool2d(box(xh * th) / (box(xh * xh) * box(th * th)).sqrt().clamp_min(1e-9), 4)[0, 0].cpu().numpy()
for method in sys.argv[2:]:
    wm = load_watermark(method, models_dir=HERE / "kit" / "models", device=dev); st = {r["i"]: r for r in json.load(open(ROOT / f"stats_{method}.json"))}
    ids = [i for i, m in enumerate(trm) if m["method"] == method and i % 5 != 0]
    G = {"pleines": [i for i in ids if trm[i]["full_watermark"]][:N], "objets": [i for i in ids if not trm[i]["full_watermark"] and st[i]["rect"] <= 0.98][:N]}
    print(f"\n## {method} : AUC moyenne par image de la carte de corrélation, selon la marge p de la toile")
    for g, L in G.items():
        res = {}
        for i in L:
            if Y[i].min() == Y[i].max(): continue
            xu = np.ascontiguousarray(RT[i]); bits = np.array([int(c) for c in trm[i]["message"]]); x = torch.from_numpy(xu).permute(2, 0, 1)[None].float().to(dev)
            for p in (0, 6, 10, 12, 14, 18):
                c = np.pad(xu, ((p, p), (p, p), (0, 0)), mode="reflect") if p else xu; T = torch.from_numpy(wm.embed(np.ascontiguousarray(c), bits)[p:p + 512, p:p + 512].astype(np.float32)).permute(2, 0, 1)[None].to(dev) - x
                for sig, k in ((2.0, 17), (2.0, 33), (4.0, 33), (1.0, 33)): res.setdefault((p, sig, k), []).append(auc(cmap(x, T, sig, k), Y[i]))
        for p in (0, 6, 10, 12, 14, 18): print(f"  {g:8s} p={p:2d} : " + " ; ".join(f"sig={s:.0f} k={k} {np.mean(v):.3f}" for (pp, s, k), v in res.items() if pp == p))
    del wm; torch.cuda.empty_cache()
