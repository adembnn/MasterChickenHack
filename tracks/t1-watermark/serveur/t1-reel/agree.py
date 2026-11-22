"""TrustMark et PixelSeal : carte d'accord du décodeur. Avant sa moyenne finale, le décodeur donne des logits par position (grille 8x8) : là où ils
s'accordent avec le message connu, le filigrane est lisible. On affine la grille en décalant l'image (miroir) et en moyennant. AUC par image contre le vrai masque."""
import sys, json, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
ROOT = Path(__file__).resolve().parent; HERE = ROOT.parent if (ROOT.parent / "kit").exists() else ROOT.parent.parent; CACHE = ROOT / "cache"; sys.path.insert(0, str(HERE / "kit"))
from wmloc.watermarks import load_watermark
dev = "cuda"; trm = json.loads((HERE / "release/train/metadata.json").read_text()); RT = np.load(CACHE / "REAL_train.npy", mmap_mode="r"); Y = np.load(CACHE / "Y128_train.npy"); N = int(sys.argv[1])
def auc(s, l):
    s = s.ravel(); l = l.ravel() > 0; o = np.argsort(s, kind="stable"); r = np.empty(len(o)); r[o] = np.arange(1, len(o) + 1); n1 = l.sum(); n0 = len(l) - n1; return (r[l].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
for method in sys.argv[2:]:
    wm = load_watermark(method, models_dir=HERE / "kit" / "models", device=dev); st = {r["i"]: r for r in json.load(open(ROOT / f"stats_{method}.json"))}
    ids = [i for i, m in enumerate(trm) if m["method"] == method and i % 5 != 0]
    G = {"pleines": [i for i in ids if trm[i]["full_watermark"]][:N], "objets": [i for i in ids if not trm[i]["full_watermark"] and st[i]["rect"] <= 0.98][:N]}
    @torch.no_grad()
    def amap(x, sgn, shifts, pad):                       # x : 1x3x512x512 [0,1] ; carte 512x512 moyenne sur les décalages ; pad = marge de toile ajoutée en miroir
        acc = torch.zeros(1, 1, 512, 512, device=dev); cnt = torch.zeros_like(acc)
        for dy in shifts:
            for dx in shifts:
                xp = F.pad(x, (pad + dx, pad + 64 - dx, pad + dy, pad + 64 - dy), mode="reflect") if (pad or len(shifts) > 1) else x
                pp = wm.features(xp)["per_position_logits"].float(); a = torch.tanh(pp * sgn[None, :, None, None] / pp.abs().median()).mean(1, keepdim=True)
                a = F.interpolate(a, size=xp.shape[-2:], mode="bilinear", align_corners=False); y0, x0 = (pad + dy, pad + dx) if (pad or len(shifts) > 1) else (0, 0)
                acc += a[..., y0:y0 + 512, x0:x0 + 512]; cnt += 1
        return F.avg_pool2d(acc / cnt, 4)[0, 0].cpu().numpy()
    print(f"\n## {method} : AUC moyenne par image de la carte d'accord du décodeur (0,5 = hasard)")
    for g, L in G.items():
        res = {}
        for i in L:
            if Y[i].min() == Y[i].max(): continue
            x = torch.from_numpy(np.ascontiguousarray(RT[i])).permute(2, 0, 1)[None].float().div(255).to(dev); sgn = torch.tensor([1.0 if c == "1" else -1.0 for c in trm[i]["message"]], device=dev)
            for nm, sh, pad in (("grille 8x8 brute", [0], 0), ("4 décalages", [0, 32], 0), ("16 décalages", [0, 16, 32, 48], 0), ("16 décalages + toile 12 px", [0, 16, 32, 48], 12)):
                res.setdefault(nm, []).append(auc(amap(x, sgn, sh, pad), Y[i]))
        print(f"  {g:8s} : " + " ; ".join(f"{k} {np.mean(v):.3f} (part > 0,9 : {np.mean(np.array(v) > .9):.2f})" for k, v in res.items()))
    del wm; torch.cuda.empty_cache()
