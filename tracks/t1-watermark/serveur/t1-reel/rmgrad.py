"""Idée pour les pleines TrustMark : un test « contrefactuel » propre. L'effaceur d'Adobe donne d = effaceur(x) - x (ce qu'il retirerait). Le décodeur donne
g = gradient de son accord au message connu. Le produit -g.d, pixel par pixel, estime de combien l'accord BAISSERAIT si on effaçait le filigrane à cet endroit :
positif là où il y a du filigrane lisible, nul dans un trou propre (l'effaceur n'y retire rien d'utile au message). Vérification exacte par fenêtres :
on efface pour de vrai dans une fenêtre et on mesure la baisse d'accord. AUC par image contre le vrai masque, vraies images hors réserve."""
import sys, json, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from PIL import Image
ROOT = Path(__file__).resolve().parent; HERE = ROOT.parent if (ROOT.parent / "kit").exists() else ROOT.parent.parent; CACHE = ROOT / "cache"; sys.path.insert(0, str(HERE / "kit"))
from wmloc.watermarks import load_watermark
dev = "cuda"; N = int(sys.argv[1]); wm = load_watermark("trustmark", models_dir=HERE / "kit" / "models", device=dev); tm = wm.tm
trm = json.loads((HERE / "release/train/metadata.json").read_text()); RT = np.load(CACHE / "REAL_train.npy", mmap_mode="r"); Y = np.load(CACHE / "Y128_train.npy"); st = {r["i"]: r for r in json.load(open(ROOT / "stats_trustmark.json"))}
def auc(s, l):
    s = s.ravel(); l = l.ravel() > 0; o = np.argsort(s, kind="stable"); r = np.empty(len(o)); r[o] = np.arange(1, len(o) + 1); n1 = l.sum(); n0 = len(l) - n1; return (r[l].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
def margin(x01, sgn): return (wm.net(wm._prep(x01)).float() * sgn[None]).mean(1)          # accord moyen des logits avec le message, par image
@torch.no_grad()
def removed(x01):                                        # image après passage de l'effaceur (calcul en 256 puis résidu agrandi, comme le paquet trustmark)
    x256 = F.interpolate(x01, size=(256, 256), mode="bilinear", align_corners=False, antialias=True) * 2 - 1
    res = tm.removal(x256).clamp(-1, 1) - x256; return (x01 + F.interpolate(res, size=x01.shape[-2:], mode="bilinear", align_corners=False) / 2).clamp(0, 1)
ids = [i for i, m in enumerate(trm) if m["method"] == "trustmark" and i % 5 != 0]
G = {"pleines": [i for i in ids if trm[i]["full_watermark"]][:N], "objets": [i for i in ids if not trm[i]["full_watermark"] and st[i]["rect"] <= 0.98][:N], "rectangles": [i for i in ids if not trm[i]["full_watermark"] and st[i]["rect"] > 0.98][:N]}
for g, L in G.items():
    res = {}
    for i in L:
        if Y[i].min() == Y[i].max(): continue
        x = torch.from_numpy(np.ascontiguousarray(RT[i])).permute(2, 0, 1)[None].float().div(255).to(dev); sgn = torch.tensor([1.0 if c == "1" else -1.0 for c in trm[i]["message"]], device=dev)
        xr = removed(x); d = xr - x; xg = x.clone().requires_grad_(True)
        with torch.enable_grad(): margin(xg, sgn).sum().backward()
        c = -(xg.grad * d).sum(1, keepdim=True)          # baisse d'accord attendue si on efface ici (1er ordre)
        for k in (17, 33, 65): res.setdefault(f"produit gradient x effaceur k={k}", []).append(auc(F.avg_pool2d(F.avg_pool2d(c, k, 1, k // 2), 4)[0, 0].cpu().numpy(), Y[i]))
        with torch.no_grad():                            # version exacte : effacer vraiment dans des fenêtres W x W (pas W/2) et mesurer la baisse d'accord
            m0 = float(margin(x, sgn))
            for W in (64, 96):
                acc = torch.zeros(1, 1, 512, 512, device=dev); cnt = torch.zeros_like(acc); pos = [(y, xx) for y in range(0, 512 - W + 1, W // 2) for xx in range(0, 512 - W + 1, W // 2)]
                for j in range(0, len(pos), 64):
                    xb = x.repeat(len(pos[j:j + 64]), 1, 1, 1)
                    for b, (y, xx) in enumerate(pos[j:j + 64]): xb[b, :, y:y + W, xx:xx + W] = xr[0, :, y:y + W, xx:xx + W]
                    dm = m0 - margin(xb, sgn)
                    for b, (y, xx) in enumerate(pos[j:j + 64]): acc[..., y:y + W, xx:xx + W] += dm[b]; cnt[..., y:y + W, xx:xx + W] += 1
                res.setdefault(f"effacement réel par fenêtres de {W} px", []).append(auc(F.avg_pool2d(acc / cnt.clamp_min(1), 4)[0, 0].cpu().numpy(), Y[i]))
    print(f"## TrustMark {g} ({len(L)}) : AUC moyenne par image (0,5 = hasard)")
    for k, v in res.items(): print(f"  {k:42s} {np.mean(v):.3f}  médiane {np.median(v):.3f}  part > 0,9 : {np.mean(np.array(v) > .9):.2f}  part < 0,6 : {np.mean(np.array(v) < .6):.2f}")
