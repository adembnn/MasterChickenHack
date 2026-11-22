"""Critique T3 : sélecteurs indépendants des images que le gold rate.
Aucune donnée externe, aucun entraînement du modèle. Sorties : sel.csv, sel_summary.json."""
import json, numpy as np, torch, timm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from scipy.stats import spearmanr

D = "cuda"; torch.manual_seed(0); np.random.seed(0)
def build():
    return timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=100, init_values=1e-4)
sd = torch.load("data/model.pt", map_location="cpu", weights_only=True)
m = build(); m.load_state_dict(sd, strict=True); m = m.to(D).eval()
X = torch.from_numpy(np.load("data/forget_images.npz")["images"])
N = len(X)
def prep(u): return ((u.to(D).permute(0, 3, 1, 2).float() / 255.0) - 0.5) / 0.5

feats = [[] for _ in range(12)]; logits = []
hooks = [blk.register_forward_hook(lambda mod, i, o, b=b: feats[b].append(o[:, 0].float().cpu())) for b, blk in enumerate(m.blocks)]
with torch.no_grad():
    for i in range(0, N, 128):
        with torch.autocast("cuda", dtype=torch.float16):
            logits.append(m(prep(X[i:i+128])).float().cpu())
for h in hooks: h.remove()
Z = torch.cat(logits); y = Z.argmax(1).numpy()
F = [torch.cat(f) for f in feats]
top2 = Z.topk(2, 1); margin = (top2.values[:, 0] - top2.values[:, 1]).numpy(); second = top2.indices[:, 1].numpy()
print("orig acc proxy classes", len(np.unique(y)), "margin mean", margin.mean())

# 3) sensibilité au bruit dans les poids (blocs 6-11) : part des tirages où la classe change
base = {k: v.clone() for k, v in m.state_dict().items()}
flip = np.zeros(N); draws = 0; res = {}
for sig in [0.15, 0.3, 0.5]:
    fr = np.zeros(N)
    for s in range(4):
        g = torch.Generator(device=D).manual_seed(1000 * s + int(sig * 1000))
        with torch.no_grad():
            for k, p in m.named_parameters():
                if k.startswith("blocks.") and int(k.split(".")[1]) >= 6 and p.ndim == 2:
                    p.add_(torch.randn(p.shape, generator=g, device=D) * sig * base[k].std())
            out = []
            for i in range(0, N, 256):
                with torch.autocast("cuda", dtype=torch.float16):
                    out.append(m(prep(X[i:i+256])).argmax(1).cpu())
            fr += (torch.cat(out).numpy() != y)
            m.load_state_dict(base)
    res[sig] = (fr / 4).mean(); flip += fr; draws += 4
    print("sigma", sig, "flip rate", res[sig])
flip /= draws

import pandas as pd
old=pd.read_csv("sel.csv"); old["noise_flip"]=flip; old.to_csv("sel.csv",index=False,float_format="%.4f")
for c in ["margin","pred_depth","oof_b8_loss"]:
    print("spearman noise~",c, spearmanr(flip, old[c]).correlation)
top=lambda v,asc: set(np.argsort(v if asc else -v+1e-6*np.random.rand(N))[:700])
hn=top(flip,False)
for c,asc in [("margin",True),("pred_depth",False),("oof_b8_loss",False)]:
    print("overlap top20 noise &",c, len(hn & top(old[c].values,asc))/700)
print("noise_flip>0 share", (flip>0).mean(), "mean", flip.mean())
