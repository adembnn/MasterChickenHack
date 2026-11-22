"""Pourquoi le réseau voit-il de faux trous dans les pleines TrustMark ? Hypothèse : le filigrane y est physiquement faible (l'encodeur écrit peu à cet endroit,
ou les pixels sont saturés à 0 ou 255 et le résidu est écrêté). Mesures sur les vraies pleines hors réserve, cases réellement filigranées seulement :
 - E = force attendue du filigrane = amplitude locale du résidu que l'encodeur poserait sur cette image (toile +12 px) ;
 - S = part de pixels saturés (< 3 ou > 252) ;
 - logit du réseau (dernier point de contrôle de g_s1, entraîné sur fabriqué seul).
On compare E et S entre les cases bien vues (logit > 0) et les faux trous (logit < 0), et on donne la corrélation de rang logit / E par image.
Témoin sur fabriqué : la force attendue calculée sur l'image DÉJÀ filigranée ressemble-t-elle à la vraie force du résidu ?"""
import sys, json, argparse, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parent)); import reel5 as R
from wmloc.watermarks import load_watermark
method, tag, N = sys.argv[1], sys.argv[2], int(sys.argv[3]); dev = "cuda"; rng = np.random.default_rng(0)
trm, tem = R.metas(); RT = np.load(R.CACHE / "REAL_train.npy", mmap_mode="r"); Y = np.load(R.CACHE / "Y128_train.npy")
P = R.ROOT / "runs" / tag / method / "fold-1"; ck = torch.load(P / "ema.pt", map_location=dev) if (P / "ema.pt").exists() else None
a = argparse.Namespace(**(ck["args"] if ck else json.loads(sys.argv[4]))); a.pretrained = 0; run = R.Runner(a, dev); run.net.load_state_dict(ck["state"] if ck else torch.load(P / "ckpt.pt", map_location=dev)["ema"])
wm = load_watermark(method, models_dir=R.HERE / "kit" / "models", device=dev)
def res(b, m, p=12): c = np.pad(b, ((p, p), (p, p), (0, 0)), mode="reflect"); return wm.embed(np.ascontiguousarray(c), m)[p:p + 512, p:p + 512].astype(np.float32) - b
def energy(r): return np.abs(r).mean(2).reshape(128, 4, 128, 4).mean((1, 3))          # amplitude moyenne par case 4x4, en niveaux
def smooth(e, k=5): return F.avg_pool2d(torch.from_numpy(e)[None, None], k, 1, k // 2)[0, 0].numpy()
# témoin fabriqué
others = [i for i, m in enumerate(trm) if m["method"] != method and i % 5 != 0]; cc = []; cv = []
for _ in range(8):
    b = np.ascontiguousarray(RT[int(rng.choice(others))]); m = rng.integers(0, 2, wm.nbits); r = res(b, m); w = np.clip(b + r, 0, 255).astype(np.uint8); e1 = smooth(energy(r)); e2 = smooth(energy(res(w, rng.integers(0, 2, wm.nbits))))
    cc.append(spearmanr(e1.ravel(), e2.ravel())[0]); cv.append(e1.std() / e1.mean())
print(f"## {method} témoin fabriqué : la force du vrai résidu varie dans l'image (écart-type / moyenne = {np.mean(cv):.2f}) ; corrélation de rang avec la force recalculée sur l'image filigranée (autre message) : {np.mean(cc):.2f}")
ids = [i for i, m in enumerate(trm) if m["method"] == method and i % 5 != 0 and m["full_watermark"]][:N]; X = np.stack([np.ascontiguousarray(RT[i]) for i in ids]); lg = run.predict(run.net, X, np.ones(len(ids)), tta=1, bs=8)
rows = []
for x, l, i in zip(X, lg, ids):
    wmk = Y[i] > 0; E = smooth(energy(res(x, np.array([int(c) for c in trm[i]["message"]])))); S = ((x < 3) | (x > 252)).any(2).reshape(128, 4, 128, 4).mean((1, 3)); ok = wmk & (l > 0); fh = wmk & (l < 0)
    if fh.sum() < 20 or ok.sum() < 20: continue
    rows.append((E[ok].mean(), E[fh].mean(), S[ok].mean(), S[fh].mean(), spearmanr(l[wmk], E[wmk])[0], fh.sum() / wmk.sum()))
r = np.array(rows); print(f"## {method} : {len(r)} vraies pleines avec de faux trous (part moyenne des cases filigranées vues comme trou : {r[:, 5].mean():.2f})")
print(f"  force attendue E (niveaux) : cases bien vues {r[:, 0].mean():.2f} | faux trous {r[:, 1].mean():.2f} | faux trous plus faibles dans {np.mean(r[:, 1] < r[:, 0]):.2f} des images")
print(f"  pixels saturés             : cases bien vues {r[:, 2].mean():.3f} | faux trous {r[:, 3].mean():.3f}")
print(f"  corrélation de rang logit / E dans les zones filigranées : moyenne {r[:, 4].mean():.2f}, quartiles {np.percentile(r[:, 4], 25):.2f} à {np.percentile(r[:, 4], 75):.2f}")
