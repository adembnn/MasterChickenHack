"""Étalonnage de l'estimateur de cadre : sur des images FABRIQUÉES dont on connaît le bord (haut, gauche, bas, droite) de la toile,
l'argmax de la marge du décodeur retrouve-t-il ce bord ? Trois lots : bord égal des 4 côtés (rognage centré), bord tiré indépendamment par côté
(rognage non centré), et les vraies images pleines. Si les vraies ressemblent au lot « non centré », la recette doit tirer un bord par côté."""
import sys, json, itertools
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
HERE = Path(__file__).resolve().parent.parent; REL = HERE / "release"; sys.path.insert(0, str(HERE / "kit"))
from wmloc.watermarks import load_watermark
meta = json.load(open(REL / "train" / "metadata.json")); me = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 24
rng = np.random.default_rng(1); wm = load_watermark(me, models_dir=HERE / "kit" / "models", device="cuda"); vals = (0, 8, 16, 24)
rd = lambda i: np.asarray(Image.open(REL / "train" / "images" / f"{meta[i]['image_name']}.png").convert("RGB"))
bg = [i for i, m in enumerate(meta) if m["method"] != me and not m["full_watermark"] and i % 5 != 0]; bg = list(rng.choice(bg, N, replace=False))
real = [i for i, m in enumerate(meta) if m["method"] == me and m["full_watermark"] and i % 5 != 0][:N]
def fab(i, pads):
    t, l, b, r = pads; img = rd(i); bits = rng.integers(0, 2, wm.nbits)
    c = np.pad(img, ((t, b), (l, r), (0, 0)), mode="reflect"); w = wm.embed(c, bits)[t:t + 512, l:l + 512]
    return w, bits
def estimate(X, S):
    X = torch.from_numpy(np.stack(X)).permute(0, 3, 1, 2).float().div(255).cuda(); S = torch.tensor(np.stack(S)).float() * 2 - 1; K = []; M = []
    for t, l, b, r in itertools.product(vals, repeat=4):
        xp = F.pad(X, (l, r, t, b), mode="reflect") if max(t, l, b, r) else X
        M.append((wm.logits(xp).cpu() * S).mean(1).numpy()); K.append((t, l, b, r))
    M = np.stack(M, 1); return np.array([K[j] for j in M.argmax(1)]), M.max(1), M[:, 0]
def stat(tag, est, truth=None, mx=None, m0=None):
    asym = float(((est.max(1) - est.min(1)) > 8).mean()); s = f"{tag:38s} n={len(est)} | bord estimé moyen par côté {est.mean(0).round(1)} | images aux côtés inégaux (> 8 px) : {asym:.2f} | marge sans bord {m0.mean():.2f} -> au meilleur bord {mx.mean():.2f}"
    if truth is not None: s += f" | bord retrouvé exactement, par côté : {(est == truth).mean():.2f} ; erreur absolue moyenne {np.abs(est - truth).mean():.1f} px"
    print(s, flush=True)
T1 = [(p, p, p, p) for p in rng.choice([8, 16], N)]; X, S = zip(*[fab(i, p) for i, p in zip(bg, T1)]); e, mx, m0 = estimate(X, S); stat(f"[{me}] fabriqué, rognage centré", e, np.array(T1), mx, m0)
T2 = [tuple(rng.choice(vals, 4)) for _ in range(N)]; X, S = zip(*[fab(i, p) for i, p in zip(bg, T2)]); e, mx, m0 = estimate(X, S); stat(f"[{me}] fabriqué, bord libre par côté", e, np.array(T2), mx, m0)
X = [rd(i) for i in real]; S = [np.array([int(c) for c in meta[i]["message"]]) for i in real]; e, mx, m0 = estimate(X, S); stat(f"[{me}] VRAIES pleines", e, None, mx, m0)
print("   vraies, bords estimés :", [tuple(int(v) for v in x) for x in e[:16]])
