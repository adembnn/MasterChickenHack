"""Test court : peut-on estimer, image par image, le cadre sur lequel le filigrane a été posé (toile plus grande que l'image, puis rognée) ?
On re-borde la vraie image de (haut, gauche, bas, droite) pixels en miroir et on mesure la marge du décodeur sur le vrai message.
Questions : (1) la marge remonte-t-elle ? (2) le meilleur bord est-il le même des quatre côtés (rognage centré) ou non ? (3) varie-t-il d'une image à l'autre ?
Second volet : dans les masques des images pleines, la zone propre est-elle une bande de bord (rectangle filigrané plus petit que l'image) ou un trou intérieur ?"""
import sys, json, itertools
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
HERE = Path(__file__).resolve().parent.parent; REL = HERE / "release"; sys.path.insert(0, str(HERE / "kit"))
from wmloc.watermarks import load_watermark
meta = json.load(open(REL / "train" / "metadata.json")); me = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 40
z = np.load(REL / "train" / "masks_128.npz"); g = {str(n): m.astype(bool) for n, m in zip(z["names"], z["masks"])}
# volet 2 : géométrie des masques pleins
band = []; inner = []
for m in meta:
    if not m["full_watermark"]: continue
    G = g[m["image_name"]]; ys, xs = np.where(G); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    out = (~G).sum() - (~G[y0:y1, x0:x1]).sum(); band.append(out / max((~G).sum(), 1)); inner.append((y0, 128 - y1, x0, 128 - x1))
band = np.array(band); inner = np.array(inner)
print(f"[masques pleins, 3 filigranes, n={len(band)}] part de la zone propre située HORS de la boîte englobante du filigrane : moyenne {band.mean():.3f} ; images où la boîte du filigrane ne touche pas les 4 bords : {(inner.max(1) > 0).mean():.2f} ; marge médiane quand elle existe (cases de 4 px) : {np.median(inner[inner > 0]) if (inner > 0).any() else 0}")
# volet 1
wm = load_watermark(me, models_dir=HERE / "kit" / "models", device="cuda")
ids = [i for i, m in enumerate(meta) if m["method"] == me and m["full_watermark"] and i % 5 != 0][:N]
X = torch.from_numpy(np.stack([np.asarray(Image.open(REL / "train" / "images" / f"{meta[i]['image_name']}.png").convert("RGB")) for i in ids])).permute(0, 3, 1, 2).float().div(255).cuda()
S = torch.tensor([[2.0 * int(c) - 1 for c in meta[i]["message"]] for i in ids])
vals = (0, 4, 8, 12, 16, 20, 24, 32); res = {}
for t, l, b, r in itertools.product(vals, repeat=4):
    if me == "stegastamp" and (t % 8 or l % 8 or b % 8 or r % 8): continue
    xp = F.pad(X, (l, r, t, b), mode="reflect") if max(t, l, b, r) else X
    lg = wm.logits(xp).cpu(); res[(t, l, b, r)] = ((lg * S).mean(1).numpy(), ((lg > 0) == (S > 0)).float().mean(1).numpy())
K = list(res); MG = np.stack([res[k][0] for k in K], 1); BA = np.stack([res[k][1] for k in K], 1)        # [image, combinaison]
i0 = K.index((0, 0, 0, 0)); sym = [j for j, k in enumerate(K) if len(set(k)) == 1]
bs = MG[:, sym].argmax(1); b4 = MG.argmax(1)
print(f"[{me}, {len(ids)} pleines hors réserve] marge du décodeur : sans bord {MG[:, i0].mean():.2f} (bits {BA[:, i0].mean():.3f}) ; meilleur bord égal des 4 côtés, choisi par image {MG[:, sym].max(1).mean():.2f} (bits {BA[np.arange(len(ids)), np.array(sym)[bs]].mean():.3f}) ; meilleur bord libre par côté {MG.max(1).mean():.2f} (bits {BA[np.arange(len(ids)), b4].mean():.3f})")
gm = MG.mean(0); print("   meilleure combinaison commune à toutes les images :", K[int(gm.argmax())], f"marge {gm.max():.2f}")
B4 = np.array([K[j] for j in b4]); BS = np.array([K[sym[j]][0] for j in bs])
print("   bord égal choisi par image, répartition :", dict(zip(*np.unique(BS, return_counts=True))))
print("   bord libre : moyenne par côté (haut, gauche, bas, droite)", B4.mean(0).round(1), "| écart-type", B4.std(0).round(1), "| images dont les 4 côtés diffèrent de plus de 8 px :", float(((B4.max(1) - B4.min(1)) > 8).mean()))
print("   gain de marge libre contre égal, par image : médiane", round(float(np.median(MG.max(1) - MG[:, sym].max(1))), 3), "| contre sans bord :", round(float(np.median(MG.max(1) - MG[:, i0])), 3))
print("   exemples (haut, gauche, bas, droite) :", [tuple(int(v) for v in x) for x in B4[:12]])
