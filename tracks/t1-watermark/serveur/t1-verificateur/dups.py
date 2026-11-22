"""Y a-t-il des images sources partagées (quasi-doublons) entre réserve et entraînement, entre train et test, ou entre filigranes ?
Vignette 24x24 en gris, centrée-réduite ; corrélation maximale de chaque image avec toutes les autres."""
import json
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent.parent; REL = HERE / "release"
trm = json.load(open(REL / "train" / "metadata.json")); tem = json.load(open(REL / "test" / "metadata.json"))
def thumb(p):
    a = np.asarray(Image.open(p).convert("L").resize((24, 24), Image.Resampling.BOX), np.float32).ravel()
    a -= a.mean(); return a / (np.linalg.norm(a) + 1e-6)
T = np.stack([thumb(REL / "train" / "images" / f"{m['image_name']}.png") for m in trm] + [thumb(REL / "test" / "images" / f"{m['image_name']}.png") for m in tem])
C = T @ T.T; np.fill_diagonal(C, -1)
ntr = len(trm); lab = [("train", i, m) for i, m in enumerate(trm)] + [("test", i, m) for i, m in enumerate(tem)]
best = C.max(1); arg = C.argmax(1)
print("corrélation maximale avec une autre image, quantiles 50/90/99/100 :", np.round(np.percentile(best, [50, 90, 99, 100]), 3))
for th in (0.98, 0.95, 0.9, 0.8):
    pairs = np.argwhere(np.triu(C > th, 1)); print(f"paires au-dessus de {th} : {len(pairs)}")
pairs = np.argwhere(np.triu(C > 0.9, 1))
kinds = {}
for a, b in pairs:
    A, B = lab[a], lab[b]
    key = (A[0], B[0], "même filigrane" if A[2]["method"] == B[2]["method"] else "filigranes différents")
    kinds[key] = kinds.get(key, 0) + 1
print("répartition des paires > 0,9 :", kinds)
res_leak = sum(1 for a, b in pairs if lab[a][0] == lab[b][0] == "train" and ((lab[a][1] % 5 == 0) != (lab[b][1] % 5 == 0)))
print("paires > 0,9 entre une image de réserve et une image hors réserve du train :", res_leak)
for a, b in pairs[:25]:
    A, B = lab[a], lab[b]; print(f"  {C[a, b]:.3f}  {A[0]}:{A[2]['image_name']} ({A[2]['method']}, {A[2].get('full_watermark')})  ~  {B[0]}:{B[2]['image_name']} ({B[2]['method']}, {B[2].get('full_watermark')})")
print("clés des métadonnées test :", sorted(tem[0].keys()), "| effectifs test :", {k: sum(m['method'] == k for m in tem) for k in ('pixelseal', 'trustmark', 'stegastamp')})
# messages : un même message revient-il entre images (même source / même clé) ?
from collections import Counter
cm = Counter(m["message"] for m in trm + tem); rep = [c for c in cm.values() if c > 1]
print("messages distincts :", len(cm), "sur", len(trm) + len(tem), "| messages répétés :", len(rep), "max", max(cm.values()))
