"""Les trous de mes pleines fabriquées ont-ils la même distribution que les vrais ? Vrais = trous des pleines du filigrane (hors réserve).
Fabriqués = ce que tire l'entraînement : 70 % formes réelles des deux autres filigranes (symétrie + décalage jusqu'à 160 px, complété par du vide), 30 % taches aléatoires lisses.
Mesures par trou : part de l'image, nombre de morceaux, touche le bord, remplissage de sa boîte."""
import sys, json, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from scipy import ndimage as ndi
ROOT = Path(__file__).resolve().parent; HERE = ROOT.parent if (ROOT.parent / "kit").exists() else ROOT.parent.parent; CACHE = ROOT / "cache"
trm = json.loads((HERE / "release/train/metadata.json").read_text()); MT = np.load(CACHE / "MASK_train.npy", mmap_mode="r"); method = sys.argv[1]; rng = np.random.default_rng(0); torch.manual_seed(0)
def stats(h):
    if h.sum() == 0: return None
    lab, n = ndi.label(h); sz = ndi.sum(h, lab, np.arange(1, n + 1)); n_big = int((sz >= 0.05 * h.sum()).sum()); ys, xs = np.where(h); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sides = int(h[0].any()) + int(h[-1].any()) + int(h[:, 0].any()) + int(h[:, -1].any()); return h.mean(), n_big, sides, h.sum() / ((y1 - y0) * (x1 - x0))
def show(nm, L):
    A = np.array([s for s in L if s is not None]); print(f"  {nm:34s} n={len(A):4d} | part de l'image : médiane {np.median(A[:, 0]):.3f}, < 2 % : {np.mean(A[:, 0] < .02):.2f}, > 20 % : {np.mean(A[:, 0] > .2):.2f} | plusieurs morceaux : {np.mean(A[:, 1] >= 2):.2f} | touche 0 / 1 / 2 / 3-4 côtés : {np.mean(A[:, 2] == 0):.2f} / {np.mean(A[:, 2] == 1):.2f} / {np.mean(A[:, 2] == 2):.2f} / {np.mean(A[:, 2] >= 3):.2f} | remplissage de la boîte : médiane {np.median(A[:, 3]):.2f}")
real = [~np.array(MT[i]).astype(bool) for i, m in enumerate(trm) if m["method"] == method and m["full_watermark"] and i % 5 != 0]
bank = [~np.array(MT[i]).astype(bool) for i, m in enumerate(trm) if m["method"] != method and m["full_watermark"] and i % 5 != 0]
def dih(a, k):
    if k & 4: a = a.T
    if k & 1: a = a[:, ::-1]
    if k & 2: a = a[::-1]
    return a
def shifted(s0, maxs):
    dy, dx = int(rng.integers(-maxs, maxs + 1)), int(rng.integers(-maxs, maxs + 1)); s = np.roll(s0, (dy, dx), (0, 1)).copy()
    if dy > 0: s[:dy] = False
    elif dy < 0: s[dy:] = False
    if dx > 0: s[:, :dx] = False
    elif dx < 0: s[:, dx:] = False
    return s if s.sum() > 600 else s0
def blob(lo, hi):
    area = float(np.exp(rng.uniform(np.log(lo), np.log(hi)))); g = int(rng.integers(3, 11)); up = F.interpolate(torch.randn(1, 1, g, g), size=(512, 512), mode="bicubic", align_corners=False)[0, 0]
    return (up > torch.quantile(up.flatten(), 1 - area)).numpy()
print(f"## {method} : forme des trous des pleines"); show("VRAIS (ce filigrane)", [stats(h) for h in real]); show("vrais des deux autres filigranes", [stats(h) for h in bank])
show("fabriqué : forme réelle décalée (160)", [stats(shifted(dih(bank[int(rng.integers(len(bank)))], int(rng.integers(8))), 160)) for _ in range(600)])
show("fabriqué : forme réelle, symétrie seule", [stats(dih(bank[int(rng.integers(len(bank)))], int(rng.integers(8)))) for _ in range(600)])
show("fabriqué : tache aléatoire", [stats(blob(0.004, 0.5)) for _ in range(400)])
