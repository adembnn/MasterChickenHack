"""Dans quel cadre le vrai filigrane des pleines a-t-il été posé ? Hypothèse : l'image d'origine était plus grande (S x S) puis rognée en 512.
Pour chaque hypothèse (taille S, position du rognage), on re-filigrane une toile S x S contenant l'image et on mesure la corrélation entre le détail de la
vraie image et le motif obtenu, au centre et aux bords. Le bon cadre donne une corrélation aussi forte aux bords qu'au centre."""
import sys, json, numpy as np, torch
from pathlib import Path
from scipy import ndimage as ndi
ROOT = Path(__file__).resolve().parent; HERE = ROOT.parent if (ROOT.parent / "kit").exists() else ROOT.parent.parent; CACHE = ROOT / "cache"; sys.path.insert(0, str(HERE / "kit"))
from wmloc.watermarks import load_watermark
method, N = sys.argv[1], int(sys.argv[2]); wm = load_watermark(method, models_dir=HERE / "kit" / "models", device="cuda")
trm = json.loads((HERE / "release/train/metadata.json").read_text()); RT = np.load(CACHE / "REAL_train.npy", mmap_mode="r"); MT = np.load(CACHE / "MASK_train.npy", mmap_mode="r")
fulls = [i for i, m in enumerate(trm) if m["method"] == method and i % 5 != 0 and m["full_watermark"] and MT[i].mean() > 0.9][:N]
hp = lambda a: a - ndi.gaussian_filter(a, (2, 2, 0))
edge = np.zeros((512, 512), bool); edge[:64] = edge[-64:] = True; edge[:, :64] = edge[:, -64:] = True
def corr(a, b, m): a, b = a[m], b[m]; return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-9))
def motif(x, bits, S, oy, ox):                           # toile S x S (bords en miroir), image posée en (oy, ox), filigrane, rognage
    c = np.pad(x, ((oy, S - 512 - oy), (ox, S - 512 - ox), (0, 0)), mode="reflect") if S > 512 else x
    return wm.embed(np.ascontiguousarray(c), bits)[oy:oy + 512, ox:ox + 512].astype(np.float32) - x
print(f"## {method} : {len(fulls)} vraies pleines (filigranées à plus de 90 %)")
tot = {}
for i in fulls:
    x = np.ascontiguousarray(RT[i]); bits = np.array([int(c) for c in trm[i]["message"]]); M = np.array(MT[i]).astype(bool); hx = hp(x.astype(np.float32))
    for S in (512, 516, 520, 524, 528, 536, 544, 576, 640, 768, 1024):
        o = (S - 512) // 2; t = hp(motif(x, bits, S, o, o)); tot.setdefault(S, []).append((corr(hx, t, M & ~edge), corr(hx, t, M & edge)))
print("taille S du cadre (rognage centré) -> corrélation au centre / aux bords")
for S, v in tot.items(): v = np.array(v); print(f"  S={S:4d} : {v[:, 0].mean():.3f} / {v[:, 1].mean():.3f}")
# témoin : la même mesure sur des pleines fabriquées par le kit en cadre 512 (fonds = images d'un autre filigrane)
rng = np.random.default_rng(0); others = [i for i, m in enumerate(trm) if m["method"] != method and i % 5 != 0]; t0 = []
for _ in range(len(fulls)):
    b = np.ascontiguousarray(RT[int(rng.choice(others))]); bits = rng.integers(0, 2, wm.nbits); w = wm.embed(b, bits); t = hp(wm.embed(w, bits).astype(np.float32) - w); hw = hp(w.astype(np.float32)); a = np.ones((512, 512), bool)
    t0.append((corr(hw, t, a & ~edge), corr(hw, t, a & edge)))
t0 = np.array(t0); print(f"témoin fabriqué (cadre 512, re-filigrané en 512) : {t0[:, 0].mean():.3f} / {t0[:, 1].mean():.3f}")
