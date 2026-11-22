"""T1 PixelSeal · prédiction avec le décodeur local dense (session t1-pixelseal).
Usage : python predict.py --runs run_F1 run_F2 run_F3 run_F4 --split test --out pred_test_pixelseal.npz
Moyenne les logits des modèles donnés. Sortie : names, logits float16 N x 128 x 128 (> 0 = filigrané).
Pour --split train, la prédiction est hors-pli : chaque image n'est prédite que par les modèles qui ne l'ont pas vue
(réserve index % 5 == 0 : tous les modèles ; pli f : le modèle run_F<f>)."""
import os, sys, json, argparse, copy
os.environ.setdefault("OMP_NUM_THREADS", "8")
from pathlib import Path
import numpy as np, torch
from PIL import Image
HERE = Path(os.environ.get("WMLOC_ROOT", "/p/scratch/training2666/destors1/MasterChickenHack/Hackathon_Watermark_Localisation_Task"))
sys.path.insert(0, str(HERE / "kit")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from wmloc.watermarks import load_watermark
from wmloc.utils import miou_image
from locnet import Loc
ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--folds", nargs="*", type=int, default=None, help="pli tenu à l'écart de chaque run (pour --split train)")
ap.add_argument("--split", default="test"); ap.add_argument("--out", required=True); ap.add_argument("--res", type=int, default=256); ap.add_argument("--zoom", type=int, default=0); ap.add_argument("--ppinit", type=int, default=1)
ap.add_argument("--dir", default=str(HERE / "adem" / "t1-pixelseal")); ap.add_argument("--dev", default="cuda"); ap.add_argument("--th", type=float, default=0.0)
a = ap.parse_args(); dev = a.dev; torch.set_num_threads(8)
wm = load_watermark("pixelseal", models_dir=HERE / "kit" / "models", device=dev)
allmeta = json.loads((HERE / "release" / a.split / "metadata.json").read_text())
sel = [(i, m) for i, m in enumerate(allmeta) if m["method"] == "pixelseal"]
X = torch.from_numpy(np.stack([np.asarray(Image.open(HERE / "release" / a.split / "images" / f"{m['image_name']}.png").convert("RGB")) for _, m in sel]))
B = torch.tensor([[int(c) for c in m["message"]] for _, m in sel]); idx = np.array([i for i, _ in sel])
tot = np.zeros((len(sel), 128, 128), np.float32); cnt = np.zeros(len(sel), np.float32)
for r, run in enumerate(a.runs):
    cfg = json.loads((Path(a.dir) / run / "args.json").read_text()) if (Path(a.dir) / run / "args.json").exists() else {}
    net = Loc(copy.deepcopy(wm.det).float(), d=cfg.get("fpn", 160), res=cfg.get("res", a.res), agree=cfg.get("agree", 1), ppinit=cfg.get("ppinit", a.ppinit), hr=cfg.get("hr", 0)).to(dev); net.load_state_dict(torch.load(Path(a.dir) / run / "net.pt", map_location=dev)); net.eval(); out = []
    use = np.ones(len(sel), bool) if (a.split != "train" or not a.folds) else ((idx % 5 == 0) | (idx % 5 == a.folds[r])); ui = torch.from_numpy(np.where(use)[0])
    with torch.no_grad():
        for k in range(0, len(ui), 17):
            x = X[ui[k:k + 17]].to(dev).permute(0, 3, 1, 2).float() / 255
            sg = B[ui[k:k + 17]].to(dev).float() * 2 - 1
            if not a.zoom:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled="cuda" in dev): m, _, _ = net(x, sg)
                m = m.float()
            else:                                         # inférence zoomée : tuiles de 256 px (pas 128) agrandies x2, carte = maximum sur les tuiles
                import torch.nn.functional as F
                m = torch.full((len(x), 128, 128), -1e4, device=dev)
                for ty in (0, 128, 256):
                    for tx in (0, 128, 256):
                        t = F.interpolate(x[:, :, ty:ty + 256, tx:tx + 256], size=512, mode="bilinear", align_corners=False)
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled="cuda" in dev): l, _, _ = net(t, sg)
                        l = F.avg_pool2d(l.float()[:, None], 2)[:, 0]; v = m[:, ty // 4:ty // 4 + 64, tx // 4:tx // 4 + 64]; m[:, ty // 4:ty // 4 + 64, tx // 4:tx // 4 + 64] = torch.maximum(v, l)
            out.append(m.cpu().numpy())
    tot[use] += np.concatenate(out); cnt[use] += 1
assert (cnt > 0).all(), "une image n'est couverte par aucun modèle"
lg = tot / cnt[:, None, None]
np.savez_compressed(a.out, names=np.array([m["image_name"] for _, m in sel]), logits=lg.astype(np.float16))
print("écrit", a.out, lg.shape, "fraction filigranée prédite", round(float((lg > a.th).mean()), 3))
if a.split == "train":
    z = np.load(HERE / "release" / "train" / "masks_128.npz"); G = dict(zip(z["names"].tolist(), z["masks"]))
    sc = np.array([miou_image(lg[j] > a.th, G[m["image_name"]]) for j, (_, m) in enumerate(sel)]); full = np.array([bool(m.get("full_watermark")) for _, m in sel]); res = idx % 5 == 0
    for nm, s in (("RÉSERVE", res), ("hors-pli (4 autres cinquièmes)", ~res)):
        print(f"{nm:32s} n={s.sum():3d} mIoU {sc[s].mean():.4f} | pleines {sc[s & full].mean():.4f} | partielles {sc[s & ~full].mean():.4f}")
