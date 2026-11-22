"""Candidats "objet" pour les images partielles : masques SAM ViT-H (mode automatique) + test du decodeur region par region.
Pour chaque image probablement partielle (bits justes image entiere < 0,9) :
  - jusqu'a K masques SAM (aire entre 0,2 % et 60 % de l'image), ramenes a la grille officielle 128x128 ;
  - "garder" : on ne garde que la region (reste = couleur moyenne) et on decode -> marge et bits justes par rapport au message connu ;
  - "effacer" : on efface la region et on mesure la chute de marge par rapport a l'image entiere.
Sortie : un fichier par image, samcand/<split>/<nom>.npz (masks packbits [nv,2048], feats [nv,8], m0, acc0) : reprise possible apres coupure.
feats = aire (fraction), iou predit par SAM, stabilite, marge garder, bits garder, chute de marge effacer, chute de bits effacer, aire a 128."""
import sys, argparse, time, numpy as np, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); from t1lib import *
sys.path.insert(0, str(HERE / "kit"))
from wmloc.watermarks import load_watermark
from wmloc.utils import to_tensor
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
ap = argparse.ArgumentParser(); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--K", type=int, default=80); ap.add_argument("--splits", default="train,test"); a = ap.parse_args()
torch.set_num_threads(4); dev = "cuda"; K = a.K
sam = sam_model_registry["vit_h"](checkpoint=str(ME / "weights" / "sam_vit_h_4b8939.pth")).to(dev).eval()
gen = SamAutomaticMaskGenerator(sam, points_per_side=32, points_per_batch=128, pred_iou_thresh=0.80, stability_score_thresh=0.88, min_mask_region_area=0)
wms = {m: load_watermark(m, models_dir=HERE / "kit" / "models", device=dev) for m in METHODS}
def down(m): return m.reshape(128, 4, 128, 4).mean(axis=(1, 3)) >= 0.5
def one(split, r):
    img = load_img(split, r["image_name"]); wm = wms[r["method"]]; bs = 8 if r["method"] == "stegastamp" else 48
    with torch.no_grad(): ms = gen.generate(img)
    ms = [m for m in ms if 0.002 <= m["area"] / 262144 <= 0.6]
    ms = sorted(ms, key=lambda m: -m["predicted_iou"] * m["stability_score"])[:K]; ms = sorted(ms, key=lambda m: -m["area"])
    x = to_tensor(img).to(dev); sgn = torch.tensor([int(c) for c in r["message"]]).float() * 2 - 1; mc = x.mean(dim=(2, 3), keepdim=True)
    def marg(b):
        out = []
        for j in range(0, len(b), bs):
            with torch.no_grad(): out.append(wm.logits(b[j:j + bs]).cpu().float())
        lg = torch.cat(out); return (lg * sgn).mean(1).numpy(), ((((lg > 0).float() * 2 - 1) == sgn).float().mean(1)).numpy()
    m0, a0 = marg(x); MK = np.zeros((len(ms), 2048), np.uint8); FT = np.zeros((len(ms), 8), np.float32)
    if ms:
        S = torch.from_numpy(np.stack([m["segmentation"] for m in ms])).to(dev)[:, None]
        km, ka = marg(torch.where(S, x, mc)); om, oa = marg(torch.where(S, mc, x))
        for k, m in enumerate(ms):
            d = down(m["segmentation"]); MK[k] = np.packbits(d.ravel())
            FT[k] = (m["area"] / 262144, m["predicted_iou"], m["stability_score"], km[k], ka[k], m0[0] - om[k], a0[0] - oa[k], d.mean())
    return dict(masks=MK, feats=FT, m0=np.float32(m0[0]), acc0=np.float32(a0[0]))
for split in a.splits.split(","):
    z = np.load(ME / f"feat_{split}.npz"); acc0 = dict(zip(z["names"].tolist(), z["acc"])); OUT = ME / "samcand" / split; OUT.mkdir(parents=True, exist_ok=True)
    rows = [r for r in (load_train() if split == "train" else load_test()) if acc0[r["image_name"]] < 0.9][a.shard::a.nshards]
    if a.limit: rows = rows[:a.limit]
    t0 = time.time(); done = 0
    for i, r in enumerate(rows):
        f = OUT / f"{r['image_name']}.npz"
        if f.exists(): continue
        for essai in range(30):                                  # la carte est partagee : si la memoire manque, on attend et on reessaie
            try:
                res = one(split, r); break
            except Exception as e:
                print("essai", essai, type(e).__name__, str(e)[:120], flush=True); torch.cuda.empty_cache(); time.sleep(30)
        else: sys.exit("abandon : memoire GPU indisponible")
        np.savez_compressed(f, **res); done += 1
        if done % 20 == 1: print(split, a.shard, i, "/", len(rows), len(res["feats"]), "masques", round(time.time() - t0), "s", flush=True)
    print("SAMCAND fait", split, a.shard, len(rows), round(time.time() - t0), "s", flush=True)
