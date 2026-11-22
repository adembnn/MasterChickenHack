"""TrustMark : le décodeur comme DÉTECTEUR de rectangle. Dans une image partielle « rectangle », rogner la bonne boîte fait relire le message connu.
On balaie des fenêtres de plusieurs tailles et formes, on garde celle où les bits décodés s'accordent le mieux au message, on affine, et on rend la boîte.
Mesure sur les vraies partielles hors réserve : taux de détection, IoU de la boîte, mIoU officiel si on rend la boîte ; et fausses détections sur les objets."""
import sys, json, time, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from torchvision.ops import roi_align
ROOT = Path(__file__).resolve().parent; HERE = ROOT.parent if (ROOT.parent / "kit").exists() else ROOT.parent.parent; CACHE = ROOT / "cache"; sys.path.insert(0, str(HERE / "kit"))
from wmloc.watermarks import load_watermark
from wmloc.utils import miou_image, downsample_mask
dev = "cuda"; method = "trustmark"; wm = load_watermark(method, models_dir=HERE / "kit" / "models", device=dev); net = wm.net.half(); RES = wm.resolution; TAU = 9.0
trm = json.loads((HERE / "release/train/metadata.json").read_text()); RT = np.load(CACHE / "REAL_train.npy", mmap_mode="r"); Y = np.load(CACHE / "Y128_train.npy"); st = {r["i"]: r for r in json.load(open(ROOT / f"stats_{method}.json"))}

def windows():
    out = []
    for S in (36, 48, 64, 88, 120, 160, 220, 300, 400):
        for ar in (0.5, 0.7, 1.0, 1.43, 2.0):
            h = int(round(S * ar ** 0.5)); w = int(round(S / ar ** 0.5))
            if h > 512 or w > 512: continue
            sy = max(6, h // 4); sx = max(6, w // 4)
            for y in list(range(0, 512 - h, sy)) + [512 - h]:
                for x in list(range(0, 512 - w, sx)) + [512 - w]: out.append((y, x, y + h, x + w))
    return np.array(sorted(set(out)), np.float32)
WIN = windows(); print("fenêtres par image :", len(WIN), flush=True)

@torch.no_grad()
def score_boxes(img, boxes, sgn, bs=384):               # img : 1x3x512x512 dans [0,1] ; boxes N x 4 (y0, x0, y1, x1) ; renvoie accord doux et bits justes
    pyr = [img, F.avg_pool2d(img, 2)]; out_s, out_a = [], []
    for j in range(0, len(boxes), bs):
        b = torch.as_tensor(boxes[j:j + bs], device=dev).float(); big = ((b[:, 2] - b[:, 0]) > 1.6 * RES) | ((b[:, 3] - b[:, 1]) > 1.6 * RES); crops = torch.empty(len(b), 3, RES, RES, device=dev)
        for lvl, sel in ((0, ~big), (1, big)):
            if sel.any():
                q = b[sel] / (2 ** lvl); rois = torch.cat([torch.zeros(len(q), 1, device=dev), q[:, [1, 0, 3, 2]]], 1); crops[sel] = roi_align(pyr[lvl], rois, (RES, RES), spatial_scale=1.0, sampling_ratio=2, aligned=True)
        lg = net((crops * 2 - 1).half()).float() * sgn[None]; out_s.append(torch.tanh(lg / TAU).mean(1)); out_a.append((lg > 0).float().mean(1))
    return torch.cat(out_s).cpu().numpy(), torch.cat(out_a).cpu().numpy()

def refine(img, box, sgn, steps=(16, 8, 4, 2), passes=2):
    best = np.array(box, np.float32); bs_, ba = score_boxes(img, best[None], sgn); bs_, ba = float(bs_[0]), float(ba[0])
    for stp in steps:
        for _ in range(passes):
            moved = False
            for e in range(4):
                c = []
                for d in (-stp, stp):
                    b = best.copy(); b[e] += d; b = np.clip(b, 0, 512)
                    if b[2] - b[0] >= 12 and b[3] - b[1] >= 12 and max(b[2] - b[0], b[3] - b[1]) / min(b[2] - b[0], b[3] - b[1]) <= 2.3: c.append(b)
                if not c: continue
                s_, a_ = score_boxes(img, np.stack(c), sgn); k = int(s_.argmax())
                if s_[k] > bs_ + 1e-4: best, bs_, ba = c[k], float(s_[k]), float(a_[k]); moved = True
            if not moved: break
    return best, bs_, ba
def box_mask(b, shrink=0):
    m = np.zeros((512, 512), np.uint8); y0, x0, y1, x1 = [int(round(v)) for v in b]; m[y0 + shrink:y1 - shrink, x0 + shrink:x1 - shrink] = 1; return downsample_mask(m)
def iou(a, b):
    ih = max(0, min(a[2], b[2]) - max(a[0], b[0])); iw = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ih * iw; return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)

N = int(sys.argv[1]); which = sys.argv[2] if len(sys.argv) > 2 else "hors"
ids = [i for i, m in enumerate(trm) if m["method"] == method and not m["full_watermark"] and ((i % 5 != 0) if which == "hors" else (i % 5 == 0))]
rows = []; t0 = time.time()
for i in ids[:N]:
    img = torch.from_numpy(np.ascontiguousarray(RT[i])).permute(2, 0, 1)[None].float().div(255).to(dev); sgn = torch.tensor([1.0 if c == "1" else -1.0 for c in trm[i]["message"]], device=dev)
    s_, a_ = score_boxes(img, WIN, sgn); k = int(s_.argmax()); b, bs_, ba = refine(img, WIN[k], sgn); r = st[i]; isrect = r["rect"] > 0.98; gt = (r["y0"], r["x0"], r["y0"] + r["h"], r["x0"] + r["w"])
    rows.append(dict(i=i, rect=isrect, frac=r["frac"], acc0=float(a_[k]), acc=ba, soft=bs_, iou0=iou(WIN[k], gt), iou=iou(b, gt), box=[float(v) for v in b],
                     miou=[miou_image(box_mask(b, sh), Y[i]) for sh in (0, 2, 4, 6)], vide=miou_image(np.zeros((128, 128)), Y[i])))
json.dump(rows, open(ROOT / f"rectdet_{which}.json", "w")); print("temps par image", round((time.time() - t0) / len(rows), 1), "s")
R_ = [r for r in rows if r["rect"]]; O = [r for r in rows if not r["rect"]]
print(f"RECTANGLES ({len(R_)}) : bits justes de la meilleure fenêtre, quantiles 10/50/90 : {np.round(np.quantile([r['acc'] for r in R_], [.1, .5, .9]), 2)}")
print(f"OBJETS ({len(O)})     : bits justes de la meilleure fenêtre, quantiles 10/50/90 : {np.round(np.quantile([r['acc'] for r in O], [.1, .5, .9]), 2)}")
for th in (0.75, 0.8, 0.85, 0.9):
    d = [r for r in R_ if r["acc"] >= th]; fp = [r for r in O if r["acc"] >= th]
    if d: print(f"seuil {th} : rectangles détectés {len(d)}/{len(R_)}, fausses détections sur objets {len(fp)}/{len(O)} ; IoU boîte médiane {np.median([r['iou'] for r in d]):.2f} (avant affinage {np.median([r['iou0'] for r in d]):.2f}), part à IoU > 0,7 : {np.mean([r['iou'] > .7 for r in d]):.2f} ; "
                f"mIoU si on rend la boîte, rognée de 0/2/4/6 px : {np.round(np.mean([r['miou'] for r in d], 0), 3)} ; mIoU d'un masque vide sur ces images : {np.mean([r['vide'] for r in d]):.3f}")
print("petits rectangles (< 3 %) détectés au seuil 0,8 :", sum(r["acc"] >= .8 for r in R_ if r["frac"] < .03), "/", sum(r["frac"] < .03 for r in R_))
