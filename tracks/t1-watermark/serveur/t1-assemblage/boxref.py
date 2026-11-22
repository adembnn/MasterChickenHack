"""Affinage des boites du decodeur. Le score z de bs.py sature (tous les bits justes sur un plateau de boites) : on remonte la pente d'un critere
NON sature (marge brute du decodeur : moyenne des logits comptes dans le sens du bit attendu) avec un rognage exact, bord par bord.
Mesure sur les rectangles du train : IoU avec la vraie boite avant / apres, pour plusieurs criteres. Puis ecrit les boites affinees (train et test).
Usage : python boxref.py trustmark [--write]"""
import sys, glob, json, time, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); from t1lib import *
sys.path.insert(0, str(HERE / "kit")); from wmloc.watermarks import load_watermark
m = sys.argv[1]; WRITE = "--write" in sys.argv; dev = "cuda"; torch.set_num_threads(4)
wm = load_watermark(m, models_dir=HERE / "kit" / "models", device=dev); RES = {"pixelseal": 256, "trustmark": 245, "stegastamp": 400}[m]
def crops_logits(x, boxes):
    cr = torch.cat([F.interpolate(x[:, :, b[0]:b[2], b[1]:b[3]], size=(RES, RES), mode="bilinear", align_corners=False, antialias=True) for b in boxes])
    out = []
    for i in range(0, len(cr), 256):
        with torch.no_grad(): out.append(wm.logits(cr[i:i + 256]).float().to(dev))
    return torch.cat(out)
def crit(x, boxes, s, kind):
    lg = crops_logits(x, boxes) * s
    if kind == "marge": return lg.mean(1).cpu().numpy()
    if kind == "logvrais": return F.logsigmoid(lg).mean(1).cpu().numpy()
    if kind == "min10": return lg.sort(1).values[:, :10].mean(1).cpu().numpy()          # les 10 bits les moins surs
    if kind == "z": return (lg.sum(1) / lg.pow(2).sum(1).sqrt().clamp_min(1e-9)).cpu().numpy()
def clip(b): y0, x0, y1, x1 = b; y0 = max(0, min(y0, 500)); x0 = max(0, min(x0, 500)); return (y0, x0, max(y0 + 12, min(y1, 512)), max(x0 + 12, min(x1, 512)))
def ascent(x, b, s, kind, steps=(8, 4, 2, 1)):
    b = clip(tuple(int(v) for v in b)); best = float(crit(x, [b], s, kind)[0])
    for st in steps:
        for _ in range(40):
            c = []
            for k in range(4):
                for d in (-st, st): t = list(b); t[k] += d; c.append(clip(tuple(t)))
            for d in (-st, st): c += [clip((b[0] + d, b[1], b[2] + d, b[3])), clip((b[0], b[1] + d, b[2], b[3] + d)), clip((b[0] - d, b[1] - d, b[2] + d, b[3] + d))]
            c = [t for t in set(c) if t != b]; v = crit(x, c, s, kind); j = int(np.argmax(v))
            if v[j] > best + 1e-5: best = float(v[j]); b = c[j]
            else: break
    return b, best
def iou(b, g):
    iy = max(0, min(b[2], g[2]) - max(b[0], g[0])); ix = max(0, min(b[3], g[3]) - max(b[1], g[1])); i = iy * ix
    return i / ((b[2] - b[0]) * (b[3] - b[1]) + (g[2] - g[0]) * (g[3] - g[1]) - i)
def load_boxes(split):
    R = []
    for f in sorted(glob.glob(str(ME / "boxtm" / f"boxes_{m}_{split}_s*.json"))): R += json.load(open(f))
    return [r for r in R if r.get("box") is not None]
meta = {sp: {r["image_name"]: r for r in (load_train(m) if sp == "train" else load_test(m))} for sp in ("train", "test")}
def prep(split, name):
    x = torch.from_numpy(load_img(split, name).copy()).permute(2, 0, 1)[None].float().div(255).to(dev); s = torch.tensor([int(c) for c in meta[split][name]["message"]], device=dev).float() * 2 - 1; return x, s
R = [r for r in load_boxes("train") if not r["full"] and r["rect"] and r["z"] >= 6]; t0 = time.time(); res = {}
for kind in ("marge",):
    I0, I1, up = [], [], []
    for r in R:
        x, s = prep("train", r["name"]); g = r["gt_box"]; b, v = ascent(x, r["box"], s, kind); vg = float(crit(x, [tuple(g)], s, kind)[0])
        I0.append(iou(r["box"], g)); I1.append(iou(b, g)); up.append(v >= vg - 1e-6); res.setdefault(kind, []).append((r["name"], b))
    I0, I1 = np.array(I0), np.array(I1)
    print(f"{m} critere {kind:9s} n={len(R)} | IoU avant {I0.mean():.3f} (mediane {np.median(I0):.3f}) -> apres {I1.mean():.3f} (mediane {np.median(I1):.3f}, part > 0,9 : {(I1 > .9).mean():.2f}) | critere(boite affinee) >= critere(vraie boite) : {np.mean(up):.2f} | {round(time.time() - t0)} s", flush=True)
# decalage restant apres affinage par la marge : systematique ? (le cadre du filigrane est une toile un peu plus grande que le rectangle visible)
B = np.array([b for _, b in res["marge"]], float); G = np.array([r["gt_box"] for r in R], float); D = B - G; hh, ww = B[:, 2] - B[:, 0], B[:, 3] - B[:, 1]; S = np.stack([hh, ww, hh, ww], 1)
print("apres affinage 'marge' : decalage median par bord (px)", np.median(D, 0), "| relatif a la taille", np.median(D / S, 0).round(3), "| ecart absolu median", np.median(np.abs(D), 0))
fold = np.array([meta["train"][r["name"]]["idx"] % 5 for r in R]); P1 = np.zeros_like(B); P2 = np.zeros_like(B); P3 = np.zeros_like(B)
for f in range(5):
    tr = fold != f
    P1[~tr] = B[~tr] - np.median(D[tr], 0); P2[~tr] = B[~tr] - np.median((D / S)[tr], 0) * S[~tr]
    for k in range(4):
        X = np.stack([B[:, k], S[:, k], np.ones(len(B))], 1); c, *_ = np.linalg.lstsq(X[tr], G[tr, k], rcond=None); P3[~tr, k] = X[~tr] @ c
for nom, P in (("decalage median en px", P1), ("decalage median relatif", P2), ("regression lineaire par bord", P3)):
    v = np.array([iou(b, g) for b, g in zip(P, G)]); print(f"   + correction ({nom}, validation croisee) : IoU {v.mean():.3f} (mediane {np.median(v):.3f}, part > 0,9 : {(v > .9).mean():.2f})")
CORR = np.median(D / S, 0)
if WRITE:
    kind = sys.argv[sys.argv.index("--write") + 1]
    for split in ("train", "test"):
        out = []
        for r in load_boxes(split):
            q = dict(name=r["name"], z=r["z"], box0=r["box"])
            if r["z"] >= 5:
                x, s = prep(split, r["name"]); b, v = ascent(x, r["box"], s, kind); b = np.array(b, float); sz = np.array([b[2] - b[0], b[3] - b[1]] * 2)
                q["box_marge"] = [int(t) for t in b]; q["box"] = [int(round(t)) for t in (b - CORR * sz if "--corr" in sys.argv else b)]
            else: q["box"] = [int(v) for v in r["box"]]
            out.append(q)
        json.dump(out, open(ME / "boxtm" / f"boxes_{m}_{split}_affine.json", "w")); print("ecrit", split, len(out), flush=True)
