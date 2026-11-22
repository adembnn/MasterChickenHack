"""T1 PixelSeal · post-traitement par population (session t1-pixelseal).
Entrée : logits hors-pli du train et logits du test (names, logits). Sortie : les mêmes fichiers, corrigés, où > 0 = filigrané.
Étapes, toutes réglées sur les images HORS réserve (jamais sur index % 5 == 0) :
  1. population : 'pleine' si le décodeur relit > 75 % des bits sur l'image entière, sinon 'partielle' ;
  2. un seuil par population ;
  3. 'billet de loterie' : part minimale de la classe rare (propre dans une pleine, filigranée dans une partielle), prise aux cases les plus sûres ;
  4. rectangle trouvé par décodage (boxsearch.py) : si la boîte relit >= tau des bits, on rend la boîte.
Usage : python post.py --train pred_train.npz --test pred_test.npz --out_train ... --out_test ... [--box 1]"""
import os, sys, json, argparse
os.environ.setdefault("OMP_NUM_THREADS", "8")
from pathlib import Path
import numpy as np, torch
from PIL import Image
HERE = Path(os.environ.get("WMLOC_ROOT", "/p/scratch/training2666/destors1/MasterChickenHack/Hackathon_Watermark_Localisation_Task"))
sys.path.insert(0, str(HERE / "kit"))
from wmloc.utils import miou_image
D = HERE / "adem" / "t1-pixelseal"
ap = argparse.ArgumentParser(); ap.add_argument("--train", required=True); ap.add_argument("--test", default=""); ap.add_argument("--out_train", default=""); ap.add_argument("--out_test", default="")
ap.add_argument("--box", type=int, default=1); ap.add_argument("--sam", type=int, default=1); ap.add_argument("--train_part", default=""); ap.add_argument("--test_part", default=""); ap.add_argument("--dev", default="cuda"); ap.add_argument("--tau", type=float, default=0.0)
a = ap.parse_args()
def bits_acc(split, names):                           # bits justes de l'image entière, mis en cache
    f = D / f"pop_{split}.json"; d = json.loads(f.read_text()) if f.exists() else {}
    miss = [n for n in names if n not in d]
    if miss:
        from wmloc.watermarks import load_watermark
        wm = load_watermark("pixelseal", models_dir=HERE / "kit" / "models", device=a.dev); msg = {m["image_name"]: m["message"] for m in json.loads((HERE / "release" / split / "metadata.json").read_text())}
        for n in miss:
            x = torch.from_numpy(np.asarray(Image.open(HERE / "release" / split / "images" / f"{n}.png").convert("RGB"))).permute(2, 0, 1)[None].float() / 255
            d[n] = float(((wm.logits(x)[0] > 0).cpu().numpy() == np.array([int(c) for c in msg[n]])).mean())
        f.write_text(json.dumps(d))
    return np.array([d[n] for n in names])
def load(p, split, ppart=""):                          # ppart : logits du modèle spécialisé 'partielles', utilisés pour les images de cette population
    z = np.load(p); names = z["names"].tolist(); lg = z["logits"].astype(np.float32); role = z["role"] if "role" in z.files else None
    full = bits_acc(split, names) > 0.75
    if ppart:
        zp = np.load(ppart); d = dict(zip(zp["names"].tolist(), zp["logits"].astype(np.float32)))
        rp = dict(zip(zp["names"].tolist(), zp["role"])) if "role" in zp.files else None      # rôle 2 = image vue à l'entraînement par ce modèle
        role = np.zeros(len(names), int) if role is None else np.array(role)
        for j, n in enumerate(names):
            if not full[j]:
                lg[j] = d[n]; role[j] = 0 if rp is None else rp[n]
    return names, lg, role, full
def boxes(split, names):
    f = D / f"box_{split}.json"
    if not (a.box and f.exists()): return {}
    return {r["name"]: r for r in json.loads(f.read_text()) if "box" in r}
_SAM = {}
def sam_masks(split, name):                           # masques SAM (K x 128 x 128) calculés par sam_masks.py ; vide si absents
    if split not in _SAM:
        f = D / f"sam_{split}.npz"; _SAM[split] = np.load(f) if (a.sam and f.exists()) else None
    S = _SAM[split]
    if S is None or name + "__n" not in S.files: return np.zeros((0, 128, 128), bool)
    k = int(S[name + "__n"]); return np.unpackbits(S[name])[:k * 16384].reshape(k, 128, 128).astype(bool) if k else np.zeros((0, 128, 128), bool)
def sam_snap(l, M):                                   # union gloutonne de masques qui maximise l'IoU attendue d'après p = sigmoïde(l)
    p = (1 / (1 + np.exp(-l))).ravel(); sp = p.sum(); A = M.reshape(len(M), -1); cur = np.zeros(16384, bool); best = 0.0
    for _ in range(6):
        cand = A | cur[None]; inter = cand @ p; sc = inter / (cand.sum(1) + sp - inter + 1e-6); k = int(sc.argmax())
        if sc[k] <= best + 1e-4: break
        best = float(sc[k]); cur = cand[k]
    return cur.reshape(128, 128), best
def smooth(a, k=9):
    from numpy.lib.stride_tricks import sliding_window_view
    return sliding_window_view(np.pad(a, k // 2, mode="edge"), (k, k)).mean((2, 3))
def apply(lg, full, prm, bx, names, split="train"):
    out = np.empty_like(lg)
    for j in range(len(lg)):
        t, q = (prm["t_full"], prm["q_full"]) if full[j] else (prm["t_part"], prm["q_part"]); l = lg[j] - t
        if q > 0:                                     # part minimale de la classe rare, prise sur la carte LISSÉE (9x9) pour rendre une tache d'un seul tenant
            ls = smooth(l)
            if full[j]:
                c = float(np.quantile(ls, q))
                if c > 0: l = np.where(ls <= c, -np.abs(l) - 0.01, l)          # au moins q des cases passent propres
            else:
                c = float(np.quantile(ls, 1 - q))
                if c < 0: l = np.where(ls >= c, np.abs(l) + 0.01, l)           # au moins q des cases passent filigranées
        if not full[j] and prm.get("tsam", 9.0) < 1.0:   # objets des partielles : calage sur les contours SAM
            M = sam_masks(split, names[j])
            if len(M):
                m, e = sam_snap(l, M)
                if e >= prm["tsam"]: l = np.where(m, 6.0, -6.0).astype(np.float32)
        b = bx.get(names[j])
        if b is not None and not full[j] and b["bits"] >= prm["tau"] and b["bits"] - b["glob"] >= 0.35:
            y, x, h, w = b["box"]; m = np.zeros((512, 512), np.float32); m[y:y + h, x:x + w] = 1; l = np.where(m.reshape(128, 4, 128, 4).mean((1, 3)) >= 0.5, 6.0, -6.0).astype(np.float32)
        out[j] = l
    return out
names, lg, role, full = load(a.train, "train", a.train_part)
meta = json.loads((HERE / "release" / "train" / "metadata.json").read_text()); idx = {m["image_name"]: i for i, m in enumerate(meta)}; tf = {m["image_name"]: bool(m.get("full_watermark")) for m in meta}
z = np.load(HERE / "release" / "train" / "masks_128.npz"); G = dict(zip(z["names"].tolist(), z["masks"])); G = np.stack([G[n] for n in names]).astype(bool)
res = np.array([idx[n] % 5 == 0 for n in names]); truefull = np.array([tf[n] for n in names]); ok = np.ones(len(names), bool) if role is None else (role != 2)
tune = ~res & ok; bx = boxes("train", names)
print(f"images : {len(names)} dont réserve {res.sum()}, réglage sur {tune.sum()} ; population bien devinée : {(full == truefull).mean():.3f} ; boîtes disponibles : {len(bx)}")
def sc(prm, sel): o = apply(lg[sel], full[sel], prm, bx, [n for n, s in zip(names, sel) if s]); return np.array([miou_image(o[j] > 0, g) for j, g in enumerate(G[sel])])
def rep(nm, prm):
    s = sc(prm, np.ones(len(names), bool)); r = res & ok
    print(f"{nm:44s} réglage {s[tune].mean():.4f} | RÉSERVE {s[r].mean():.4f} pleines {s[r & truefull].mean():.4f} partielles {s[r & ~truefull].mean():.4f}   {json.dumps({k: round(float(v), 3) for k, v in prm.items()})}")
prm = {"t_full": 0.0, "t_part": 0.0, "q_full": 0.0, "q_part": 0.0, "tau": 9.0, "tsam": 9.0}; rep("0. brut (seuil 0)", prm)
def best(key, grid, sel):
    global prm
    v = [(sc({**prm, key: g}, sel).mean(), g) for g in grid]; prm = {**prm, key: max(v)[1] if max(v)[0] > sc(prm, sel).mean() + 1e-4 else prm[key]}
TG = np.linspace(-3, 3, 25)
best("t_full", TG, tune & full); best("t_part", TG, tune & ~full); rep("1. + un seuil par population", prm)
best("q_full", [0, .01, .02, .03, .04, .06, .08], tune & full); best("q_part", [0, .005, .01, .02, .03, .04], tune & ~full); rep("2. + part minimale de la classe rare", prm)
if a.sam and (D / "sam_train.npz").exists():
    best("tsam", [9.0, 0.1, 0.2, 0.3, 0.4, 0.5], tune & ~full); rep("3. + calage des partielles sur les contours SAM", prm)
if bx:
    prm["tau"] = a.tau or 0.85                        # règle fixe (pas de réglage) : boîte relue à >= 85 % et écart >= 0,35 avec l'image entière
    rep("4. + rectangle trouvé par décodage", prm)
(D / "post_params.json").write_text(json.dumps(prm))
if a.out_train: np.savez_compressed(a.out_train, names=np.array(names), logits=apply(lg, full, prm, bx, names).astype(np.float16)); print("écrit", a.out_train)
if a.test and a.out_test:
    tn, tl, _, tfull = load(a.test, "test", a.test_part); o = apply(tl, tfull, prm, boxes("test", tn), tn, "test")
    np.savez_compressed(a.out_test, names=np.array(tn), logits=o.astype(np.float16)); print("écrit", a.out_test, "| test : part de pleines", round(float(tfull.mean()), 3), "| part filigranée prédite", round(float((o > 0).mean()), 3))
