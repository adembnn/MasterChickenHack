"""Les images du hackathon viennent-elles de COCO val2017 ? Vignettes 24x24 en gris, normalisées, similarité cosinus au plus proche voisin."""
import sys, json, glob, os
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
COCO = sys.argv[1] if len(sys.argv) > 1 else "/home/adem/eac/shared_data/coco/val2017"; REL = "/home/adem/eac/t1/release"
def norm(a):
    a = a.astype(np.float32).ravel(); a -= a.mean(); return a / (np.linalg.norm(a) + 1e-6)
def thumb_hack(p): return norm(np.asarray(Image.open(p).convert("L").resize((24, 24), Image.BILINEAR)))
def thumb_coco(p):
    im = Image.open(p).convert("L"); w, h = im.size; s = min(w, h); c = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return norm(np.asarray(im.resize((24, 24), Image.BILINEAR))), norm(np.asarray(c.resize((24, 24), Image.BILINEAR)))
if __name__ == "__main__":
    out = {}
    for side in ("train", "test"):
        meta = json.load(open(f"{REL}/{side}/metadata.json")); hp = [f"{REL}/{side}/images/{m['image_name']}.png" for m in meta]; hp = [p for p in hp if os.path.exists(p)]
        if not hp: print(side, "images introuvables", f"{REL}/{side}/images"); continue
        print(side, len(hp), "images ; taille", Image.open(hp[0]).size, Image.open(hp[0]).mode)
        with ProcessPoolExecutor(16) as ex: H = np.stack(list(ex.map(thumb_hack, hp, chunksize=32)))
        out[side] = (meta, hp, H)
    cp = sorted(glob.glob(COCO + "/*.jpg")); print("COCO", len(cp), "images")
    with ProcessPoolExecutor(16) as ex: T = list(ex.map(thumb_coco, cp, chunksize=64))
    C1 = np.stack([t[0] for t in T]); C2 = np.stack([t[1] for t in T])
    for side, (meta, hp, H) in out.items():
        s = np.maximum(H @ C1.T, H @ C2.T); o = np.sort(s, 1); best = o[:, -1]; second = o[:, -2]; j = s.argmax(1)
        print(f"{side} : similarité au plus proche voisin COCO : médiane {np.median(best):.3f}, 5e centile {np.percentile(best, 5):.3f} ; deuxième voisin médiane {np.median(second):.3f} ; part >= 0,95 : {np.mean(best >= .95):.3f} ; part >= 0,90 : {np.mean(best >= .90):.3f} ; voisins distincts {len(set(j.tolist()))}")
        for me in ("pixelseal", "trustmark", "stegastamp"):
            k = np.array([m["method"] == me for m in meta][:len(best)]); print(f"   {me:10s} n={k.sum()} part >= 0,95 : {np.mean(best[k] >= .95):.3f} ; médiane {np.median(best[k]):.3f}")
        # la même image d'origine sert-elle à plusieurs images du hackathon (par exemple une fois par filigrane) ?
        ok = best >= .95; u, cnt = np.unique(j[ok], return_counts=True); print(f"   images COCO retrouvées {len(u)} ; utilisées 1 fois {np.sum(cnt == 1)}, 2 fois {np.sum(cnt == 2)}, 3 fois et plus {np.sum(cnt >= 3)}")
        np.savez_compressed(f"cocomatch_{side}.npz", names=np.array([m["image_name"] for m in meta][:len(best)]), coco=np.array([os.path.basename(cp[x]) for x in j]), sim=best, second=second)
    # doublons internes : deux images du hackathon issues de la même image d'origine ?
    if "train" in out and "test" in out:
        Ha, Hb = out["train"][2], out["test"][2]; s = Ha @ Hb.T; print(f"train x test : paires de similarité >= 0,95 : {(s >= .95).sum()} ; images du test ayant un jumeau dans le train : {(s.max(0) >= .95).sum()} sur {len(Hb)}")
        s = Ha @ Ha.T; np.fill_diagonal(s, 0); print(f"train x train : images du train ayant un jumeau dans le train : {(s.max(0) >= .95).sum()} sur {len(Ha)}")
