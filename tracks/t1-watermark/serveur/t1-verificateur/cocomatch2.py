"""Les images du hackathon viennent-elles d'un jeu COCO (archive zip lue sans la décompresser) ? Vignettes 24x24 en gris, normalisées ; pour chaque
image COCO : image entière écrasée en carré + carrés gauche / centre / droite (ou haut / centre / bas). Similarité cosinus au plus proche voisin."""
import sys, json, os, zipfile, io
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
ZIP = sys.argv[1]; TAG = sys.argv[2] if len(sys.argv) > 2 else "train2017"; REL = "/home/adem/eac/t1/release"; S = 24
def norm(a):
    a = a.astype(np.float32).ravel(); a -= a.mean(); return a / (np.linalg.norm(a) + 1e-6)
def thumb_hack(p): return norm(np.asarray(Image.open(p).convert("L").resize((S, S), Image.BILINEAR)))
def variants(im):
    w, h = im.size; s = min(w, h); out = [im.resize((S, S), Image.BILINEAR)]
    for f in (0.0, 0.5, 1.0):
        x0 = int(round((w - s) * f)); y0 = int(round((h - s) * f)); out.append(im.crop((x0, y0, x0 + s, y0 + s)).resize((S, S), Image.BILINEAR))
    return np.stack([norm(np.asarray(o)) for o in out])
def chunk(names):
    zf = zipfile.ZipFile(ZIP); out = []
    for n in names:
        try:
            im = Image.open(io.BytesIO(zf.read(n))); im.draft("L", (S * 4, S * 4)); out.append(variants(im.convert("L")))
        except Exception: out.append(np.zeros((4, S * S), np.float32))
    return np.stack(out)
if __name__ == "__main__":
    H = {}
    for side in ("train", "test"):
        meta = json.load(open(f"{REL}/{side}/metadata.json")); hp = [f"{REL}/{side}/images/{m['image_name']}.png" for m in meta]
        with ProcessPoolExecutor(16) as ex: H[side] = (meta, np.stack(list(ex.map(thumb_hack, hp, chunksize=32))))
    names = sorted(n for n in zipfile.ZipFile(ZIP).namelist() if n.lower().endswith(".jpg")); print(TAG, len(names), "images dans l'archive", flush=True)
    parts = [names[i:i + 500] for i in range(0, len(names), 500)]
    with ProcessPoolExecutor(16) as ex: C = np.concatenate(list(ex.map(chunk, parts)))             # N x 4 x 576
    N = len(C); Cf = C.reshape(N * 4, -1)
    for side, (meta, Hh) in H.items():
        best = np.full(len(Hh), -1.0); arg = np.zeros(len(Hh), int); second = np.full(len(Hh), -1.0)
        for i in range(0, N * 4, 40000):
            s = Hh @ Cf[i:i + 40000].T; m = s.max(1); a = s.argmax(1) + i; up = m > best; second = np.where(up, best, np.maximum(second, m)); arg = np.where(up, a, arg); best = np.where(up, m, best)
        img = arg // 4; var = arg % 4; me = np.array([m["method"] for m in meta])
        print(f"{side} ({len(Hh)}) : similarité au plus proche voisin : médiane {np.median(best):.3f}, 95e centile {np.percentile(best, 95):.3f}, max {best.max():.3f} ; part >= 0,95 : {np.mean(best >= .95):.3f} ; >= 0,90 : {np.mean(best >= .90):.3f} ; >= 0,85 : {np.mean(best >= .85):.3f}")
        for m_ in ("pixelseal", "trustmark", "stegastamp"): k = me == m_; print(f"   {m_:10s} >= 0,95 : {np.mean(best[k] >= .95):.3f} ; >= 0,90 : {np.mean(best[k] >= .90):.3f} ; médiane {np.median(best[k]):.3f}")
        if side == "train":
            full = np.array([m["full_watermark"] for m in meta]); print(f"   pleines >= 0,90 : {np.mean(best[full] >= .90):.3f} ; partielles >= 0,90 : {np.mean(best[~full] >= .90):.3f}")
        ok = best >= .90; print(f"   variante retenue parmi les trouvées (0 écrasée, 1 gauche/haut, 2 centre, 3 droite/bas) : {np.bincount(var[ok], minlength=4).tolist()} ; images COCO distinctes {len(set(img[ok].tolist()))} pour {ok.sum()} images")
        np.savez_compressed(f"cocomatch_{TAG}_{side}.npz", names=np.array([m["image_name"] for m in meta]), coco=np.array([os.path.basename(names[x]) for x in img]), var=var, sim=best, second=second)
