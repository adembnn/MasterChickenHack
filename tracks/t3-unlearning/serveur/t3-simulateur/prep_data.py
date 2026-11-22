# Jeu du simulateur : 100 classes d'images generees (SD 1.4, classes ImageNet), regroupees en familles voisines.
# Sortie : data/x.npy uint8 [N,224,224,3], data/y.npy, data/classes.json
import os, io, sys, json, tarfile, numpy as np
from multiprocessing import Pool
from huggingface_hub import hf_hub_download, list_repo_files
from timm.data import ImageNetInfo
from PIL import Image
REPO = 'ChristophSchuhmann/Imagenet-1k-SD-1.4'
PER = int(os.environ.get('PER', 560)); OUT = 'data'; os.makedirs(OUT, exist_ok=True)
GROUPS = [  # mots-cles -> familles de classes voisines (ordre = indices contigus, comme dans le reel)
 ['fox'], ['hare', 'rabbit'], ['lamp', 'lantern', 'candle', 'torch', 'spotlight', 'beacon'],
 ['mushroom', 'agaric', 'bolete', 'gyromitra', 'stinkhorn', 'earthstar', 'hen-of-the-woods'], ['pot,', 'flowerpot', 'teapot', 'coffeepot'],
 ['boat', 'ship', 'canoe', 'schooner', 'catamaran', 'trimaran', 'yawl', 'gondola', 'liner'],
 ['chair', 'throne', 'bench', 'couch'], ['windmill', 'water tower', 'church', 'barn', 'boathouse'], ['quilt', 'blanket', 'basket', 'hamper'],
 ['shoe', 'boot', 'sandal', 'clog', 'loafer'], ['bread', 'bagel', 'pretzel', 'loaf', 'dough'], ['kite', 'umbrella', 'parachute', 'balloon'],
 ['apiary', 'birdhouse', 'honeycomb'], ['envelope', 'book jacket', 'binder', 'notebook'], ['anchor', 'sundial', 'clock', 'barometer', 'compass'],
 ['bottle'], ['guitar', 'violin', 'cello', 'banjo'], ['bridge', 'viaduct', 'pier'], ['truck', 'van', 'wagon'], ['cup', 'mug', 'goblet', 'pitcher'],
 ['bag', 'backpack', 'purse'], ['hat', 'bonnet', 'sombrero', 'helmet'], ['bear'], ['owl', 'eagle', 'vulture'], ['butterfly'], ]
info = ImageNetInfo(); wn = info.label_names(); desc = [info.label_descriptions()[i] if isinstance(info.label_descriptions(), list) else '' for i in range(1000)]
files = set(list_repo_files(REPO, repo_type='dataset'))
chosen, used = [], set()
for gi, kws in enumerate(GROUPS):
    for i in range(1000):
        d = desc[i].lower()
        if i in used or f'{wn[i]}.tar' not in files: continue
        if any(k in d for k in kws):
            chosen.append((i, gi)); used.add(i)
chosen = chosen[:100]
print(len(chosen), 'classes'); assert len(chosen) == 100, len(chosen)
for c, (i, gi) in enumerate(chosen): print(c, gi, wn[i], desc[i][:50])
def load_class(a):
    c, (i, gi) = a
    p = hf_hub_download(REPO, f'{wn[i]}.tar', repo_type='dataset')
    xs = []
    with tarfile.open(p) as t:
        ms = sorted([m for m in t.getmembers() if m.isfile() and m.name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))], key=lambda m: m.name)
        if c == 0: print('exemple membres', [m.name for m in ms[:3]], 'n', len(ms), flush=True)
        for m in ms[:PER]:
            try:
                im = Image.open(io.BytesIO(t.extractfile(m).read())).convert('RGB')
                if c == 0 and len(xs) == 0: print('taille source', im.size, flush=True)
                w, h = im.size; s = min(w, h); im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((224, 224), Image.BICUBIC)
                xs.append(np.asarray(im, dtype=np.uint8))
            except Exception as e: pass
    os.remove(os.path.realpath(p))
    x = np.stack(xs); np.save(f'{OUT}/c{c:03d}.npy', x); print('classe', c, wn[i], x.shape, flush=True)
    return len(xs)
if __name__ == '__main__':
    with Pool(4) as pool: ns = pool.map(load_class, list(enumerate(chosen)), chunksize=1)
    n = min(ns); print('min par classe', n, 'max', max(ns))
    X = np.lib.format.open_memmap(f'{OUT}/x.npy', mode='w+', dtype=np.uint8, shape=(100 * n, 224, 224, 3)); Y = np.zeros(100 * n, dtype=np.int64)
    for c in range(100):
        X[c * n:(c + 1) * n] = np.load(f'{OUT}/c{c:03d}.npy')[:n]; Y[c * n:(c + 1) * n] = c; os.remove(f'{OUT}/c{c:03d}.npy')
    X.flush(); np.save(f'{OUT}/y.npy', Y)
    json.dump({'per_class': n, 'classes': [{'c': c, 'group': gi, 'wnid': wn[i], 'name': desc[i]} for c, (i, gi) in enumerate(chosen)]}, open(f'{OUT}/classes.json', 'w'), indent=1)
    print('OK', X.shape)
