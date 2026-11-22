import os, json, numpy as np
from PIL import Image
R = os.path.expanduser('~/eac/t1/release')
md = {s: json.load(open(f'{R}/{s}/metadata.json')) for s in ('train', 'test')}
m0 = md['train']; print('metadata type', type(m0).__name__, str(m0)[:600])
names, split = [], []
for s in ('train', 'test'):
    for f in sorted(os.listdir(f'{R}/{s}/images')): names.append(f'{R}/{s}/images/{f}'); split.append(s)
print(len(names), 'images')
T = []
for p in names:
    im = Image.open(p).convert('L'); 
    if len(T) == 0: print('taille', Image.open(p).size, Image.open(p).mode)
    a = np.asarray(im.resize((24, 24), Image.BILINEAR), dtype=np.float32).ravel(); a = (a - a.mean()) / (a.std() + 1e-6); T.append(a)
T = np.stack(T); C = T @ T.T / T.shape[1]; np.fill_diagonal(C, -1)
for thr in (0.999, 0.99, 0.97, 0.95, 0.90):
    i, j = np.where(np.triu(C, 1) > thr); print(f'paires corr > {thr}: {len(i)} ; images concernees {len(set(i) | set(j))}')
i, j = np.where(np.triu(C, 1) > 0.97)
kinds = {}
for a, b in zip(i, j):
    k = (split[a], split[b]); kinds[k] = kinds.get(k, 0) + 1
print('par couple de jeux', kinds)
for a, b in list(zip(i, j))[:12]: print(round(float(C[a, b]), 4), os.path.basename(names[a]), split[a], '|', os.path.basename(names[b]), split[b])
best = C.max(1); print('quantiles de la meilleure correlation par image', np.quantile(best, [0.1, 0.5, 0.9, 0.99]).round(3))
np.save(os.path.expanduser('~/eac/t1/dup_C.npy'), C.astype(np.float16)); json.dump(names, open(os.path.expanduser('~/eac/t1/dup_names.json'), 'w'))
