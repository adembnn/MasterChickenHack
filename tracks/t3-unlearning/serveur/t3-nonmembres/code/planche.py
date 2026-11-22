"""Planche de contrôle visuel : pour chaque classe, paires (image source du forget set | image générée), avec la prédiction d'orig si eval disponible.
usage : planche.py TAG [n_par_classe=3] [classes_par_planche=15]  -> meas/planche_<TAG>_<k>.jpg"""
import nm, sys, os, numpy as np
from PIL import Image, ImageDraw
tag = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 3; per = int(sys.argv[3]) if len(sys.argv) > 3 else 15
X = nm.forget(); y = np.load(nm.MEAS + '/forget_logits.npy').argmax(1); sub = np.load(nm.MEAS + '/subset.npy'); S = 180
have = [i for i in sub if os.path.exists(f'{nm.IMG}/{tag}/full/{i}.jpg')]; cls = sorted(set(y[have].tolist()))
pred = {}
p = f'{nm.MEAS}/logits_{tag}.npz'
if os.path.exists(p): d = np.load(p); pred = dict(zip(d['src'].tolist(), d['propre'].argmax(1).tolist()))
for k in range(0, len(cls), per):
    cc = cls[k:k + per]; W = Image.new('RGB', (40 + n * (2 * S + 8), len(cc) * (S + 2)), 'white'); dr = ImageDraw.Draw(W)
    for r, c in enumerate(cc):
        dr.text((2, r * (S + 2) + 80), f'c{c}', fill='black')
        for j, i in enumerate([i for i in have if y[i] == c][:n]):
            x0 = 40 + j * (2 * S + 8); W.paste(Image.fromarray(X[i]).resize((S, S)), (x0, r * (S + 2)))
            W.paste(Image.open(f'{nm.IMG}/{tag}/full/{i}.jpg').convert('RGB').resize((S, S), Image.LANCZOS), (x0 + S, r * (S + 2)))
            if i in pred: dr.rectangle((x0 + S, r * (S + 2), x0 + S + 34, r * (S + 2) + 12), fill='black'); dr.text((x0 + S + 2, r * (S + 2)), f'{pred[i]}', fill='lime' if pred[i] == c else 'red')
    W.save(f'{nm.MEAS}/planche_{tag}_{k // per}.jpg', quality=85); print('planche', tag, k // per, cc)
