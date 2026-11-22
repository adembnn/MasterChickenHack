"""Pour nommer les 100 classes d'orig (dont les 55 absentes du forget set) : logits d'orig sur les 56 000 images SD-1.4 du simulateur
(100 classes ImageNet connues), puis pour chaque classe k d'orig : les classes ImageNet dont les images activent le plus k, et une planche des 10 images les plus activantes.
Sorties : meas/topact.json, meas/topact_<a>_<b>.jpg"""
import nm, json, numpy as np, torch, torch.nn.functional as F, os
from PIL import Image, ImageDraw
S = os.environ.get('SIMDIR', os.path.expanduser('~/eac/t3/t3-simulateur')); X = np.load(S + '/data/x.npy', mmap_mode='r'); Y = np.load(S + '/data/y.npy')
names = [c['name'].split(',')[0] for c in json.load(open(S + '/data/classes.json'))['classes']]
p = nm.MEAS + '/sim_logits.npy'
if os.path.exists(p): Z = np.load(p)
else:
    m = nm.orig(); out = []
    for i in range(0, len(X), 2000):
        out.append(nm.logits(m, X[i:i + 2000]).numpy()); print(i, flush=True)
    Z = np.concatenate(out); np.save(p, Z)
P = torch.softmax(torch.from_numpy(Z).double(), 1).numpy(); pred = Z.argmax(1); present = set(np.load(nm.MEAS + '/forget_logits.npy').argmax(1).tolist())
print('max-prob moyenne sur images SD-1.4 :', P.max(1).mean().round(4), ' classes prédites :', len(set(pred.tolist())))
res = {}
for k in range(100):
    sel = np.where(pred == k)[0]; cnt = np.bincount(Y[sel], minlength=100); top = np.argsort(-cnt)[:3]
    res[k] = dict(present=k in present, n_pred=int(len(sel)), top_imagenet=[(names[t], int(cnt[t])) for t in top if cnt[t] > 0], best=np.argsort(-P[:, k])[:10].tolist(), pmax=float(np.sort(P[:, k])[-10:].mean()))
    print(k, 'PRESENT' if k in present else 'absent ', res[k]['n_pred'], res[k]['top_imagenet'], round(res[k]['pmax'], 3), flush=True)
json.dump(res, open(nm.MEAS + '/topact.json', 'w'))
for a in range(0, 100, 25):
    W = Image.new('RGB', (10 * 114 + 60, 25 * 114), 'white'); d = ImageDraw.Draw(W)
    for r, k in enumerate(range(a, a + 25)):
        d.text((2, r * 114 + 40), f"c{k}\n{'P' if k in present else 'abs'}", fill='black')
        for j, i in enumerate(res[k]['best']): W.paste(Image.fromarray(np.asarray(X[i])).resize((112, 112)), (60 + j * 114, r * 114))
    W.save(f'{nm.MEAS}/topact_{a}_{a + 24}.jpg', quality=88)
