"""t3-verificateur : planche d'images pour tester a l'oeil l'hypothese 'les images a oublier que le gold rate sont des images dont l'etiquette ne correspond pas au contenu'.
Pour une paire (c -> c2) : ligne 1 = images etiquetees c que les modeles redistilles profonds (6 et 8 blocs, hors pli) classent c ; ligne 2 = images etiquetees c qu'ils classent c2 ;
ligne 3 = images etiquetees c2 qu'ils classent c2. usage : python look.py"""
import os, numpy as np
from PIL import Image, ImageDraw
ROOT = os.environ['T3ROOT']; ME = ROOT + '/t3-verificateur'; OOF = ME + '/oof'
fg = np.load(ROOT + '/forget_images.npz')['images']; O = np.load(ME + '/out/orig.npz')['exact']; y = O.argmax(1); n = len(y)
Z = [np.load(f'{OOF}/redistill_k6m4_logits_s{i}.npy') for i in range(4)] + [np.load(f'{OOF}/redistill_k8m2_logits_s{i}.npy') for i in range(2)]
P = np.stack([z.argmax(1) for z in Z]); votes = np.zeros((n, 100), int)
for p in P: votes[np.arange(n), p] += 1
top = votes.argmax(1); nt = votes.max(1); ok = (top == y) & (nt >= 5); bad = (top != y) & (nt >= 5)
print(f'images ou >= 5 modeles profonds sur 6 donnent la classe d orig : {ok.mean():.3f} ; une meme AUTRE classe : {bad.mean():.3f} ; pas de majorite de 5 : {1 - ok.mean() - bad.mean():.3f}')
from collections import Counter
cnt = Counter(zip(y[bad].tolist(), top[bad].tolist())); print('paires (etiquette -> classe des modeles profonds) les plus frequentes :', cnt.most_common(12))
per = Counter(y[bad].tolist()); tot = Counter(y.tolist()); print('part d images "autre classe" par etiquette (10 plus fortes) :', sorted(((per[c] / tot[c], c, tot[c]) for c in per), reverse=True)[:10])
dst = Counter(top[bad].tolist()); print('classes de destination les plus frequentes :', dst.most_common(10), '; nb de classes de destination', len(dst))
pairs = [p for p, _ in cnt.most_common(40)]; sel = []
for c, c2 in pairs:
    if (ok & (y == c)).sum() >= 6 and (ok & (y == c2)).sum() >= 6 and (bad & (y == c) & (top == c2)).sum() >= 6: sel.append((c, c2))
    if len(sel) == 3: break
print('paires montrees :', sel); T = 112; K = 8
for pi, (c, c2) in enumerate(sel):
    rows = [np.where(ok & (y == c))[0][:K], np.where(bad & (y == c) & (top == c2))[0][:K], np.where(ok & (y == c2))[0][:K]]; lab = [f'etiquette {c}, modeles : {c}', f'etiquette {c}, modeles : {c2}', f'etiquette {c2}, modeles : {c2}']
    im = Image.new('RGB', (K * T, 3 * (T + 14)), 'white'); d = ImageDraw.Draw(im)
    for r, idx in enumerate(rows):
        d.text((2, r * (T + 14)), lab[r], fill='black')
        for k, i in enumerate(idx): im.paste(Image.fromarray(fg[i]).resize((T, T)), (k * T, r * (T + 14) + 13))
    im.save(f'{ME}/out/look_{pi}.png')
