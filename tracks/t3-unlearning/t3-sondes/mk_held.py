"""Regenere sur une autre machine les copies tenues a l'ecart de t3-methodes (cache.py, meme code, meme graine 1234, meme ordre) : held_*.npy dans MON cache.
Controle : les logits d'orig sur ces copies doivent retrouver ceux mesures sur la premiere H200."""
import sys, os, io; sys.path.insert(0, os.path.expanduser('~/eac/t3/t3-sondes/code'))
os.environ.setdefault('T3ME', os.path.expanduser('~/eac/t3/t3-sondes'))
from common import *
from PIL import Image
orig = load_orig(); fg = load_forget(); N = len(fg); lg0 = logits_u8(orig, fg); yhat = lg0.argmax(1)
g = gen(1234); outs = {k: [] for k in ['jpeg50', 'rot15', 'cropflip', 'mild', 'color_strong', 'blur_strong']}
for i in range(0, N, 250):
    x = to01(fg[i:i + 250]); B = x.shape[0]
    v, _ = affine(x, g, rot=15., zoom=1.1); outs['rot15'].append(to_u8(v).cpu())
    v, _ = affine(x, g, scale=(0.5, 1.), flip=True); outs['cropflip'].append(to_u8(v).cpu())
    v, _ = affine(x, g, scale=(0.8, 1.)); v = color(v, g, 0.15, 0.15, 0.15); v = (v + torch.randn(v.shape, device=dev, generator=g) * 0.02).clamp(0, 1); outs['mild'].append(to_u8(v).cpu())
    outs['color_strong'].append(to_u8(color(x, g, 0.4, 0.4, 0.4, 0.1)).cpu())
    outs['blur_strong'].append(to_u8(blur(x, U(B, 1.5, 2.5, g))).cpu())
    jp = []
    for im in fg[i:i + 250].numpy():
        bb = io.BytesIO(); Image.fromarray(im).save(bb, 'JPEG', quality=50); jp.append(np.array(Image.open(bb).convert('RGB')))
    outs['jpeg50'].append(torch.from_numpy(np.stack(jp)))
d = {'exact': lg0.numpy()}
for k, v in outs.items():
    u8 = torch.cat(v).numpy(); np.save(f'{CACHE}/held_{k}.npy', u8); lg = logits_u8(orig, u8); d['held_' + k] = lg.numpy(); ce = F.cross_entropy(lg, yhat, reduction='none')
    print(f'orig sur {k}: accord {(lg.argmax(1) == yhat).float().mean():.4f} CE moy {ce.mean():.4f} sd {ce.std():.4f}', flush=True)
os.makedirs(ADEM + '/logits', exist_ok=True); np.savez(ADEM + '/logits/orig_h200b.npz', **d)
