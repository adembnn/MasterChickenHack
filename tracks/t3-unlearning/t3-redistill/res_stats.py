"""t3-redistill etape 1 : couverture de classes et confiance d'orig sur chaque reservoir candidat."""
import sys, os, glob
sys.path.insert(0, os.path.expanduser('~/eac/t3/t3-methodes/code'))
from common import *
ME = BASE + '/t3-redistill'
ref = load_orig(); fgd = load_forget(); lg0 = torch.from_numpy(np.load(f'{CACHE}/exact_orig_logits.npy')); yhat = lg0.argmax(1); fc = set(yhat.unique().tolist())
print(f'classes du jeu a oublier : {len(fc)}')
print('| reservoir | images | classes couvertes /100 (>=1 image) | >=10 images | part des images dans les 45 classes | confiance orig (max-prob T=1) moy / mediane | part max-prob > 0,9 | entropie moy (nats) | precision si etiquette connue |')
print('|---|---|---|---|---|---|---|---|---|')


def row(name, lg, y=None):
    p = F.softmax(lg, 1); mp, am = p.max(1); cnt = torch.bincount(am, minlength=100); ent = -(p * p.clamp_min(1e-12).log()).sum(1)
    acc = f'{(am == y).float().mean():.3f}' if y is not None else '-'
    print(f'| {name} | {len(lg)} | {(cnt > 0).sum().item()} | {(cnt >= 10).sum().item()} | {torch.isin(am, torch.tensor(sorted(fc))).float().mean():.3f} | {mp.mean():.3f} / {mp.median():.3f} | {(mp > 0.9).float().mean():.3f} | {ent.mean():.3f} | {acc} |', flush=True)


row('images a oublier exactes', lg0, yhat)
sys.argv = ['x', '--name', 'x']
# melanges "moitie au plus" (meme fonction que rd.py)
src = torch.arange(len(fgd), device=dev); fgd = fgd.to(dev)
exec(open(os.path.dirname(os.path.abspath(__file__)) + '/mixfn.py').read())
g = gen(5); mm = torch.cat([to_u8(mix_half(src, 300, g)).cpu() for _ in range(20)]); lm = logits_u8(ref, mm)
n = 100
row('melanges moitie au plus (c) : tous', lm); row('  dont mixup 2-3 images', lm.view(20, 300, 100)[:, :n].reshape(-1, 100)); row('  dont coupe moitie-moitie', lm.view(20, 300, 100)[:, n:2 * n].reshape(-1, 100)); row('  dont mosaique 2x2', lm.view(20, 300, 100)[:, 2 * n:].reshape(-1, 100))
row('melanges de t3-methodes (cache)', torch.from_numpy(np.load(f'{CACHE}/mixes_orig_logits.npy')))
for k in ['blur_strong', 'color_strong']: row(f'copies degradees {k}', torch.from_numpy(np.load(f'{CACHE}/held_{k}_orig_logits.npy')), yhat)
for f in sorted(glob.glob(ME + '/gen/*.npz')):
    z = np.load(f); row('genere (a) ' + os.path.basename(f)[:-4], logits_u8(ref, torch.from_numpy(z['images'])), torch.from_numpy(z['y']))
c = np.load(ME + '/res/coco5k.npy', mmap_mode='r'); row('COCO val2017 (b)', logits_u8(ref, c))
