"""Orig simule ressemble-t-il au vrai ? Memes transformations que reports/t3-etape-zero-mesures.md (accord avec la prediction propre, max-prob, loss).
usage : robust.py WORLD NAME   |   robust.py REAL   (vrai model.pt + vraies images)"""
from common import *
if sys.argv[1] == 'REAL':
    m = load_sd('/home/adem/eac/t3/model.pt'); fg = torch.from_numpy(np.load('/home/adem/eac/t3/forget_images.npz')['images']); tag = 'VRAI orig'
else:
    W = World(sys.argv[1]); m = load_sd(f'{ROOT}/runs/{sys.argv[1]}/{sys.argv[2]}.pt'); fg = W.x('forget'); tag = f'simule {sys.argv[1]}/{sys.argv[2]}'
g = gen(7); N = len(fg); y0 = None; print(f'--- robustesse de {tag} sur les {N} images a oublier : accord / max-prob / loss vs prediction propre')
for name, fn in ROB.items():
    zs = []
    with torch.no_grad(), ac():
        for i in range(0, N, 250): zs.append(m(norm(fn(to01(fg[i:i + 250]), g))).float())
    z = torch.cat(zs)
    if y0 is None: y0 = z.argmax(1); print(f'   marge logit top1-top2 moyenne {(z.topk(2, 1).values[:, 0] - z.topk(2, 1).values[:, 1]).mean():.2f} ; max-prob a T=2 {F.softmax(z / 2, 1).max(1).values.mean():.4f}')
    print(f'{name:20s} {(z.argmax(1) == y0).float().mean():.4f}  {F.softmax(z, 1).max(1).values.mean():.4f}  {F.cross_entropy(z, y0).item():.4f}')
