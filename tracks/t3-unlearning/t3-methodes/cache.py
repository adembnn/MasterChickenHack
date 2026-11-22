"""Cache commun t3-methodes (H200) : logits d'orig, decoupe a/b/c des images a oublier, copies de controle, melanges, CIFAR (regle de mesure), prep S1."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from PIL import Image
orig = load_orig(); fg = load_forget(); N = len(fg)


def save(name, u8):
    u8 = u8.cpu().numpy() if torch.is_tensor(u8) else u8
    np.save(f'{CACHE}/{name}.npy', u8); lg = logits_u8(orig, u8); np.save(f'{CACHE}/{name}_orig_logits.npy', lg.numpy())
    print(name, tuple(lg.shape), 'maxprob', F.softmax(lg, 1).max(1)[0].mean().item(), flush=True); return lg


assert torch.equal(norm(to01(fg[:8])), tt.preprocess(fg[:8]).to(dev)), 'normalisation != gabarit'
xx = norm(to01(fg[:16]))
with torch.no_grad(): a = orig(xx); b = top(orig, trunk(orig, xx, 4), 4)
print('ecart forward decoupe vs model(x) =', (a - b).abs().max().item(), flush=True)
lg0 = logits_u8(orig, fg); np.save(f'{CACHE}/exact_orig_logits.npy', lg0.numpy()); yhat = lg0.argmax(1)
rng = np.random.RandomState(20261121); part = np.zeros(N, dtype=np.int64)
for c in yhat.unique().tolist():
    idx = (yhat == c).nonzero().flatten().numpy(); rng.shuffle(idx); n = len(idx); na = int(round(.4 * n)); nc = int(round(.2 * n))
    part[idx[:na]] = 0; part[idx[na:n - nc]] = 1; part[idx[n - nc:]] = 2
torch.save(torch.from_numpy(part), f'{CACHE}/split.pt'); print('split a/b/c', np.bincount(part), flush=True)
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
for k, v in outs.items():
    lg = save('held_' + k, torch.cat(v)); ce = F.cross_entropy(lg, yhat, reduction='none')
    print(f'   orig sur {k}: accord {(lg.argmax(1) == yhat).float().mean():.4f} CE moy {ce.mean():.4f} sd {ce.std():.4f}', flush=True)
g = gen(777); ms = []
for i in range(0, 3000, 300):
    idx = torch.randint(0, N, (300,), generator=torch.Generator().manual_seed(i + 5))
    ms.append(to_u8(make_mix(to01(fg[idx]), g)).cpu())
save('mixes', torch.cat(ms))
try:
    assert os.environ.get("T3CIFAR"), "CIFAR saute (regle de mesure hors domaine, facultative)"
    import torchvision
    ds = torchvision.datasets.CIFAR100(ADEM + '/cifar', train=False, download=True)
    Xc = torch.from_numpy(ds.data[:3000]); Xc = F.interpolate(Xc.permute(0, 3, 1, 2).float(), size=224, mode='bicubic').clamp(0, 255).round().byte().permute(0, 2, 3, 1)
    save('ood_cifar100', Xc)
except Exception as e: print('CIFAR indisponible', e)
print('cache fini')
