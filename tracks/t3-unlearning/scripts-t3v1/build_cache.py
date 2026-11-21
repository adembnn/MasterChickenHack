"""Construit une fois : copies de controle fixes (types NON utilises a l'entrainement), melanges, jeux OOD (mesure seulement), logits d'orig."""
import sys; sys.path.insert(0, '/p/scratch/training2666/destors1/MasterChickenHack/unlearning/adem/t3v1')
from unlearn import *
from PIL import Image
sys.path.insert(0, ADEM + '/pylib')
orig = load_orig(); fg = load_forget(); N = len(fg)


def save(name, u8):
    np.save(f'{CACHE}/{name}.npy', u8.cpu().numpy() if torch.is_tensor(u8) else u8)
    lg = logits_u8(orig, np.load(f'{CACHE}/{name}.npy', mmap_mode='r')); np.save(f'{CACHE}/{name}_orig_logits.npy', lg.numpy()); print(name, tuple(lg.shape), flush=True)


assert torch.equal(norm(to01(fg[:8])), tt.preprocess(fg[:8])), 'normalisation != gabarit'
xx = norm(to01(fg[:16]))
with torch.no_grad(): a = orig(xx); b = head_from_cls(orig, prenorm_cls(orig, xx))
print('normalisation identique au gabarit : OK ; ecart forward manuel vs model(x) =', (a - b).abs().max().item(), flush=True)
lg0 = logits_u8(orig, fg); np.save(f'{CACHE}/exact_orig_logits.npy', lg0.numpy()); print('exact', flush=True)
g = gen(1234); outs = {k: [] for k in ['jpeg50', 'rot15', 'color_strong', 'blur_strong']}
for i in range(0, N, 250):
    x = to01(fg[i:i + 250]); B = x.shape[0]
    v, _ = affine(x, g, rot=15., zoom=1.1); outs['rot15'].append(to_u8(v).cpu())
    outs['color_strong'].append(to_u8(color(x, g, 0.4, 0.4, 0.4, 0.1)).cpu())
    outs['blur_strong'].append(to_u8(blur(x, U(B, 1.5, 2.5, g))).cpu())
    jp = []
    for im in fg[i:i + 250].numpy():
        bb = io.BytesIO(); Image.fromarray(im).save(bb, 'JPEG', quality=50); jp.append(np.array(Image.open(bb).convert('RGB')))
    outs['jpeg50'].append(torch.from_numpy(np.stack(jp)))
for k, v in outs.items(): save('held_' + k, torch.cat(v))
g = gen(777); ms = []
for i in range(0, 3000, 300):
    idx = torch.randint(0, N, (300,), generator=torch.Generator().manual_seed(i + 5))
    ms.append(to_u8(make_mix(to01(fg[idx]), g)).cpu())
save('mixes', torch.cat(ms))
import pyarrow.parquet as pq
tb = pq.read_table(ADEM + '/in100_val.parquet'); rows = tb.column('image').to_pylist()[:5000]
X = np.stack([np.array(Image.open(io.BytesIO(r['bytes'])).convert('RGB').resize((224, 224), Image.BICUBIC)) for r in rows]); save('ood_in100', X)
import torchvision
ds = torchvision.datasets.CIFAR100(ADEM + '/cifar', train=False, download=False)
Xc = torch.from_numpy(ds.data[:5000]); Xc = F.interpolate(Xc.permute(0, 3, 1, 2).float(), size=224, mode='bicubic').clamp(0, 255).round().byte().permute(0, 2, 3, 1)
save('ood_cifar100', Xc)
print('cache fini')
