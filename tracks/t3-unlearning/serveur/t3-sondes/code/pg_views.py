"""Pseudo-gold par vues 'de-memorisantes' : sorties d'orig sur des versions degradees de chaque image a oublier (5 familles x 5 forces x 4 tirages).
Idee : un gold n'a jamais vu l'image ; orig, sur une version assez degradee, ne peut plus s'appuyer sur sa memoire et repond avec ses seuls traits generaux.
Sortie : cache/pg_logits.npy float16 [force, famille, tirage, N, 100]. Aucune image externe."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
orig = load_orig(); fgd = load_forget().to(dev); N = len(fgd)
LEV = [0.2, 0.4, 0.6, 0.8, 1.0]; FAM = ['blur', 'crop', 'mask', 'noise', 'color']; V = 4
lg0 = torch.from_numpy(np.load(f'{CACHE}/exact_orig_logits.npy')); yhat = lg0.argmax(1)


def view(x, fam, s, g):
    B = x.shape[0]
    if fam == 'blur': return blur(x, torch.full((B,), 0.6 + 2.4 * s, device=dev) * U(B, 0.85, 1.15, g))
    if fam == 'crop': a = 0.6 - 0.5 * s; return affine(x, g, scale=(a, a), flip=True)[0]
    if fam == 'mask':
        m = (torch.rand(B, 1, 14, 14, device=dev, generator=g) < 0.15 + 0.6 * s).float(); m = F.interpolate(m, scale_factor=16, mode='nearest'); return x * (1 - m) + 0.5 * m
    if fam == 'noise': return (x + torch.randn(x.shape, device=dev, generator=g) * (0.04 + 0.21 * s)).clamp(0, 1)
    if fam == 'color': c = 0.15 + 0.35 * s; return color(x, g, c, c, c, 0.12 * s)


out = np.zeros((len(LEV), len(FAM), V, N, 100), dtype=np.float16); g = gen(4242); t0 = time.time()
for li, s in enumerate(LEV):
    for fi, fam in enumerate(FAM):
        for v in range(V):
            for i in range(0, N, 500):
                with torch.no_grad(), ac(): z = orig(norm(view(to01(fgd[i:i + 500]), fam, s, g))).float().cpu()
                out[li, fi, v, i:i + 500] = z.numpy().astype(np.float16)
        z = torch.from_numpy(out[li, fi].astype(np.float32)).reshape(-1, 100); y = yhat.repeat(V)
        ce = F.cross_entropy(z, y, reduction='none'); wr = z.argmax(1) != y; pw = z.argmax(1)[wr]
        sink = (torch.bincount(pw, minlength=100).max().item() / max(1, len(pw)))
        print(f'force {s:.1f} {fam:6s} acc {1 - wr.float().mean():.3f} CE moy {ce.mean():.3f} sd {ce.std():.3f} CE des erreurs {ce[wr].mean() if wr.any() else 0:.2f} maxprob T2 {F.softmax(z / 2, 1).max(1)[0].mean():.3f} part de la classe-puits {sink:.2f}  ({time.time() - t0:.0f}s)', flush=True)
np.save(f'{CACHE}/pg_logits.npy', out); print('pg fini')
