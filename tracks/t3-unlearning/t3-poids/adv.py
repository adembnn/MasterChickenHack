"""t3-poids : classe vers laquelle orig bascule sous la plus petite perturbation adversariale (source de mauvaise classe par image).
Descente de la marge (logit de la pseudo-etiquette - plus grand autre logit) par pas signes de 0,25/255 par pixel (norme L-infini), jusqu'a 48 pas (12/255).
Pour chaque image : numero du premier pas ou l'argmax change (49 = jamais), classe atteinte, logits d'orig a ce moment.
Sorties dans targets/ : adv_class.npy [3500] int64, adv_steps.npy [3500] int64, adv_logits.npy [3500,100] float32. Aucune image externe."""
import sys, os
sys.path.insert(0, os.path.expanduser('~/eac/t3/t3-methodes/code')); os.environ.setdefault('T3ME', os.path.expanduser('~/eac/t3/t3-poids'))
from common import *
OUT = ADEM + '/targets'; os.makedirs(OUT, exist_ok=True); K = int(os.environ.get('ADV_K', '4')); STEP = 0.25 / 255 / K; NS = 48 * K      # K = finesse du pas (K=4 : 0,0625/255)
orig = load_orig(); fg = load_forget(); N = len(fg); yhat = torch.from_numpy(np.load(f'{CACHE}/exact_orig_logits.npy')).argmax(1).to(dev)
for p in orig.parameters(): p.requires_grad_(False)
first = torch.full((N,), NS + 1, dtype=torch.long); cls = torch.full((N,), -1, dtype=torch.long); ZF = torch.zeros(N, NC); t0 = time.time()
for i in range(0, N, 250):
    x = to01(fg[i:i + 250]); y = yhat[i:i + 250]; B = len(y); f = torch.full((B,), NS + 1, dtype=torch.long, device=dev); c = torch.full((B,), -1, dtype=torch.long, device=dev); zf = torch.zeros(B, NC, device=dev)
    for t in range(NS + 1):
        x.requires_grad_(True)
        with torch.autocast('cuda', dtype=torch.bfloat16): z = orig(norm(x)).float()
        pred = z.argmax(1); new = (pred != y) & (f > NS)
        if t > 0 and new.any(): f[new] = t; c[new] = pred[new]; zf[new] = z.detach()[new]
        if t == NS or bool((f <= NS).all()): break
        mar = z.gather(1, y[:, None])[:, 0] - z.scatter(1, y[:, None], -1e4).max(1)[0]; g, = torch.autograd.grad(mar.sum(), x)
        x = (x.detach() - STEP * g.sign()).clamp(0, 1)
    nf = f > NS; zl = z.detach()
    if nf.any(): c[nf] = zl.scatter(1, y[:, None], -1e4).argmax(1)[nf]; zf[nf] = zl[nf]
    first[i:i + B] = f.cpu(); cls[i:i + B] = c.cpu(); ZF[i:i + B] = zf.cpu(); print(f'{i + B}/{N} {time.time() - t0:.0f}s  bascules {(first[:i + B] <= NS).float().mean():.3f}  pas median {first[:i + B].float().median():.0f}', flush=True)
np.save(f'{OUT}/adv_class.npy', cls.numpy()); np.save(f'{OUT}/adv_steps.npy', first.numpy()); np.save(f'{OUT}/adv_logits.npy', ZF.numpy().astype(np.float32))
q = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]); print(f'pas elementaire 0,25/255/{K} ; part d images basculees', (first <= NS).float().mean().item(), '| quantiles 10/25/50/75/90 % de la perturbation L-infini de bascule, en 1/255 :', (torch.quantile(first.float(), q) * 0.25 / K).tolist())
print('part des images basculant en <= 1/255', (first <= 4 * K).float().mean().item(), ' <= 2/255', (first <= 8 * K).float().mean().item(), ' <= 4/255', (first <= 16 * K).float().mean().item(), '| niveaux distincts', len(first.unique())); print('FINADV')
