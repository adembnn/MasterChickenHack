"""t3-poids : vecteur de tache. Sur-entraine orig sur le forget set (pseudo-etiquettes = argmax d'orig) et ecrit delta = plus - orig.
modes : exact (images exactes), aug (vues fortement augmentees), mix (melanges d'images, etiquette = argmax d'orig sur le melange).
pertes : ce (entropie croisee), cet (entropie croisee a temperature T), margin (-(z_y - max autre)), non saturee quand orig est deja sur de lui."""
import sys, os, argparse
sys.path.insert(0, os.path.expanduser('~/eac/t3/t3-methodes/code')); os.environ.setdefault('T3ME', os.path.expanduser('~/eac/t3/t3-poids'))
from common import *
ap = argparse.ArgumentParser()
ap.add_argument('--name', required=True); ap.add_argument('--fset', default='ab'); ap.add_argument('--mode', default='exact'); ap.add_argument('--loss', default='ce')
ap.add_argument('--T', type=float, default=4.0); ap.add_argument('--opt', default='sgd'); ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--epochs', type=int, default=3)
ap.add_argument('--bs', type=int, default=64); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--sign', type=float, default=1.0)
a = ap.parse_args()
torch.manual_seed(a.seed); g = gen(a.seed)
S = split_sets(); idx = S[a.fset]; fg = load_forget()[idx]
yhat = torch.from_numpy(np.load(f'{CACHE}/exact_orig_logits.npy')).argmax(1)[idx].to(dev)
orig_sd = torch.load(BASE + '/model.pt', map_location='cpu', weights_only=True)
m = tt.build_model().to(dev); m.load_state_dict(orig_sd, strict=True); m.train()
teacher = tt.build_model().to(dev).eval(); teacher.load_state_dict(orig_sd, strict=True)
P = [p for p in m.parameters()]
opt = torch.optim.SGD(P, lr=a.lr, momentum=0.9) if a.opt == 'sgd' else torch.optim.AdamW(P, lr=a.lr, weight_decay=0.0)
N = len(idx); t0 = time.time()


def lossf(z, y):
    z = z.float()
    if a.loss == 'ce': return F.cross_entropy(z, y)
    if a.loss == 'cet': return F.cross_entropy(z / a.T, y)
    zy = z.gather(1, y[:, None])[:, 0]; zo = z.scatter(1, y[:, None], -1e4).max(1)[0]; return -(zy - zo).mean()


for ep in range(a.epochs):
    perm = torch.randperm(N); tot = 0.; mar = 0.; nb = 0
    for i in range(0, N - a.bs + 1, a.bs):
        b = perm[i:i + a.bs]; x = to01(fg[b]); y = yhat[b]
        if a.mode == 'aug': x, _ = train_view(x, g, True)
        elif a.mode == 'mix':
            x = make_mix(x, g)
            with torch.no_grad(), ac(): y = teacher(norm(x)).float().argmax(1)
        with torch.autocast('cuda', dtype=torch.bfloat16): z = m(norm(x))
        l = a.sign * lossf(z, y); opt.zero_grad(set_to_none=True); l.backward(); opt.step()
        with torch.no_grad():
            zf = z.float(); zy = zf.gather(1, y[:, None])[:, 0]; zo = zf.scatter(1, y[:, None], -1e4).max(1)[0]; mar += (zy - zo).mean().item()
        tot += l.item(); nb += 1
    print(f'ep {ep + 1} loss {tot / nb:.5f} marge {mar / nb:.3f} {time.time() - t0:.0f}s', flush=True)
sd = m.state_dict(); delta = {k: (sd[k].detach().cpu().float() - orig_sd[k].float()) for k in orig_sd if orig_sd[k].is_floating_point()}
nrm = math.sqrt(sum((v ** 2).sum().item() for v in delta.values())); on = math.sqrt(sum((v.float() ** 2).sum().item() for v in orig_sd.values() if v.is_floating_point()))
print(f'norme delta {nrm:.4f}  norme orig {on:.2f}  ratio {nrm / on:.5f}', flush=True)
torch.save({'delta': delta, 'meta': vars(a)}, f'{DELTA}/{a.name}.pt'); print('ecrit', f'{DELTA}/{a.name}.pt')
