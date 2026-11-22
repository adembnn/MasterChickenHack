"""Images d'inversion du modele d'origine (sans aucune donnee externe) : pour chaque classe, on optimise des pixels
pour que orig predise cette classe. Sert d'ancre sur les classes ABSENTES du lot a oublier (55 % du jeu de controle)."""
import sys, os, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.expanduser('~/eac/t3/t3-methodes/code'))
import common as C
torch.manual_seed(0); dev = 'cuda'
m = C.load_orig(); [p.requires_grad_(False) for p in m.parameters()]
lg = np.load(C.CACHE + '/exact_orig_logits.npy'); present = set(np.unique(lg.argmax(1)).tolist())
n_abs, n_pre, steps = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
cls = [c for c in range(100) for _ in range(n_pre if c in present else n_abs)]
cls = torch.tensor(cls); out = []
for i in range(0, len(cls), 96):
    y = cls[i:i + 96].to(dev); B = len(y)
    w = torch.nn.Parameter(F.interpolate(torch.randn(B, 3, 28, 28, device=dev) * 0.5, size=224, mode='bilinear') + 0.1 * torch.randn(B, 3, 224, 224, device=dev))
    opt = torch.optim.Adam([w], lr=0.05)
    for s in range(steps):
        x = torch.sigmoid(w); dx, dy = np.random.randint(-24, 25, 2)
        xa = torch.roll(x, (int(dx), int(dy)), (2, 3))
        if np.random.rand() < 0.5: xa = xa.flip(3)
        with C.ac(): z = m(C.norm(xa)).float()
        tv = (x[:, :, 1:] - x[:, :, :-1]).abs().mean() + (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
        loss = F.cross_entropy(z, y) + 0.3 * tv
        opt.zero_grad(); loss.backward(); opt.step()
    out.append(C.to_u8(torch.sigmoid(w).detach()).cpu()); print(i, float(loss), flush=True)
X = torch.cat(out); z0 = C.logits_u8(m, X); p = F.softmax(z0, 1)
ok = (z0.argmax(1) == cls).float().mean(); conf = p[torch.arange(len(cls)), cls].mean()
print(f'images {tuple(X.shape)} ; orig predit la classe visee : {ok:.3f} ; confiance moyenne {conf:.3f}')
isabs = torch.tensor([c not in present for c in cls.tolist()])
np.savez(C.CACHE + '/inv_orig.npz', images=X.numpy(), cls=cls.numpy(), absent=isabs.numpy(), orig_logits=z0.numpy())
# derive des candidats existants sur ces images, classes absentes / presentes
import glob
for name in sys.argv[4:]:
    sd = torch.load(C.CAND + f'/{name}.pt', map_location='cpu'); m2 = C.load_orig(); m2.load_state_dict(sd if 'head.weight' in sd else sd['state_dict']); m2.eval()
    z1 = C.logits_u8(m2, X)
    for T in (1.0, 2.0):
        kl = (F.softmax(z0 / T, 1) * (F.log_softmax(z0 / T, 1) - F.log_softmax(z1 / T, 1))).sum(1)
        acc = (z1.argmax(1) == z0.argmax(1)).float()
        print(f'{name} T={T}: KL(orig||sub) absentes {kl[isabs].mean():.3f} presentes {kl[~isabs].mean():.3f} | accord absentes {acc[isabs].mean():.3f} presentes {acc[~isabs].mean():.3f}')
