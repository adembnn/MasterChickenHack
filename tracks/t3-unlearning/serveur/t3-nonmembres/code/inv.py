"""Inversion du modèle : optimise une image pour qu'orig prédise la classe k, avec des a priori de régularité (variation totale, norme, augmentations aléatoires).
Sert à deviner le contenu des classes (dont les 55 absentes du forget set). Sorties : img/inv.npz (4 images par classe), meas/inv_<a>_<b>.jpg
usage : inv.py [pas=600] [n_par_classe=4]"""
import nm, sys, numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageDraw
C = nm.C; steps = int(sys.argv[1]) if len(sys.argv) > 1 else 600; n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
m = nm.orig()
for p in m.parameters(): p.requires_grad_(False)
present = set(np.load(nm.MEAS + '/forget_logits.npy').argmax(1).tolist()); out = []; g = C.gen(0)
for a in range(0, 100, 25):
    y = torch.arange(a, a + 25, device='cuda').repeat_interleave(n); B = len(y)
    # paramétrisation multi-échelle : somme de cartes 28, 56, 112, 224 px (favorise les basses fréquences)
    zs = [(torch.randn(B, 3, s, s, device='cuda', generator=g) * 0.5).requires_grad_(True) for s in (28, 56, 112, 224)]
    opt = torch.optim.Adam(zs, lr=0.05)
    def img(): return torch.sigmoid(sum(F.interpolate(z, size=224, mode='bilinear', align_corners=False) for z in zs))
    for t in range(steps):
        x = img(); xa = C.affine(x, g, scale=(0.5, 1.0), flip=True, rot=10)[0]; xa = xa + torch.randn(xa.shape, device='cuda', generator=g) * 0.03
        with C.ac(): z = m(C.norm(xa)).float()
        tv = (x[:, :, 1:] - x[:, :, :-1]).abs().mean() + (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
        loss = F.cross_entropy(z, y) + 2.0 * tv + 0.05 * ((x - 0.5) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if t % 100 == 0 or t == steps - 1: print(a, t, round(loss.item(), 4), 'acc', (z.argmax(1) == y).float().mean().item(), flush=True)
    with torch.no_grad():
        x = img(); u8 = C.to_u8(x).cpu().numpy(); out.append(u8)
        with C.ac(): zc = m(C.norm(x)).float()
        pc = F.softmax(zc, 1).gather(1, y[:, None])[:, 0].cpu().numpy()
    W = Image.new('RGB', (n * 226 + 70, 25 * 226), 'white'); d = ImageDraw.Draw(W)
    for r in range(25):
        k = a + r; d.text((2, r * 226 + 100), f"c{k}\n{'P' if k in present else 'abs'}\np={pc[r * n:(r + 1) * n].mean():.2f}", fill='black')
        for j in range(n): W.paste(Image.fromarray(u8[r * n + j]), (70 + j * 226, r * 226))
    W.save(f'{nm.MEAS}/inv_{a}_{a + 24}.jpg', quality=88)
X = np.concatenate(out); np.savez(nm.IMG + '/inv.npz', images=X, src=-np.ones(len(X), int), y=np.arange(100).repeat(n)); print('fini', X.shape)
