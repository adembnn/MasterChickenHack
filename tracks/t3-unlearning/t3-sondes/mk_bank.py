"""Banque de gabarits de gold : sorties d'orig sur des copies decalees des 3 500 images (hypothese de travail : un modele
reentraine sans une image se comporte sur elle comme orig sur une image qu'il n'a pas vue telle quelle).
6 niveaux de decalage x 8 tirages. Sortie : ~/eac/t3/t3-sondes/bank.npz  (z [6,8,3500,100] float16, kind [6,8,3500])"""
import os, sys, numpy as np, torch
os.environ.setdefault('T3ME', os.path.expanduser('~/eac/t3/t3-sondes'))
sys.path.insert(0, os.path.expanduser('~/eac/t3/t3-methodes/code'))
import common as C
dev = 'cuda'; u8 = C.load_forget(); m = C.load_orig().eval(); N = len(u8)
LEV = [0.25, 0.4, 0.55, 0.7, 0.85, 1.0]; ND = 8; Z = np.zeros((len(LEV), ND, N, 100), np.float16); KIND = np.zeros((len(LEV), ND, N), np.int8)
for li, s in enumerate(LEV):
    for d in range(ND):
        g = C.gen(1000 * li + d); out = []; kinds = []
        for i in range(0, N, 250):
            x = C.to01(u8[i:i + 250]); B = x.shape[0]; k = torch.randint(0, 5, (B,), device=dev, generator=g); v = x.clone()
            a, _ = C.affine(x, g, scale=(1 - 0.75 * s, 1 - 0.5 * s), flip=True, rot=20 * s)          # 0 recadrage + rotation
            c = C.color(x, g, 0.6 * s, 0.6 * s, 0.6 * s, 0.15 * s)                                    # 1 couleur
            b = C.blur(x, C.U(B, 1.0 * s, 3.0 * s, g))                                                # 2 flou
            n = (x + torch.randn(x.shape, device=dev, generator=g) * (0.25 * s)).clamp(0, 1)          # 3 bruit
            cb = C.blur(C.color(a, g, 0.4 * s, 0.4 * s, 0.4 * s, 0.1 * s), C.U(B, 0.5 * s, 1.5 * s, g))  # 4 tout a la fois, plus doux
            for j, t in enumerate((a, c, b, n, cb)): v = torch.where((k == j).view(B, 1, 1, 1), t, v)
            with torch.no_grad(), C.ac(): out.append(m(C.norm(v)).float().cpu())
            kinds.append(k.cpu())
        z = torch.cat(out).numpy(); Z[li, d] = z; KIND[li, d] = torch.cat(kinds).numpy()
        print(li, s, d, 'acc vs pseudo', float((z.argmax(1) == Z[0, 0].argmax(1)).mean()) if (li or d) else 1.0, flush=True)
zo = np.load(os.path.expanduser('~/eac/t3/t3-methodes/cache/exact_orig_logits.npy')); y = zo.argmax(1)
for li, s in enumerate(LEV):
    z = Z[li].astype(np.float64); zz = z - z.max(-1, keepdims=True); l = -(zz - np.log(np.exp(zz).sum(-1, keepdims=True)))[:, np.arange(N), y]
    print(f'niveau {s}: acc {np.mean(z.argmax(-1) == y):.3f}  CE moy {l.mean():.3f}  sd {l.std(1).mean():.3f}', flush=True)
np.savez_compressed(os.path.expanduser('~/eac/t3/t3-sondes/bank.npz'), z=Z, kind=KIND, lev=np.array(LEV), y=y)
