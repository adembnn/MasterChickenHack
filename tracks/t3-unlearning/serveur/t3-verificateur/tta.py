"""t3-verificateur, levier 3 : la memorisation se lit sur orig. Sortie moyenne d'orig (T=2) sur K alterations FORTES de chaque image a oublier.
Compare le rang (probabilite de la vraie classe) et la classe fausse a la source de RDq (moyenne a T=2 de 8 partitions hors pli a 4 blocs) et aux redistillations a 6 / 8 blocs.
Ecrit out/tta_p2.npy (moyenne des softmax(logits/2)), out/tta_logits_mean.npy. usage : python tta.py [K]"""
import os, sys, math, numpy as np, torch
import vlogits as V
K = int(sys.argv[1]) if len(sys.argv) > 1 else 16; ROOT = V.ROOT; OUT = V.OUT; OOF = V.ME + '/oof'
fg = np.load(ROOT + '/forget_images.npz')['images']; N = len(fg)
m = V.build(); m.load_state_dict(torch.load(ROOT + '/model.pt', map_location='cpu', weights_only=True)); m = m.to(V.dev).eval()
def jpeg(x, q):
    import io
    from PIL import Image
    out = []
    for im in (x.permute(0, 2, 3, 1).clamp(0, 1) * 255).round().to(torch.uint8).numpy():
        b = io.BytesIO(); Image.fromarray(im).save(b, 'JPEG', quality=q); out.append(np.array(Image.open(b).convert('RGB')))
    return torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2).float() / 255
FAM = [lambda x, g: V.t_affine(x, g, smin=0.3, flip=True), lambda x, g: V.t_affine(x, g, smin=0.5, flip=True, rot=20.0), lambda x, g: V.t_blur(x, g, 2.0, 3.5),
       lambda x, g: V.t_color(x, g, 0.5, hue=0.15), lambda x, g: jpeg(V.t_affine(x, g, smin=0.6, flip=True), 25),
       lambda x, g: (V.t_affine(x, g, smin=0.5, flip=True) + torch.randn(x.shape, generator=g) * 0.08).clamp(0, 1),
       lambda x, g: V.t_blur(V.t_color(x, g, 0.4), g, 1.5, 2.5), lambda x, g: V.t_affine(V.t_color(x, g, 0.4, hue=0.1), g, smin=0.4, flip=True)]
def lsm(z, t=1.0): z = np.asarray(z, np.float64) / t; z = z - z.max(-1, keepdims=True); return z - np.log(np.exp(z).sum(-1, keepdims=True))
g = torch.Generator().manual_seed(4242); P2 = np.zeros((N, 100)); LZ = np.zeros((N, 100)); PY = []
y = np.load(OUT + '/orig.npz')['exact'].argmax(1); ar = np.arange(N)
for k in range(K):
    fam = FAM[k % len(FAM)]; zs = []
    for i in range(0, N, 250):
        x = torch.from_numpy(fg[i:i + 250]).permute(0, 3, 1, 2).float() / 255.0; xa = (fam(x, g).clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous()
        zs.append(V.predict(m, xa, bs=125))
    z = np.concatenate(zs).astype(np.float64); P2 += np.exp(lsm(z, 2)); LZ += z; PY.append(np.exp(lsm(z, 2))[ar, y]); print(f'alteration {k:2d} famille {k % len(FAM)} : precision d orig {(z.argmax(1) == y).mean():.3f} p_y T2 {np.exp(lsm(z, 2))[ar, y].mean():.3f}', flush=True)
P2 /= K; LZ /= K; PY = np.array(PY); np.save(OUT + '/tta_p2.npy', P2.astype(np.float32)); np.save(OUT + '/tta_logits_mean.npy', LZ.astype(np.float32))
def spear(a, b): ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])
def src(k, mm, t=2.0): Ps = [np.exp(lsm(np.load(f'{OOF}/redistill_k{k}m{mm}_logits_s{i}.npy'), t)) for i in range(mm)]; return np.mean(Ps, 0)
S4, S6, S8 = src(4, 8), src(6, 4), src(8, 2)
py = P2[ar, y]; W = P2.copy(); W[ar, y] = 0; wc = W.argmax(1)
print(f'\nTTA : p_y T2 moyen {py.mean():.3f} ; ecart-type entre alterations par image (moyenne) {PY.std(0).mean():.3f} ; classe en tete != orig {(P2.argmax(1) != y).mean():.3f}')
k7 = int(0.7 * N)
for nm, S in [('RDq (4 blocs x8, T=2)', S4), ('6 blocs x4', S6), ('8 blocs x2', S8)]:
    sy = S[ar, y]; Sw = S.copy(); Sw[ar, y] = 0; swc = Sw.argmax(1); ta = set(np.argsort(py)[:k7].tolist()); sa = set(np.argsort(sy)[:k7].tolist()); both = np.array(sorted(ta & sa))
    print(f'{nm:24s} : correlation de rang des p_y {spear(py, sy):.3f} ; images communes dans les 70 % du haut {len(both)} / {k7} ; meme classe fausse (1re hors orig) sur toutes les images {(wc == swc).mean():.3f}, sur les communes {(wc[both] == swc[both]).mean():.3f}')
print(f'reference : RDq / 6 blocs : correlation {spear(S4[ar, y], S6[ar, y]):.3f}, meme classe fausse {(np.argmax(np.where(np.eye(100)[y] > 0, 0, S4), 1) == np.argmax(np.where(np.eye(100)[y] > 0, 0, S6), 1)).mean():.3f}')
err6 = np.mean([np.load(f'{OOF}/redistill_k6m4_logits_s{i}.npy').argmax(1) != y for i in range(4)] + [np.load(f'{OOF}/redistill_k8m2_logits_s{i}.npy').argmax(1) != y for i in range(2)], 0)
from numpy import corrcoef
print(f'part de modeles profonds (6 / 8 blocs) qui se trompent, par image : correlation de rang avec 1 - p_y TTA {spear(1 - py, err6):.3f} ; avec 1 - p_y RDq {spear(1 - S4[ar, y], err6):.3f}')
