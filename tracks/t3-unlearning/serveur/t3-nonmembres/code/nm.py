"""t3-nonmembres : briques communes (orig, prétraitement serveur, augmentations du simulateur)."""
import os, sys, json, math, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.environ.get('SIMDIR', os.path.expanduser('~/eac/t3/t3-simulateur')))
import common as C   # to01, norm, to_u8, affine, color, blur, noise, altered, ROB, gen, ac, load_sd
T3 = os.environ.get('T3DIR', os.path.expanduser('~/eac/t3')); ROOT = os.environ.get('NMROOT', T3 + '/t3-nonmembres'); IMG = ROOT + '/img'; MEAS = ROOT + '/meas'
def orig(): return C.load_sd(T3 + '/model.pt')
def forget(): return np.load(T3 + '/forget_images.npz')['images']
@torch.no_grad()
def logits(m, u8, fn=None, seed=7, bs=250):
    g = C.gen(seed); out = []
    for i in range(0, len(u8), bs):
        x = C.to01(torch.from_numpy(np.ascontiguousarray(u8[i:i + bs])))
        if fn is not None: x = fn(x, g)
        with C.ac(): out.append(m(C.norm(x)).float().cpu())
    return torch.cat(out)
def jpeg(q):
    import io
    from PIL import Image
    def f(x, g):
        u = C.to_u8(x).cpu().numpy(); o = []
        for a in u:
            b = io.BytesIO(); Image.fromarray(a).save(b, 'JPEG', quality=q); o.append(np.array(Image.open(b).convert('RGB')))
        return C.to01(torch.from_numpy(np.stack(o)))
    return f
AUG = dict(C.ROB)
AUG.update({'alt0 recadrage+flip': lambda x, g: C.altered(x, 0, g), 'alt1 couleur douce': lambda x, g: C.altered(x, 1, g),
            'alt2 flou doux': lambda x, g: C.altered(x, 2, g), 'alt3 bruit+recadrage': lambda x, g: C.altered(x, 3, g),
            'jpeg q50': jpeg(50), 'jpeg q20': jpeg(20)})
