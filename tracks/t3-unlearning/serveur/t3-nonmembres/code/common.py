"""Simulateur T3 : briques communes. Donnees SD-1.4 (100 classes), monde = decoupage train / oubli / controle / validation + copies modifiees."""
import os, sys, math, json, time, numpy as np, torch, torch.nn.functional as F, timm
ROOT = os.path.dirname(os.path.abspath(__file__)); DATA = ROOT + '/data'; dev = 'cuda'; NC = 100
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True


def warm():
    """Carte partagee et souvent pleine : on initialise cuBLAS/cuDNN et on reserve RESERVE_GB Go des le depart, pour ne pas echouer en cours de route."""
    x = torch.randn(64, 64, device=dev); (x @ x).sum().item(); torch.linalg.inv(x + 64 * torch.eye(64, device=dev)); F.conv2d(x.view(1, 1, 64, 64), x[:3, :3].reshape(1, 1, 3, 3)).sum().item()
    gb = float(os.environ.get('RESERVE_GB', 0))
    while gb > 0:
        try: t = torch.empty(int(gb * 2 ** 30), dtype=torch.uint8, device=dev); del t; break
        except torch.OutOfMemoryError: print('reservation impossible pour l instant, nouvel essai dans 30 s', flush=True); time.sleep(30)


def build(drop_path=0.0, init_values=1e-4):
    return timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=NC, init_values=init_values, drop_path_rate=drop_path)


def build_pre(drop_path=0.0):
    """Meme architecture, poids pre-entraines augreg_in21k ; LayerScale mis a 1 (equivalent a l'absence de LayerScale)."""
    m = build(drop_path, init_values=1.0); p = timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=True, num_classes=0).state_dict()
    miss = m.load_state_dict(p, strict=False); assert all(('gamma' in k or 'head' in k) for k in miss.missing_keys), miss
    return m


def load_sd(path):
    m = build(); m.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True); return m.to(dev).eval()


def to01(u8): return u8.to(dev).permute(0, 3, 1, 2).float() / 255.0
def norm(x01): return (x01 - 0.5) / 0.5
def to_u8(x01): return (x01.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous()
def gen(seed): g = torch.Generator(device=dev); g.manual_seed(seed); return g
def U(B, lo, hi, g): return torch.empty(B, device=dev).uniform_(lo, hi, generator=g)
def ac(): return torch.autocast(device_type='cuda', dtype=torch.float16)


def affine(x, g, scale=(1., 1.), ratio=(3 / 4, 4 / 3), flip=False, rot=0.):
    B = x.shape[0]
    area = U(B, scale[0], scale[1], g); r = torch.exp(U(B, math.log(ratio[0]), math.log(ratio[1]), g))
    sx = torch.sqrt(area * r).clamp(max=1); sy = torch.sqrt(area / r).clamp(max=1)
    tx = (1 - sx) * (U(B, 0, 1, g) * 2 - 1); ty = (1 - sy) * (U(B, 0, 1, g) * 2 - 1)
    f = torch.where(U(B, 0, 1, g) < 0.5, -1., 1.) if flip else torch.ones(B, device=dev)
    a = (U(B, 0, 1, g) * 2 - 1) * rot * math.pi / 180; c, s = torch.cos(a), torch.sin(a)
    th = torch.zeros(B, 2, 3, device=dev)
    th[:, 0, 0] = c * sx * f; th[:, 0, 1] = -s * sy; th[:, 1, 0] = s * sx * f; th[:, 1, 1] = c * sy; th[:, 0, 2] = tx; th[:, 1, 2] = ty
    grid = F.affine_grid(th, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode='bilinear', padding_mode='reflection', align_corners=False), area


_YIQ = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]])


def color(x, g, b=0., c=0., s=0., h=0.):
    B = x.shape[0]

    def Uf(m): return (1 + (U(B, 0, 1, g) * 2 - 1) * m).view(B, 1, 1, 1)
    if b: x = x * Uf(b)
    if c: m = x.mean((1, 2, 3), keepdim=True); x = (x - m) * Uf(c) + m
    if s: gr = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]); x = (x - gr) * Uf(s) + gr
    if h:
        T = _YIQ.to(dev); Ti = torch.linalg.inv(T); ang = (U(B, 0, 1, g) * 2 - 1) * h * 2 * math.pi
        R = torch.zeros(B, 3, 3, device=dev); R[:, 0, 0] = 1
        R[:, 1, 1] = torch.cos(ang); R[:, 1, 2] = -torch.sin(ang); R[:, 2, 1] = torch.sin(ang); R[:, 2, 2] = torch.cos(ang)
        x = torch.einsum('bij,bjhw->bihw', Ti @ R @ T, x)
    return x.clamp(0, 1)


def blur(x, sig):
    B, C, H, W = x.shape; r = 7; t = torch.arange(-r, r + 1, device=dev).float()
    k = torch.exp(-t[None] ** 2 / (2 * sig.clamp_min(0.05)[:, None] ** 2)); k = (k / k.sum(1, keepdim=True)).repeat_interleave(C, 0)
    y = F.pad(x.reshape(1, B * C, H, W), (r, r, 0, 0), mode='reflect'); y = F.conv2d(y, k.view(B * C, 1, 1, 2 * r + 1), groups=B * C)
    y = F.pad(y, (0, 0, r, r), mode='reflect'); y = F.conv2d(y, k.view(B * C, 1, 2 * r + 1, 1), groups=B * C)
    return y.view(B, C, H, W)


def noise(x, g, s): return (x + torch.randn(x.shape, device=dev, generator=g) * s).clamp(0, 1)


ALT_NAMES = ['recadrage+flip', 'couleur douce', 'flou doux', 'bruit+recadrage leger']


def altered(x, t, g):
    """Copies 'legerement modifiees' du simulateur (le type reel est inconnu) : 4 familles."""
    B = x.shape[0]
    if t == 0: return affine(x, g, scale=(0.6, 1.), flip=True)[0]
    if t == 1: return color(x, g, 0.25, 0.25, 0.25)
    if t == 2: return blur(x, U(B, 0.6, 1.2, g))
    if t == 3: return noise(affine(x, g, scale=(0.85, 1.))[0], g, 0.04)


ROB = {'propre': lambda x, g: x, 'recadrage 0.5-1': lambda x, g: affine(x, g, scale=(0.5, 1.))[0], 'flip': lambda x, g: x.flip(3),
       'bruit 0.05': lambda x, g: noise(x, g, 0.05), 'bruit 0.10': lambda x, g: noise(x, g, 0.10),
       'flou 1.5-2.5': lambda x, g: blur(x, U(x.shape[0], 1.5, 2.5, g)), 'couleur .4.4.4.1': lambda x, g: color(x, g, 0.4, 0.4, 0.4, 0.1)}


# ---------- monde ----------
def real_counts():
    lg = np.load(ROOT + '/forget_logits.npy'); return np.bincount(lg.argmax(1), minlength=NC)


def make_world(name, n_tr=400, seed=0):
    """Par classe : n_tr images d'entrainement (dont l'oubli), puis 60 de controle et 60 de validation (jamais vues).
    Oubli : memes classes et memes effectifs que le vrai jeu (45 classes, 46 a 130 images, 3500 au total)."""
    Y = np.load(DATA + '/y.npy'); per = len(Y) // NC; assert n_tr + 120 <= per
    cnt = real_counts(); rng = np.random.default_rng(seed); tr, fg, ct, va = [], [], [], []
    for c in range(NC):
        base = c * per; p = rng.permutation(n_tr)
        fg += list(base + p[:cnt[c]]); tr += list(base + np.arange(n_tr)); ct += list(base + per - 120 + np.arange(60)); va += list(base + per - 60 + np.arange(60))
    w = dict(train=np.array(sorted(tr)), forget=np.array(sorted(fg)), ctrl=np.array(ct), val=np.array(va)); w['retain'] = np.setdiff1d(w['train'], w['forget'])
    global dev; old = dev; dev = 'cpu'            # copies generees sur CPU : identiques d'une machine a l'autre
    X = np.load(DATA + '/x.npy', mmap_mode='r'); g = torch.Generator().manual_seed(1234 + seed); xf = torch.from_numpy(X[w['forget']]); alts = []
    for t in range(4):
        out = [to_u8(altered(to01(xf[i:i + 250]), t, g)).cpu() for i in range(0, len(xf), 250)]; alts.append(torch.cat(out).numpy())
    dev = old; np.savez(f'{DATA}/world_{name}.npz', **w); np.save(f'{DATA}/alt_{name}.npy', np.stack(alts))
    import hashlib; print('md5 alt', hashlib.md5(np.stack(alts)[:, :50].tobytes()).hexdigest(), 'md5 x', hashlib.md5(np.ascontiguousarray(X[:200]).tobytes()).hexdigest())
    print('monde', name, {k: len(v) for k, v in w.items()}, 'oubli classes', int((cnt > 0).sum()), 'max', cnt.max())


class World:
    def __init__(self, name):
        self.name = name; self.X = np.load(DATA + '/x.npy', mmap_mode='r'); self.Y = np.load(DATA + '/y.npy')
        self.idx = dict(np.load(f'{DATA}/world_{name}.npz')); self.alt = np.load(f'{DATA}/alt_{name}.npy', mmap_mode='r')

    def x(self, k): return torch.from_numpy(self.X[self.idx[k]])
    def y(self, k): return torch.from_numpy(self.Y[self.idx[k]])


@torch.no_grad()
def logits_u8(model, u8, bs=250):
    model.eval(); out = []
    for i in range(0, len(u8), bs):
        xb = u8[i:i + bs] if torch.is_tensor(u8) else torch.from_numpy(np.ascontiguousarray(u8[i:i + bs]))
        with ac(): out.append(model(norm(to01(xb))).float().cpu())
    return torch.cat(out)


def eval_all(model, W, path=None):
    """Logits aux conditions du serveur (eval, fp16) sur : oubli exact (fx), 4 familles de copies (fa4), controle, validation."""
    r = {'fx': logits_u8(model, W.x('forget')).numpy(), 'fa4': np.stack([logits_u8(model, W.alt[t]).numpy() for t in range(4)]),
         'ctrl': logits_u8(model, W.x('ctrl')).numpy(), 'val': logits_u8(model, W.x('val')).numpy()}
    if path: np.savez(path, **r)
    return r


def labels(W): return {'fx': W.y('forget').numpy(), 'ctrl': W.y('ctrl').numpy(), 'val': W.y('val').numpy()}


if __name__ == '__main__':
    if sys.argv[1] == 'make': make_world(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 0)
