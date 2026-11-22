"""T3 unlearning : briques communes (A). A lancer sur JURECA avec .venv python (torch+timm).
Charge orig (strict), images, pretraitement du gabarit, vues GPU faibles/fortes, melanges d'images a oublier,
evaluation aux conditions du serveur (eval, autocast fp16)."""
import os, sys, math, json, time, io
import numpy as np, torch, torch.nn.functional as F
BASE = os.environ.get('T3BASE', os.path.expanduser('~/eac/t3'))
ADEM = os.environ.get('T3ME', BASE + '/t3-methodes'); CACHE = ADEM + '/cache'; CAND = ADEM + '/candidates'; DIAG = ADEM + '/diag'
os.makedirs(CACHE, exist_ok=True); os.makedirs(CAND, exist_ok=True); os.makedirs(DIAG, exist_ok=True)
os.chdir(BASE); sys.path.insert(0, BASE)
import task_template as tt
dev = 'cuda'
NC = 100


def load_orig(): return tt.load_model()                       # strict=True, eval, cuda
def load_forget(): return tt.load_forget_images()             # uint8 [N,224,224,3] cpu
def to01(u8): return u8.to(dev).permute(0, 3, 1, 2).float() / 255.0
def norm(x01): return (x01 - 0.5) / 0.5                      # identique a tt.preprocess
def to_u8(x01): return (x01.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous()
def gen(seed): g = torch.Generator(device=dev); g.manual_seed(seed); return g
def U(B, lo, hi, g): return torch.empty(B, device=dev).uniform_(lo, hi, generator=g)


# ---------- augmentations GPU (x01: [B,3,H,W] dans [0,1]) ----------
def affine(x, g, scale=(1., 1.), ratio=(3 / 4, 4 / 3), flip=False, rot=0., zoom=1.):
    B = x.shape[0]
    area = U(B, scale[0], scale[1], g); r = torch.exp(U(B, math.log(ratio[0]), math.log(ratio[1]), g))
    sx = torch.sqrt(area * r).clamp(max=1) / zoom; sy = torch.sqrt(area / r).clamp(max=1) / zoom
    tx = (1 - sx) * (U(B, 0, 1, g) * 2 - 1); ty = (1 - sy) * (U(B, 0, 1, g) * 2 - 1)
    f = torch.where(U(B, 0, 1, g) < 0.5, -1., 1.) if flip else torch.ones(B, device=dev)
    a = (U(B, 0, 1, g) * 2 - 1) * rot * math.pi / 180
    c, s = torch.cos(a), torch.sin(a)
    th = torch.zeros(B, 2, 3, device=dev)
    th[:, 0, 0] = c * sx * f; th[:, 0, 1] = -s * sy; th[:, 1, 0] = s * sx * f; th[:, 1, 1] = c * sy
    th[:, 0, 2] = tx; th[:, 1, 2] = ty
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
        M = Ti @ R @ T; x = torch.einsum('bij,bjhw->bihw', M, x)
    return x.clamp(0, 1)


def blur(x, sig):
    B, C, H, W = x.shape; r = 7; t = torch.arange(-r, r + 1, device=dev).float()
    k = torch.exp(-t[None] ** 2 / (2 * sig.clamp_min(0.05)[:, None] ** 2)); k = (k / k.sum(1, keepdim=True)).repeat_interleave(C, 0)
    y = F.pad(x.reshape(1, B * C, H, W), (r, r, 0, 0), mode='reflect')
    y = F.conv2d(y, k.view(B * C, 1, 1, 2 * r + 1), groups=B * C)
    y = F.pad(y, (0, 0, r, r), mode='reflect')
    y = F.conv2d(y, k.view(B * C, 1, 2 * r + 1, 1), groups=B * C)
    return y.view(B, C, H, W)


def train_view(x, g, strong):
    """Vue d'entrainement. Retourne (vue, severite in [0,1]). Types utilises : recadrage, flip, couleur<=0.2, flou sigma<=1, bruit<=0.05.
    Jamais utilises a l'entrainement (reserves au controle) : JPEG, rotation, couleur forte+teinte, flou sigma>=1.5."""
    B = x.shape[0]
    lo = 0.35 if strong else 0.6
    v, area = affine(x, g, scale=(lo, 1.), flip=True)
    crop_sev = (1 - area) / 0.65
    cm = 0.2 if strong else 0.1
    v = color(v, g, cm, cm, cm)
    col_sev = torch.full((B,), cm / 0.2, device=dev)
    blur_sev = torch.zeros(B, device=dev); noise_sev = torch.zeros(B, device=dev)
    if strong:
        sg = U(B, 0.1, 1.0, g) * (U(B, 0, 1, g) < 0.5).float()
        v = blur(v, torch.where(sg > 0, sg, torch.full_like(sg, 0.05)))
        blur_sev = sg / 1.0
        ns = U(B, 0, 0.05, g) * (U(B, 0, 1, g) < 0.5).float()
        v = (v + torch.randn(v.shape, device=dev, generator=g) * ns.view(B, 1, 1, 1)).clamp(0, 1); noise_sev = ns / 0.05
    sev = (0.4 * crop_sev + 0.2 * blur_sev + 0.2 * noise_sev + 0.2 * col_sev).clamp(0, 1)
    return v, sev


def make_mix(x, g):
    """Melanges construits UNIQUEMENT d'images a oublier : mixup / cutmix / melange de patchs (tiers chacun)."""
    B, C, H, W = x.shape; perm = torch.randperm(B, device=dev, generator=g); out = x.clone(); n = B // 3
    lam = U(n, 0.3, 0.7, g).view(n, 1, 1, 1); out[:n] = lam * x[:n] + (1 - lam) * x[perm[:n]]
    sl = slice(n, 2 * n); area = U(n, 0.2, 0.6, g); bw = (torch.sqrt(area) * W).long().clamp(8, W - 8)
    x0 = (U(n, 0, 1, g) * (W - bw)).long(); y0 = (U(n, 0, 1, g) * (H - bw)).long()
    ii = torch.arange(H, device=dev).view(1, H, 1); jj = torch.arange(W, device=dev).view(1, 1, W)
    mask = ((ii >= y0.view(n, 1, 1)) & (ii < (y0 + bw).view(n, 1, 1)) & (jj >= x0.view(n, 1, 1)) & (jj < (x0 + bw).view(n, 1, 1))).unsqueeze(1)
    out[sl] = torch.where(mask, x[perm[sl]], x[sl])
    r = B - 2 * n; xs = x[2 * n:]                                          # melange de patchs : 4x4 blocs de 56 px
    p = xs.reshape(r, C, 4, 56, 4, 56).permute(0, 1, 2, 4, 3, 5).reshape(r, C, 16, 56, 56)
    idx = torch.argsort(torch.rand(r, 16, device=dev, generator=g), 1)
    p = torch.gather(p, 2, idx.view(r, 1, 16, 1, 1).expand(-1, C, -1, 56, 56))
    out[2 * n:] = p.reshape(r, C, 4, 4, 56, 56).permute(0, 1, 2, 4, 3, 5).reshape(r, C, H, W)
    return out


# ---------- modele ----------
def prenorm_cls(m, x):
    x = m.patch_embed(x); x = m._pos_embed(x); x = m.patch_drop(x); x = m.norm_pre(x); x = m.blocks(x); return x[:, 0]


def head_from_cls(m, c): return m.head(m.norm(c))


def ac(): return torch.autocast(device_type='cuda', dtype=torch.float16)


@torch.no_grad()
def logits_u8(model, u8, bs=128):
    """logits [N,100] float32 cpu, comme le serveur : eval + autocast fp16, entree uint8 NHWC (tensor ou memmap)."""
    model.eval(); out = []
    for i in range(0, len(u8), bs):
        xb = u8[i:i + bs] if torch.is_tensor(u8) else torch.from_numpy(np.ascontiguousarray(u8[i:i + bs]))
        with ac(): out.append(model(norm(to01(xb))).float().cpu())
    return torch.cat(out)


def kl(p, q_log):  # KL(p || q) par ligne ; p probas, q_log log-probas
    return (p * (p.clamp_min(1e-12).log() - q_log)).sum(1)


def softT(z, T): return F.softmax(z / T, 1)


def build_sd(model):
    return {k: v.detach().cpu().float().contiguous() for k, v in model.state_dict().items()}


# ---------- ajouts t3-methodes ----------
DELTA = ADEM + '/deltas'; os.makedirs(DELTA, exist_ok=True)


def trunk(m, x, k):
    """Partie gelee : tout jusqu'au bloc 12-k exclu."""
    x = m.patch_embed(x); x = m._pos_embed(x); x = m.patch_drop(x); x = m.norm_pre(x)
    for b in m.blocks[:12 - k]: x = b(x)
    return x


def top(m, h, k):
    for b in m.blocks[12 - k:]: h = b(h)
    return m.forward_head(m.norm(h))


def split_sets():
    part = torch.load(f'{CACHE}/split.pt'); N = len(part); ar = torch.arange(N)
    return {'a': ar[part == 0], 'b': ar[part == 1], 'c': ar[part == 2], 'ab': ar[part < 2], 'all': ar, 'none': ar[:0]}


def trainable_names(m, k):
    out = []
    for n, p in m.named_parameters():
        blk = n.startswith('blocks.') and int(n.split('.')[1]) >= 12 - k
        if (blk or n.startswith('norm.') or n.startswith('head.')) and 'gamma' not in n: out.append(n)
    return out
