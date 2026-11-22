"""t3-verificateur : controle de fichier + sorties aux conditions du serveur (eval, autocast fp16, pretraitement du gabarit).
Implementation independante de celles des sessions fabricantes (aucun import de leur code).
usage : python vlogits.py nom=chemin.pt [...]   -> out/<nom>.npz (logits par jeu) + out/<nom>.check.json
Jeux : exact ; copies legeres a_* (7 familles) ; substituts de controle c_* (images ou orig n'est plus en regime 'membre')."""
import sys, os, io, json, math, hashlib
import numpy as np, torch, torch.nn.functional as F, timm
ROOT = os.environ.get('T3ROOT', 'E:/EAC-lab/t3'); ME = ROOT + '/t3-verificateur'; OUT = ME + '/out'; SETS = ME + '/sets'
os.makedirs(OUT, exist_ok=True); os.makedirs(SETS, exist_ok=True)
dev = 'cuda'


def build(): return timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=100, init_values=1e-4)   # = task_template.build_model
def prep(u8): return (u8.to(dev).permute(0, 3, 1, 2).float() / 255.0 - 0.5) / 0.5                                      # = task_template.preprocess


@torch.no_grad()
def predict(m, u8, bs=100, fp16=True):
    m.eval(); out = []
    for i in range(0, len(u8), bs):
        xb = u8[i:i + bs] if torch.is_tensor(u8) else torch.from_numpy(np.ascontiguousarray(u8[i:i + bs]))
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=fp16): out.append(m(prep(xb)).float().cpu())
    return torch.cat(out).numpy()


# ---------- jeux d'images (generes une fois, sur CPU, graine fixe) ----------
def _grid(x, th): return F.grid_sample(x, F.affine_grid(th, list(x.shape), align_corners=False), mode='bilinear', padding_mode='reflection', align_corners=False)


def t_affine(x, g, smin=1.0, tr=0.0, rot=0.0, flip=False):
    B = x.shape[0]; s = torch.empty(B).uniform_(smin, 1.0, generator=g).sqrt()
    a = (torch.rand(B, generator=g) * 2 - 1) * rot * math.pi / 180; f = torch.where(torch.rand(B, generator=g) < 0.5, -1.0, 1.0) if flip else torch.ones(B)
    th = torch.zeros(B, 2, 3); th[:, 0, 0] = torch.cos(a) * s * f; th[:, 0, 1] = -torch.sin(a) * s; th[:, 1, 0] = torch.sin(a) * s * f; th[:, 1, 1] = torch.cos(a) * s
    th[:, 0, 2] = (1 - s) * (torch.rand(B, generator=g) * 2 - 1) + tr * (torch.rand(B, generator=g) * 2 - 1); th[:, 1, 2] = (1 - s) * (torch.rand(B, generator=g) * 2 - 1) + tr * (torch.rand(B, generator=g) * 2 - 1)
    return _grid(x, th)


def t_color(x, g, m, hue=0.0):
    B = x.shape[0]; u = lambda: (1 + (torch.rand(B, generator=g) * 2 - 1) * m).view(B, 1, 1, 1)
    x = x * u(); mu = x.mean((1, 2, 3), keepdim=True); x = (x - mu) * u() + mu
    gr = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]); x = (x - gr) * u() + gr
    if hue:   # rotation des canaux autour de l'axe gris
        ang = (torch.rand(B, generator=g) * 2 - 1) * hue * 2 * math.pi; c, s = torch.cos(ang), torch.sin(ang); k = 1 / 3; q = math.sqrt(1 / 3)
        R = torch.stack([torch.stack([c + (1 - c) * k, (1 - c) * k - s * q, (1 - c) * k + s * q], 1), torch.stack([(1 - c) * k + s * q, c + (1 - c) * k, (1 - c) * k - s * q], 1),
                         torch.stack([(1 - c) * k - s * q, (1 - c) * k + s * q, c + (1 - c) * k], 1)], 1)
        x = torch.einsum('bij,bjhw->bihw', R, x)
    return x.clamp(0, 1)


def t_blur(x, g, lo, hi):
    B, C, H, W = x.shape; sig = torch.empty(B).uniform_(lo, hi, generator=g); r = 9; t = torch.arange(-r, r + 1).float()
    k = torch.exp(-t[None] ** 2 / (2 * sig[:, None] ** 2)); k = (k / k.sum(1, keepdim=True)).repeat_interleave(C, 0)
    y = F.conv2d(F.pad(x.reshape(1, B * C, H, W), (r, r, 0, 0), mode='reflect'), k.view(B * C, 1, 1, -1), groups=B * C)
    y = F.conv2d(F.pad(y, (0, 0, r, r), mode='reflect'), k.view(B * C, 1, -1, 1), groups=B * C); return y.view(B, C, H, W)


def make_sets(fg):
    from PIL import Image
    N = len(fg); specs = {
        'a_shift': lambda x, g: t_affine(x, g, tr=0.06), 'a_crop': lambda x, g: t_affine(x, g, smin=0.7, flip=True), 'a_rot': lambda x, g: t_affine(x, g, rot=10.0),
        'a_color': lambda x, g: t_color(x, g, 0.2), 'a_blur': lambda x, g: t_blur(x, g, 0.8, 1.2), 'a_noise': lambda x, g: (x + torch.randn(x.shape, generator=g) * 0.03).clamp(0, 1),
        'c_color_strong': lambda x, g: t_color(x, g, 0.45, hue=0.1), 'c_blur_strong': lambda x, g: t_blur(x, g, 2.0, 3.0),
        'c_crop_strong': lambda x, g: t_affine(x, g, smin=0.15, flip=True)}
    for si, (name, fn) in enumerate(specs.items()):
        p = f'{SETS}/{name}.npy'
        if os.path.exists(p): continue
        g = torch.Generator().manual_seed(9100 + si); out = []
        for i in range(0, N, 250):
            x = torch.from_numpy(fg[i:i + 250]).permute(0, 3, 1, 2).float() / 255.0
            out.append((fn(x, g).clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).numpy())
        np.save(p, np.concatenate(out)); print('jeu', name, flush=True)
    p = f'{SETS}/a_jpeg.npy'
    if not os.path.exists(p):
        out = []
        for im in fg:
            b = io.BytesIO(); Image.fromarray(im).save(b, 'JPEG', quality=60); out.append(np.array(Image.open(b).convert('RGB')))
        np.save(p, np.stack(out)); print('jeu a_jpeg', flush=True)
    p = f'{SETS}/c_mix.npy'      # melange de deux images a oublier de classes quelconques, lambda 0,35 a 0,65
    if not os.path.exists(p):
        g = torch.Generator().manual_seed(9200); perm = torch.randperm(N, generator=g).numpy(); lam = torch.empty(N).uniform_(0.35, 0.65, generator=g).numpy().reshape(N, 1, 1, 1)
        np.save(p, (lam * fg.astype(np.float32) + (1 - lam) * fg[perm].astype(np.float32)).round().astype(np.uint8)); print('jeu c_mix', flush=True)
    p = f'{SETS}/c_in100.npy'    # hors domaine : ImageNet-100 val (3000 images), simple regle de mesure
    if not os.path.exists(p) and os.path.exists(ROOT + '/proxy/in100_val.parquet'):
        try:
            import pyarrow.parquet as pq
            t = pq.read_table(ROOT + '/proxy/in100_val.parquet').slice(0, 3000).to_pylist(); out = []
            for r in t:
                im = r['image']; b = im['bytes'] if isinstance(im, dict) else im
                out.append(np.array(Image.open(io.BytesIO(b)).convert('RGB').resize((224, 224), Image.BICUBIC)))
            np.save(p, np.stack(out)); print('jeu c_in100', flush=True)
        except Exception as e: print('c_in100 indisponible :', repr(e)[:200], flush=True)
    return sorted(f[:-4] for f in os.listdir(SETS) if f.endswith('.npy'))


def check_file(path, ref_sd):
    sd = torch.load(path, map_location='cpu', weights_only=True); r = {'path': path, 'size_MB': round(os.path.getsize(path) / 1e6, 2)}
    r['sha256'] = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    r['keys_equal'] = list(sd.keys()) == list(ref_sd.keys()); r['dtypes'] = sorted({str(v.dtype) for v in sd.values()})
    r['all_float32'] = all(v.dtype == torch.float32 for v in sd.values()); r['finite'] = all(bool(torch.isfinite(v).all()) for v in sd.values())
    r['shapes_equal'] = all(k in ref_sd and tuple(v.shape) == tuple(ref_sd[k].shape) for k, v in sd.items())
    m = build(); r['strict_load'] = str(m.load_state_dict(sd, strict=True)); ch = {}
    for k, v in sd.items():
        if k in ref_sd and not torch.equal(v, ref_sd[k]):
            d = (v.double() - ref_sd[k].double()); ch[k] = round(float(d.norm() / ref_sd[k].double().norm().clamp_min(1e-12)), 5)
    r['n_changed'] = len(ch); r['changed_groups'] = sorted({('blocks.' + k.split('.')[1]) if k.startswith('blocks.') else k.split('.')[0] for k in ch})
    r['changed_gamma'] = {k: v for k, v in ch.items() if 'gamma' in k}; r['changed_other'] = {k: v for k, v in ch.items() if not k.startswith(('blocks.', 'head.', 'norm.'))}
    r['rel_change_max'] = max(ch.values()) if ch else 0.0; r['rel_change'] = {k: v for k, v in sorted(ch.items(), key=lambda kv: -kv[1])[:6]}
    r['size_ok'] = r['size_MB'] <= 450
    return m.to(dev).eval(), r


if __name__ == '__main__':
    fg = np.load(ROOT + '/forget_images.npz')['images']; assert fg.dtype == np.uint8 and fg.shape == (3500, 224, 224, 3)
    names = ['exact'] + make_sets(fg); ref = torch.load(ROOT + '/model.pt', map_location='cpu', weights_only=True)
    G = {s: (fg if s == 'exact' else np.load(f'{SETS}/{s}.npy', mmap_mode='r')) for s in names} if os.environ.get('LOWMEM') else {s: torch.from_numpy(fg if s == 'exact' else np.load(f'{SETS}/{s}.npy')).to(dev) for s in names}; print('jeux prets', len(G), flush=True)   # uint8 sur GPU : plus de travail CPU par lot
    for a in sys.argv[1:]:
        name, path = a.split('=', 1)
        if os.path.exists(f'{OUT}/{name}.npz') and os.path.exists(f'{OUT}/{name}.check.json'): print(name, 'deja fait', flush=True); continue
        m, r = check_file(path, ref); d = {}
        for s in names: d[s] = predict(m, G[s])
        z32 = predict(m, G['exact'][:500], fp16=False); r['fp16_vs_fp32_max_abs_logit'] = float(np.abs(z32 - d['exact'][:500]).max())
        r['fp16_vs_fp32_argmax_diff'] = int((z32.argmax(1) != d['exact'][:500].argmax(1)).sum()); r['logits_finite'] = bool(all(np.isfinite(v).all() for v in d.values()))
        np.savez(f'{OUT}/{name}.npz', **d); json.dump(r, open(f'{OUT}/{name}.check.json', 'w'), indent=1)
        print(name, json.dumps({k: r[k] for k in ['size_MB', 'keys_equal', 'all_float32', 'finite', 'strict_load', 'n_changed', 'changed_groups', 'changed_gamma', 'changed_other', 'rel_change_max',
                                                  'fp16_vs_fp32_max_abs_logit', 'fp16_vs_fp32_argmax_diff', 'logits_finite']}), flush=True)
        del m; torch.cuda.empty_cache()
