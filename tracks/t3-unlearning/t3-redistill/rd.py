"""t3-redistill : reinitialiser le haut du reseau (tete + k derniers blocs), puis le reapprendre par distillation d'orig
sur des images que le modele n'a jamais vues telles quelles. Aucune image a oublier exacte dans l'entrainement.
Reservoirs (--res, liste separee par des virgules, parts egales) :
  mix        melanges d'images a oublier ne contenant jamais plus de la moitie d'une image (mixup 2-3 images, coupe moitie-moitie, mosaique 2x2)
  npz:CHEMIN images uint8 [N,224,224,3] (cle 'images' ou 'x' ou premier tableau) : images generees, COCO...
--hold : images a oublier ecartees de la source des melanges (c = reserve de t3-methodes, fI/K = pli I sur K, none).
Mesures aux conditions du serveur (eval, autocast fp16). Etiquette = classe predite par orig sur l'image exacte."""
import sys, os, argparse
sys.path.append(os.path.expanduser('~/eac/t3/t3-methodes/code'))      # common.py : d'abord celui du dossier du script, sinon celui de t3-methodes
from common import *
import torch.nn as nn
ME = BASE + '/t3-redistill'; OUT = ME + '/out'; CAN = ME + '/candidates'
for d in (OUT, CAN): os.makedirs(d, exist_ok=True)
ap = argparse.ArgumentParser()
ap.add_argument('--name', required=True); ap.add_argument('--k', type=int, default=2); ap.add_argument('--mode', default='reset')
ap.add_argument('--sigma', type=float, default=1.0); ap.add_argument('--res', default='mix'); ap.add_argument('--hold', default='c')
ap.add_argument('--steps', type=int, default=2000); ap.add_argument('--lr', type=float, default=5e-4); ap.add_argument('--wd', type=float, default=0.05)
ap.add_argument('--bs', type=int, default=192); ap.add_argument('--wfeat', type=float, default=0.); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--gamma', default='keep'); ap.add_argument('--save', action='store_true'); ap.add_argument('--extra', default='')
ap.add_argument('--evalevery', type=int, default=0); ap.add_argument('--fseed', type=int, default=0); ap.add_argument('--trunkpt', default=''); ap.add_argument('--exact', type=float, default=0.); ap.add_argument('--wex', type=float, default=1.); ap.add_argument('--lite', action='store_true'); ap.add_argument('--eval0', action='store_true')
a_ = ap.parse_args(); K = a_.k
torch.manual_seed(a_.seed); ref = load_orig(); model = load_orig(); fgd = load_forget().to(dev); N = len(fgd)
if a_.trunkpt: model.load_state_dict(torch.load(os.path.expanduser(a_.trunkpt), map_location='cpu', weights_only=True), strict=True); print('tronc gele pris dans', a_.trunkpt, flush=True)
if not os.path.exists(f'{CACHE}/exact_orig_logits.npy'): np.save(f'{CACHE}/exact_orig_logits.npy', logits_u8(ref, load_forget()).numpy())
lg0 = torch.from_numpy(np.load(f'{CACHE}/exact_orig_logits.npy')); yhat = lg0.argmax(1)
part = torch.load(f'{CACHE}/split.pt') if a_.hold == 'c' else None
if a_.hold == 'c': held = part == 2
elif a_.hold == 'none': held = torch.zeros(N, dtype=torch.bool)
else:
    fi, fk = [int(v) for v in a_.hold[1:].split('/')]; rng = np.random.RandomState(777 + a_.fseed); fold = np.zeros(N, dtype=np.int64)
    for c in yhat.unique().tolist():
        idx = (yhat == c).nonzero().flatten().numpy(); rng.shuffle(idx); fold[idx] = np.arange(len(idx)) % fk
    held = torch.from_numpy(fold == fi)
yhat_d = yhat.to(dev); src = (~held).nonzero().flatten().to(dev); hidx = held.nonzero().flatten(); sidx = (~held).nonzero().flatten()
print(f'{a_.name}: {vars(a_)} | source {len(src)} images, ecartees {len(hidx)}', flush=True)


def top_tok(m, h, k):
    for b in (m.blocks[12 - k:] if k > 0 else []): h = b(h)
    return h


def headf(m, h): return m.forward_head(m.norm(h))


# ---------- reservoirs ----------
def rrc(x, g, lo, hi): return affine(x, g, scale=(lo, hi), flip=True)[0]


def mix_half(pool, B, g):
    """Melanges ou aucune image source ne pese plus de la moitie. pool : indices (cuda) dans fgd."""
    n = len(pool); n1 = B // 3; n2 = B // 3; n3 = B - n1 - n2; H = W = 224

    def pos(m): return torch.randint(0, n, (m,), device=dev, generator=g)

    def other(p): return (p + torch.randint(1, n, (len(p),), device=dev, generator=g)) % n

    def img(p, lo=0.5, hi=1.0): return rrc(to01(fgd[pool[p]]), g, lo, hi)
    p1 = pos(n1); p2 = other(p1); p3 = other(p1)
    a, b, c = img(p1), img(p2), img(p3); lam = U(n1, 0.15, 0.5, g).view(-1, 1, 1, 1); three = (U(n1, 0, 1, g) < 0.5).float().view(-1, 1, 1, 1)
    m1 = three * (0.5 * a + lam * b + (0.5 - lam) * c) + (1 - three) * (0.5 * a + 0.5 * b)
    p1 = pos(n2); p2 = other(p1); a, b = img(p1), img(p2)
    r = U(n2, 0.5, 1.0, g); bw = (r * W).long(); bh = (0.5 / r * H).long()
    x0 = (U(n2, 0, 1, g) * (W - bw + 1)).long(); y0 = (U(n2, 0, 1, g) * (H - bh + 1)).long()
    ii = torch.arange(H, device=dev).view(1, H, 1); jj = torch.arange(W, device=dev).view(1, 1, W)
    mask = ((ii >= y0.view(-1, 1, 1)) & (ii < (y0 + bh).view(-1, 1, 1)) & (jj >= x0.view(-1, 1, 1)) & (jj < (x0 + bw).view(-1, 1, 1))).unsqueeze(1)
    m2 = torch.where(mask, b, a)
    t = [F.interpolate(img(pos(n3), 0.2, 0.5), size=112, mode='bilinear', antialias=True, align_corners=False) for _ in range(4)]
    m3 = torch.cat([torch.cat([t[0], t[1]], 3), torch.cat([t[2], t[3]], 3)], 2)
    return torch.cat([m1, m2, m3]).clamp(0, 1)


RES = []
for spec in a_.res.split(','):
    if spec == 'mix': RES.append(('mix', None))
    else:
        kind, path = spec.split(':', 1); z = np.load(os.path.expanduser(path), mmap_mode='r') if path.endswith('.npy') else np.load(os.path.expanduser(path))
        arr = z if isinstance(z, np.ndarray) else z[[k for k in ('images', 'x', 'img') if k in z.files][0] if any(k in z.files for k in ('images', 'x', 'img')) else z.files[0]]
        t = torch.from_numpy(np.ascontiguousarray(arr)).to(dev); RES.append((kind, t)); print('reservoir', spec, tuple(t.shape), flush=True)


def sample(B, g):
    out = []; per = B // len(RES)
    for i, (kind, t) in enumerate(RES):
        b = per if i < len(RES) - 1 else B - per * (len(RES) - 1)
        if kind == 'mix': out.append(mix_half(src, b, g))
        else:
            j = torch.randint(0, len(t), (b,), device=dev, generator=g); out.append(rrc(to01(t[j]), g, 0.6, 1.0))
    return torch.cat(out)


# ---------- reinitialisation ----------
mods = (list(model.blocks[12 - K:]) if K > 0 else []) + [model.norm, model.head]
with torch.no_grad():
    if a_.mode == 'reset':
        for mod in mods:
            for sub in mod.modules():
                if isinstance(sub, nn.Linear): nn.init.trunc_normal_(sub.weight, std=.02); nn.init.zeros_(sub.bias)
                elif isinstance(sub, nn.LayerNorm): nn.init.ones_(sub.weight); nn.init.zeros_(sub.bias)
        if a_.gamma == 'reset':
            for b in model.blocks[12 - K:] if K > 0 else []: b.ls1.gamma.fill_(1e-4); b.ls2.gamma.fill_(1e-4)
    elif a_.mode == 'noise':
        for mod in mods:
            for n_, p in mod.named_parameters():
                if p.ndim >= 2: p.add_(a_.sigma * p.std() * torch.randn_like(p))
for p in model.parameters(): p.requires_grad_(False)
tr = [p for mod in mods for p in mod.parameters()]
for p in tr: p.requires_grad_(True)
dec = [p for p in tr if p.ndim >= 2]; nodec = [p for p in tr if p.ndim < 2]
opt = torch.optim.AdamW([{'params': dec, 'weight_decay': a_.wd}, {'params': nodec, 'weight_decay': 0.}], lr=a_.lr)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * s / max(1, a_.steps))))
g = gen(4242 + a_.seed)

# ---------- jeux de mesure ----------
SL = ['jpeg50', 'rot15', 'cropflip', 'mild']; ST = ['blur_strong', 'color_strong']; XM = {}; XY = {}
VL = []
if a_.lite:      # mode leger : images exactes + trois copies alterees fabriquees ici avec une graine fixe (memes copies pour toutes les graines et tous les plis)
    SL = []; ST = []; D = {'exact': (load_forget(), lg0)}; gv = gen(4321); fgc = load_forget(); V = {'vrot': [], 'vcrop': [], 'vmild': []}
    for i in range(0, N, 250):
        xv = to01(fgc[i:i + 250])
        V['vrot'].append(to_u8(affine(xv, gv, rot=15., zoom=1.1)[0]).cpu()); V['vcrop'].append(to_u8(affine(xv, gv, scale=(0.5, 1.), flip=True)[0]).cpu())
        vm, _ = affine(xv, gv, scale=(0.8, 1.)); vm = color(vm, gv, 0.15, 0.15, 0.15); vm = (vm + torch.randn(vm.shape, device=dev, generator=gv) * 0.02).clamp(0, 1); V['vmild'].append(to_u8(vm).cpu())
    for k in V: u = torch.cat(V[k]); D[k] = (u, logits_u8(ref, u)); VL.append(k)
else:
    D = {'exact': (load_forget(), lg0)}
    for k in SL + ST: D[k] = (np.load(f'{CACHE}/held_{k}.npy', mmap_mode='r'), torch.from_numpy(np.load(f'{CACHE}/held_{k}_orig_logits.npy')))
    XM = {'melanges_tm': (np.load(f'{CACHE}/mixes.npy', mmap_mode='r'), torch.from_numpy(np.load(f'{CACHE}/mixes_orig_logits.npy')))}
    ge = gen(999)
    if len(hidx) >= 8:
        hm = torch.cat([to_u8(mix_half(hidx.to(dev), 250, ge)).cpu() for _ in range(8)]); XM['melanges_ecartes'] = (hm, logits_u8(ref, hm))
    sm = torch.cat([to_u8(mix_half(src, 250, ge)).cpu() for _ in range(8)]); XM['melanges_source'] = (sm, logits_u8(ref, sm))
    for spec in [s for s in a_.extra.split(',') if s]:
        nm, path = spec.split(':', 1); z = np.load(os.path.expanduser(path)); arr = z if isinstance(z, np.ndarray) else z[[k for k in ('images', 'x', 'img') if k in z.files][0]]
        u = torch.from_numpy(np.ascontiguousarray(arr)); XM[nm] = (u, logits_u8(ref, u))
        if not isinstance(z, np.ndarray) and 'y' in z.files: XY[nm] = torch.from_numpy(z['y'])


def klt2(lo, lg): return kl(F.softmax(lo / 2, 1), F.log_softmax(lg / 2, 1))


def fstat(lg, lo, y):
    ce = F.cross_entropy(lg, y, reduction='none')
    return {'ce_mean': ce.mean().item(), 'ce_std': ce.std().item(), 'wrong': (lg.argmax(1) != y).float().mean().item(), 'maxprob_t2': F.softmax(lg / 2, 1).max(1)[0].mean().item(),
            'kl_orig': klt2(lo, lg).mean().item(), 'agree': (lg.argmax(1) == lo.argmax(1)).float().mean().item()}


def evaluate(tag, save_logits=False):
    out = {'name': a_.name, 'tag': tag, 'args': vars(a_)}; LG = {}
    for k in ['exact'] + SL + ST + VL:
        u8, lo = D[k]; lg = logits_u8(model, u8); LG[k] = lg
        for nm, ii in [('ecartees', hidx), ('source', sidx)]:
            if len(ii): out[f'{k}_{nm}'] = fstat(lg[ii], lo[ii], yhat[ii])
    for nm in ['ecartees', 'source'] if SL else []:
        if f'exact_{nm}' in out: out[f'legeres_{nm}'] = {k: float(np.mean([out[f'{s}_{nm}'][k] for s in SL])) for k in out[f'exact_{nm}']}
    for k, (u8, lo) in XM.items():
        lg = logits_u8(model, u8); out[k] = {'agree': (lg.argmax(1) == lo.argmax(1)).float().mean().item(), 'kl': klt2(lo, lg).mean().item(), 'maxprob_orig_t1': F.softmax(lo, 1).max(1)[0].mean().item()}
        if k in XY:
            ce = F.cross_entropy(lg, XY[k], reduction='none'); out[k].update(acc_sub=(lg.argmax(1) == XY[k]).float().mean().item(), acc_orig=(lo.argmax(1) == XY[k]).float().mean().item(), ce_mean=ce.mean().item(), ce_std=ce.std().item())
    json.dump(out, open(f'{OUT}/{a_.name}{tag}.json', 'w'), indent=1)
    if save_logits: np.savez(f'{OUT}/{a_.name}_logits.npz', held=held.numpy(), **{k: v.numpy().astype(np.float32) for k, v in LG.items()})

    def r(d): return f"{d['ce_mean']:.2f} / {d['ce_std']:.2f} / {d['wrong']:.3f} / {d['maxprob_t2']:.3f}"
    row = f"| {a_.name}{tag} | "
    for nm in ['ecartees', 'source']:
        row += (f"{r(out['exact_' + nm])} | {r(out['legeres_' + nm]) if SL else '-'} | " if f'exact_{nm}' in out else '- | - | ')
    for nm in ['ecartees', 'source']:
        row += (' ; '.join(f"{out[f'{k}_{nm}']['agree']:.3f}/{out[f'{k}_{nm}']['kl_orig']:.3f}" for k in ST) + ' | ' if f'exact_{nm}' in out else '- | ')
    row += ' ; '.join(f"{k} {v['agree']:.3f}/{v['kl']:.3f}" + (f" (prec. {v['acc_sub']:.3f} vs orig {v['acc_orig']:.3f}, CE {v['ce_mean']:.2f}/{v['ce_std']:.2f})" if 'acc_sub' in v else '') for k, v in out.items() if isinstance(v, dict) and 'maxprob_orig_t1' in v) + ' |'
    print(row, flush=True); return out


print('| variante | EXACTES ecartees : CE moy / sd / classe changee / maxprob T2 | copies legeres ecartees | EXACTES source des melanges | copies legeres source | flou fort ; couleur forte (ecartees) : accord orig / KL T2 | idem (source) | autres jeux : accord orig / KL T2 |')
print('|---|---|---|---|---|---|---|---|', flush=True)
t0 = time.time(); model.eval(); la = lf = 0.
if a_.eval0: evaluate('@0')
for it in range(a_.steps):
    nbe = int(a_.bs * a_.exact); nm = a_.bs - nbe; xm = sample(nm, g)
    if nbe: je = src[torch.randint(0, len(src), (nbe,), device=dev, generator=g)]; x = norm(torch.cat([xm, rrc(to01(fgd[je]), g, 0.6, 1.0)]))
    else: x = norm(xm)
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        h = trunk(ref, x, K); tt_ = top_tok(ref, h, K); zo = headf(ref, tt_).float()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        hs = trunk(model, x, K).detach() if a_.trunkpt else h
        st = top_tok(model, hs, K); z = headf(model, st).float()
    l2 = kl(F.softmax(zo[:nm] / 2, 1), F.log_softmax(z[:nm] / 2, 1)).mean(); l1 = kl(F.softmax(zo[:nm], 1), F.log_softmax(z[:nm], 1)).mean()
    loss = 4 * l2 + l1
    if nbe: loss = loss + a_.wex * F.cross_entropy(z[nm:], yhat_d[je])
    if a_.wfeat > 0 and K > 0:
        lf = F.mse_loss(st.float(), tt_.float()) / tt_.float().var(); loss = loss + a_.wfeat * lf
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(tr, 1.0); opt.step(); sched.step()
    if it % 100 == 0 or it == a_.steps - 1:
        print(f'# {it:5d} {time.time() - t0:5.0f}s KL2 {l2.item():.4f} KL1 {l1.item():.4f} feat {float(lf):.4f} accord lot {(z.argmax(1) == zo.argmax(1)).float().mean().item():.3f}', flush=True)
    if a_.evalevery and it > 0 and it % a_.evalevery == 0: evaluate(f'@{it}'); model.eval()
evaluate('', save_logits=True)
if a_.save:
    torch.save(build_sd(model), f'{CAN}/{a_.name}.pt'); print('ecrit', f'{CAN}/{a_.name}.pt', flush=True)
print('fini', a_.name, f'{time.time() - t0:.0f}s', flush=True)
