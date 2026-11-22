"""Entraine un modele du simulateur. usage : train.py NAME WORLD {orig|gold} {scratch|pre} EPOCHS LR SEED [BS] [RRC_LO] [DROP_PATH]
orig = entraine sur tout le train (oubli compris) ; gold = entraine sans l'oubli. Meme recette pour les deux, seule la graine change."""
from common import *
name, world, kind, init, epochs, lr, seed = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), float(sys.argv[6]), int(sys.argv[7])
bs = int(sys.argv[8]) if len(sys.argv) > 8 else 256; lo = float(sys.argv[9]) if len(sys.argv) > 9 else 0.35; dp = float(sys.argv[10]) if len(sys.argv) > 10 else 0.0
out = f'{ROOT}/runs/{world}'; os.makedirs(out, exist_ok=True); ck = f'{out}/{name}.ckpt'
torch.manual_seed(seed); W = World(world); key = 'train' if kind == 'orig' else 'retain'
Xtr = W.x(key); Ytr = W.y(key).to(dev); N = len(Xtr); spe = N // bs; steps = epochs * spe; warm = min(5 * spe, steps // 10)
m = (build(dp) if init == 'scratch' else build_pre(dp)).to(dev)
decay = [p for n, p in m.named_parameters() if p.ndim > 1]; nod = [p for n, p in m.named_parameters() if p.ndim <= 1]
opt = torch.optim.AdamW([{'params': decay, 'weight_decay': 0.05}, {'params': nod, 'weight_decay': 0.0}], lr=lr, betas=(0.9, 0.999))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
g = gen(seed * 7 + 1); ep0 = 0
if os.path.exists(ck):
    c = torch.load(ck, map_location=dev); m.load_state_dict(c['m']); opt.load_state_dict(c['o']); sched.load_state_dict(c['s']); ep0 = c['ep']; print('reprise a', ep0)
print(f'{name}: {kind} {init} N={N} epochs={epochs} steps={steps} lr={lr} bs={bs}', flush=True); t0 = time.time()
for ep in range(ep0, epochs):
    m.train(); perm = torch.randperm(N, device=dev, generator=g); tl = 0; ta = 0
    for i in range(spe):
        idx = perm[i * bs:(i + 1) * bs]; x, _ = affine(to01(Xtr[idx.cpu()]), g, scale=(lo, 1.), flip=True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16): z = m(norm(x))
        loss = F.cross_entropy(z.float(), Ytr[idx]); opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step(); tl += loss.item(); ta += (z.argmax(1) == Ytr[idx]).float().mean().item()
    if ep % 5 == 4 or ep == epochs - 1 or ep == ep0:
        zv = logits_u8(m, W.x('val')); va = (zv.argmax(1) == W.y('val')).float().mean().item()
        print(f'ep {ep + 1:3d} {time.time() - t0:5.0f}s loss {tl / spe:.4f} acc {ta / spe:.4f} val {va:.4f} img/s {(ep + 1 - ep0) * spe * bs / (time.time() - t0):.0f}', flush=True)
    if ep % 10 == 9 and ep != epochs - 1: torch.save({'m': m.state_dict(), 'o': opt.state_dict(), 's': sched.state_dict(), 'ep': ep + 1}, ck)
m.eval(); sd = {k: v.detach().cpu().float() for k, v in m.state_dict().items()}; torch.save(sd, f'{out}/{name}.pt')
r = eval_all(m, W, f'{out}/{name}.npz'); L = labels(W)
if os.path.exists(ck): os.remove(ck)
zf = torch.from_numpy(r['fx']); yf = torch.from_numpy(L['fx']); ce = F.cross_entropy(zf, yf, reduction='none')
print(f'FIN {name}: oubli exact acc {(zf.argmax(1) == yf).float().mean():.4f} maxprob {F.softmax(zf, 1).max(1).values.mean():.4f} CE moy {ce.mean():.4f} sd {ce.std():.4f} | '
      f'ctrl acc {(r["ctrl"].argmax(1) == L["ctrl"]).mean():.4f} val acc {(r["val"].argmax(1) == L["val"]).mean():.4f} | {time.time() - t0:.0f}s', flush=True)
