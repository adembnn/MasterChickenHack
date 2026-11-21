"""Candidat S1 'cibles par quantiles' (C, methode N1) : 4 derniers blocs + norme + tete ; gammas LayerScale geles.
cibles : p_y(x) = sigmoid(a - b z(x)), (a,b) choisis pour que moyenne et ecart-type de -log p_y valent (mu, sigma) lies a A (accuracy non-membre supposee).
perte : KL(q || softmax z) (T=1) + KL(q2 || softmax(z/2)) (T=2) sur vues augmentees + lam * ancre (KL T=2 vers orig sur melanges d'images a oublier)
        + wm * perte de moments (moyenne/ecart-type de la CE sur les vues faibles)
usage : train_s1.py NAME A LR STEPS LAM [WM] [NBLOCKS] [BETA_SEV]"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from targets import *
name, A, lr, steps, lam = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
wm = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0; nblk = int(sys.argv[7]) if len(sys.argv) > 7 else 4
sevc = float(sys.argv[8]) if len(sys.argv) > 8 else 0.15
zkey = sys.argv[9] if len(sys.argv) > 9 else 'z'
orig = load_orig(); ref = load_orig(); fg = load_forget(); N = len(fg); torch.manual_seed(0)
P = torch.load(f'{CACHE}/s1_prep.pt'); yhat, z, w = P['yhat'].to(dev), P[zkey], P['w'].to(dev)
mu, sd = gold_moments(A); a, b, err = fit_ab(z, mu, sd)
p_y = torch.sigmoid(a - b * z).to(dev).float()
L = -torch.log(p_y)
q_clean, q2_clean = make_q(p_y, yhat, w)
print(f'{name}: A={A} mu={mu:.3f} sd={sd:.3f} -> a={a:.2f} b={b:.2f} err={err:.2e} ; cible : loss moy {L.mean():.3f} sd {L.std():.3f} ; '
      f'part argmax(q)!=yhat {(q_clean.argmax(1) != yhat).float().mean():.3f} ; part p_y<0.5 {(p_y < 0.5).float().mean():.3f} ; part p_y>0.95 {(p_y > 0.95).float().mean():.3f}', flush=True)
torch.save({'q2': q2_clean.cpu(), 'q': q_clean.cpu(), 'p_y': p_y.cpu(), 'mu': mu, 'sd': sd, 'a': a, 'b': b}, f'{CACHE}/targets_{name}.pt')
# parametres entraines
for p in orig.parameters(): p.requires_grad_(False)
tr = []
for n, p in orig.named_parameters():
    blk = n.startswith('blocks.') and int(n.split('.')[1]) >= 12 - nblk
    if (blk or n.startswith('norm.') or n.startswith('head.')) and 'gamma' not in n: p.requires_grad_(True); tr.append(p)
print('parametres entraines', sum(p.numel() for p in tr), flush=True)
opt = torch.optim.AdamW(tr, lr=lr, weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 30) * 0.5 * (1 + math.cos(math.pi * s / steps)))
g = gen(31337); BF, BA = 128, 64
perm = torch.randperm(N, generator=torch.Generator().manual_seed(1)); pp = 0
t0 = time.time(); fgd = fg
for it in range(steps):
    if pp + BF > N: perm = torch.randperm(N, generator=torch.Generator().manual_seed(it)); pp = 0
    idx = perm[pp:pp + BF]; pp += BF; ig = idx.to(dev)
    x = to01(fgd[idx]); half = BF // 2
    vw, sw = train_view(x[:half], g, False); vs, ss = train_view(x[half:], g, True)
    v = torch.cat([vw, vs]); sev = torch.cat([sw, ss])
    py = p_y[ig] * (1 - sevc * sev); N_ = BF
    q = (1 - py)[:, None] * w[ig]; q[torch.arange(N_), yhat[ig]] += py
    h = q.clamp_min(1e-12).sqrt(); h = h / h.sum(1, keepdim=True); q2 = 0.85 * h + 0.15 * w[ig]
    j = torch.randint(0, N, (BA,), generator=torch.Generator().manual_seed(10 ** 6 + it))
    mx = make_mix(to01(fgd[j]), g)
    with torch.no_grad(), ac(): zo = ref(norm(mx)).float()
    orig.train(False)  # pas de dropout/droppath de toute facon (timm defaut 0)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        zz = orig(norm(torch.cat([v, mx]))).float()
    zf, za = zz[:BF], zz[BF:]
    lf = kl(q, F.log_softmax(zf, 1)).mean(); l2 = kl(q2, F.log_softmax(zf / 2, 1)).mean()
    po = F.softmax(zo / 2, 1); la = kl(po, F.log_softmax(za / 2, 1)).mean()
    ce = F.cross_entropy(zf[:half], yhat[ig][:half], reduction='none')            # vues faibles
    lm = (ce.mean() - L.mean()) ** 2 + (ce.std() - L.std()) ** 2
    loss = lf + l2 + lam * la + wm * lm
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(tr, 1.0); opt.step(); sched.step()
    if it % 50 == 0 or it == steps - 1:
        print(f'{it:4d} {time.time() - t0:5.0f}s  KL1 {lf.item():.3f} KL2 {l2.item():.3f} ancre {la.item():.4f} moments {lm.item():.3f}  CE faibles moy {ce.mean().item():.3f} sd {ce.std().item():.3f}  wrong {(zf.argmax(1) != yhat[ig]).float().mean().item():.3f}', flush=True)
torch.save(build_sd(orig), f'{CAND}/{name}.pt'); print('saved', name)
