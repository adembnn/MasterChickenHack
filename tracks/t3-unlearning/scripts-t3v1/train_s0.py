"""Candidat S0 'sur' (B) : on n'entraine que tete + norme finale (norm.weight/bias, head.weight/bias). Tout le reste gele.
Cibles : softmax(z_orig/tau) sur vues des images a oublier. Ancre : sorties d'orig a T=2 sur melanges d'images a oublier (aucune image externe).
Les caracteristiques du jeton CLS (avant la norme finale) sont calculees une fois (le reste du reseau est fige) : l'entrainement est instantane.
usage : train_s0.py TAU LAMBDA_ANCRE"""
import sys; sys.path.insert(0, '/p/scratch/training2666/destors1/MasterChickenHack/unlearning/adem/t3v1')
from unlearn import *
tau = float(sys.argv[1]); lam = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
name = f'S0_tau{sys.argv[1]}'
orig = load_orig(); fg = load_forget(); N = len(fg); torch.manual_seed(0)
fc = f'{CACHE}/s0_feats.pt'
if os.path.exists(fc): D = torch.load(fc)
else:
    g = gen(2024); V = 12; Fv, Zv = [], []; Fm, Zm = [], []
    with torch.no_grad():
        for v in range(V):
            for i in range(0, N, 250):
                x = to01(fg[i:i + 250])
                if v == 0: vv = x
                else: vv, _ = train_view(x, g, strong=(v % 2 == 0))
                with ac(): c = prenorm_cls(orig, norm(vv))
                with ac(): z = head_from_cls(orig, c)
                Fv.append(c.float().cpu()); Zv.append(z.float().cpu())
        for k in range(16000 // 250):
            idx = torch.randint(0, N, (250,), generator=torch.Generator().manual_seed(k + 100))
            mx = make_mix(to01(fg[idx]), g)
            with ac(): c = prenorm_cls(orig, norm(mx))
            with ac(): z = head_from_cls(orig, c)
            Fm.append(c.float().cpu()); Zm.append(z.float().cpu())
    D = {'Fv': torch.cat(Fv), 'Zv': torch.cat(Zv), 'Fm': torch.cat(Fm), 'Zm': torch.cat(Zm)}; torch.save(D, fc)
Fv, Zv, Fm, Zm = [D[k].to(dev) for k in ['Fv', 'Zv', 'Fm', 'Zm']]
print('features', Fv.shape, Fm.shape, flush=True)
head, nrm = orig.head, orig.norm
params = [orig.head.weight, orig.head.bias, orig.norm.weight, orig.norm.bias]
for p in orig.parameters(): p.requires_grad_(False)
for p in params: p.requires_grad_(True)
opt = torch.optim.AdamW(params, lr=2e-4, weight_decay=0.0)
steps = 1500; bs, ba = 512, 256
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-4, total_steps=steps, pct_start=0.05)
for it in range(steps):
    i = torch.randint(0, len(Fv), (bs,), device=dev); j = torch.randint(0, len(Fm), (ba,), device=dev)
    zf = head(nrm(Fv[i])); q = F.softmax(Zv[i] / tau, 1)
    lf = (q * (q.clamp_min(1e-12).log() - F.log_softmax(zf, 1))).sum(1).mean()
    q2 = F.softmax(Zv[i] / (2 * tau), 1); l2 = (q2 * (q2.clamp_min(1e-12).log() - F.log_softmax(zf / 2, 1))).sum(1).mean()
    za = head(nrm(Fm[j])); po = F.softmax(Zm[j] / 2, 1)
    la = (po * (po.clamp_min(1e-12).log() - F.log_softmax(za / 2, 1))).sum(1).mean()
    loss = lf + l2 + lam * la
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if it % 250 == 0 or it == steps - 1: print(it, f'forget KL {lf.item():.4f}  T2 {l2.item():.4f}  ancre {la.item():.5f}', flush=True)
sd = build_sd(orig); torch.save(sd, f'{CAND}/{name}.pt')
lg0 = torch.from_numpy(np.load(f'{CACHE}/exact_orig_logits.npy'))
torch.save({'q2': F.softmax(lg0 / (2 * tau), 1)}, f'{CACHE}/targets_{name}.pt')
print('saved', name)
