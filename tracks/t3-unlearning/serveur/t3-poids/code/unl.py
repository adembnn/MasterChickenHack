"""t3-poids : oubli dans l'espace des poids, pertes composables. Aucune image externe : forget set + orig seulement.
--init   none | noise:S:LO:HI (bruit gaussien relatif S x rms du tenseur sur les blocs LO..HI-1) | reinit:LO:HI (retirage des poids) | shrink:S:LO:HI (poids x S) | delta:NOM:COEF
--forget none | ga (montee : -CE) | npo:BETA | cap:M (charniere : pousse la marge sous M, s'arrete ensuite) | pcap:P (pousse la probabilite de la pseudo-etiquette sous P, image par image, puis s'arrete)   poids --wf, applique aux images exactes et a des vues legeres
--wa     poids d'une descente CE(pseudo-etiquette) sur des vues FORTEMENT augmentees (garde ce qui generalise)
--wk     poids de l'ancre KL(orig||sub) (T=2 et T=1) sur des melanges d'images (mixup, cutmix, permutation de patchs) ; --ka 1 ajoute les vues fortes a l'ancre
--lo     premier bloc entraine (0 = tout le modele) ; la norme finale et la tete sont toujours entrainees"""
import sys, os, argparse
sys.path.insert(0, os.path.expanduser('~/eac/t3/t3-methodes/code')); os.environ.setdefault('T3ME', os.path.expanduser('~/eac/t3/t3-poids'))
from common import *
ap = argparse.ArgumentParser()
ap.add_argument('--name', required=True); ap.add_argument('--fset', default='ab'); ap.add_argument('--init', default='none'); ap.add_argument('--forget', default='none')
ap.add_argument('--wf', type=float, default=1.0); ap.add_argument('--wa', type=float, default=0.0); ap.add_argument('--wk', type=float, default=0.0); ap.add_argument('--ka', type=int, default=0)
ap.add_argument('--lo', type=int, default=0); ap.add_argument('--steps', type=int, default=200); ap.add_argument('--lr', type=float, default=1e-5); ap.add_argument('--bs', type=int, default=48)
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--light', type=float, default=0.5, help='part du lot oubli passee en vue legere'); ap.add_argument('--log', type=int, default=50); ap.add_argument('--alt', type=int, default=1, help='1 = deux AdamW separes (oubli / ancre), pas alternes : le gradient d oubli sature n est pas ecrase par l ancre'); ap.add_argument('--lrk', type=float, default=0.0); ap.add_argument('--kn', type=int, default=1, help='pas d ancre par pas d oubli'); ap.add_argument('--ckpt_m', default='', help='seuils de CE lissee ou l on ecrit un candidat intermediaire'); ap.add_argument('--stop_m', type=float, default=0.0, help='arret quand la CE moyenne (lissee) des images exactes du lot atteint cette valeur')
a = ap.parse_args()
torch.manual_seed(a.seed); g = gen(a.seed)
S = split_sets(); idx = S[a.fset]; fg = load_forget()[idx]; N = len(idx)
lg0 = torch.from_numpy(np.load(f'{CACHE}/exact_orig_logits.npy'))[idx]; yhat = lg0.argmax(1).to(dev); lp0y = F.log_softmax(lg0, 1).gather(1, lg0.argmax(1)[:, None])[:, 0].to(dev)
orig_sd = torch.load(BASE + '/model.pt', map_location='cpu', weights_only=True)
m = tt.build_model().to(dev); m.load_state_dict(orig_sd, strict=True)
T_ = tt.build_model().to(dev).eval(); T_.load_state_dict(orig_sd, strict=True)
for p in T_.parameters(): p.requires_grad_(False)


def inblk(n, lo, hi): return n.startswith('blocks.') and lo <= int(n.split('.')[1]) < hi


it = a.init.split(':')
with torch.no_grad():
    if it[0] in ('noise', 'reinit', 'shrink'):
        lo, hi = int(it[-2]), int(it[-1])
        for n, p in m.named_parameters():
            if not inblk(n, lo, hi) or 'gamma' in n or p.dim() < 2: continue
            if it[0] == 'noise': p.add_(torch.randn(p.shape, device=dev, generator=g) * float(it[1]) * p.pow(2).mean().sqrt())
            elif it[0] == 'shrink': p.mul_(float(it[1]))
            else: torch.nn.init.trunc_normal_(p, std=0.02)
    elif it[0] == 'delta':
        d = torch.load(f'{DELTA}/{it[1]}.pt')['delta']; sd = m.state_dict()
        for k, v in d.items(): sd[k].add_(float(it[2]) * v.to(dev))
names = [n for n, p in m.named_parameters() if (a.lo == 0 or inblk(n, a.lo, 12) or n.startswith(('norm.', 'head.')))]
for n, p in m.named_parameters(): p.requires_grad_(n in names)
opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1.0, (t + 1) / 20))
fk = a.forget.split(':'); t0 = time.time(); m.train(); ema = None; CK = sorted(float(x) for x in a.ckpt_m.split(',') if x)


def margin(z, y): zy = z.gather(1, y[:, None])[:, 0]; zo = z.scatter(1, y[:, None], -1e4).max(1)[0]; return zy - zo


def fwd(x):
    with torch.autocast('cuda', dtype=torch.bfloat16): return m(norm(x)).float()


def anchor(x):
    with torch.no_grad(), ac(): zt = T_(norm(x)).float()
    z = fwd(x); return 4 * kl(F.softmax(zt / 2, 1), F.log_softmax(z / 2, 1)).mean() + kl(F.softmax(zt, 1), F.log_softmax(z, 1)).mean()


TP = [p for p in m.parameters() if p.requires_grad]
opt_k = torch.optim.AdamW(TP, lr=a.lrk or a.lr, weight_decay=0.0)
for st in range(a.steps):
    b = torch.randint(0, N, (a.bs,)); x = to01(fg[b]); y = yhat[b]; rec = {}; Lf = None; Lk = None
    if fk[0] != 'none':
        nl = int(a.bs * a.light); xf = x.clone()
        if nl: xf[:nl] = train_view(x[:nl], g, False)[0]
        z = fwd(xf)
        if fk[0] == 'ga': lf = -F.cross_entropy(z, y)
        elif fk[0] == 'npo': be = float(fk[1]); lf = (2 / be) * F.softplus(be * (F.log_softmax(z, 1).gather(1, y[:, None])[:, 0] - lp0y[b])).mean()
        elif fk[0] == 'cap': lf = F.relu(margin(z, y) - float(fk[1])).mean()
        elif fk[0] == 'pcap': lf = F.relu(F.log_softmax(z, 1).gather(1, y[:, None])[:, 0] - math.log(float(fk[1]))).mean()
        Lf = a.wf * lf; rec['f'] = lf.item(); rec['mar'] = margin(z.detach(), y).mean().item(); cex = F.cross_entropy(z.detach()[nl:], y[nl:]).item(); ema = cex if ema is None else 0.9 * ema + 0.1 * cex; rec['ce_ex'] = ema
    if a.alt and Lf is not None:
        opt.zero_grad(set_to_none=True); Lf.backward(); torch.nn.utils.clip_grad_norm_(TP, 5.0); opt.step(); sched.step(); Lf = None
    for kk in range(a.kn if a.alt else 1):
        Lk = None
        if a.wa:
            xs = train_view(x, g, True)[0]; la = F.cross_entropy(fwd(xs), y); Lk = a.wa * la; rec['a'] = la.item()
        if a.wk:
            b2 = torch.randint(0, N, (a.bs,)); lk = anchor(make_mix(to01(fg[b2]), g))
            if a.ka: lk = 0.5 * lk + 0.5 * anchor(train_view(to01(fg[b2]), g, True)[0])
            Lk = (Lk if Lk is not None else 0.) + a.wk * lk; rec['k'] = lk.item()
        if a.alt and Lk is not None: opt_k.zero_grad(set_to_none=True); Lk.backward(); torch.nn.utils.clip_grad_norm_(TP, 5.0); opt_k.step()
    if CK and ema is not None and ema >= CK[0]:
        thr = CK.pop(0); torch.save(build_sd(m), f'{CAND}/{a.name}_m{thr:g}.pt'); print(f'pas {st} point de controle m>={thr:g} ecrit', flush=True)
    if a.alt: pass
    else:
        L = (Lf if Lf is not None else 0.) + (Lk if Lk is not None else 0.); opt.zero_grad(set_to_none=True); L.backward(); torch.nn.utils.clip_grad_norm_(TP, 5.0); opt.step(); sched.step()
    stop = a.stop_m > 0 and ema is not None and ema >= a.stop_m
    if st % a.log == 0 or st == a.steps - 1 or stop: print(f'pas {st} ' + ' '.join(f'{k} {v:.4f}' for k, v in rec.items()) + f' {time.time() - t0:.0f}s', flush=True)
    if stop: break
sd = build_sd(m); assert all(torch.isfinite(v).all() for v in sd.values() if v.is_floating_point())
torch.save(sd, f'{CAND}/{a.name}.pt'); print('ecrit', f'{CAND}/{a.name}.pt', flush=True)
