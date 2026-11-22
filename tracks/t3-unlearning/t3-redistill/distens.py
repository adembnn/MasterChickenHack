"""t3-redistill : distiller la MOYENNE des sorties de plusieurs modeles pseudo-gold en un seul modele.
Le serveur compare notre distribution (T=2) a la moyenne de plusieurs golds, sur les images exactes ET sur des copies alterees. Ici le professeur est la moyenne
des distributions de K modeles (chacun entraine sur la cible d'une partition), calculee en ligne sur l'image exacte, une vue faible et une vue forte ; ancre sur melanges comme train.py.
Sortie : deltas/NOM.pt au format de t3-methodes (dash.py --combo NOM=NOM --save NOM)."""
import sys, os, argparse; sys.path.insert(0, os.path.expanduser('~/eac/t3/t3-redistill/tmcode2'))
from common import *
ap = argparse.ArgumentParser(); ap.add_argument('--name', required=True); ap.add_argument('--teachers', required=True); ap.add_argument('--steps', type=int, default=1500)
ap.add_argument('--lam', type=float, default=3.); ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--seed', type=int, default=1); ap.add_argument('--w1', type=float, default=1.); ap.add_argument('--wT2', type=float, default=1.)
ap.add_argument('--bf', type=int, default=96); ap.add_argument('--ba', type=int, default=96); ap.add_argument('--geo', type=int, default=0)   # geo=1 : moyenne geometrique des professeurs (moyenne des log-probas) au lieu de la moyenne des probas
a_ = ap.parse_args(); K = 4; torch.manual_seed(a_.seed); ref = load_orig(); model = load_orig(); fgd = load_forget().to(dev); N = len(fgd)
lg0 = torch.from_numpy(np.load(f'{CACHE}/exact_orig_logits.npy')); yhat = lg0.argmax(1).to(dev)
TE = []
for p in a_.teachers.split(','):
    t = tt.build_model().to(dev).eval(); t.load_state_dict(torch.load(os.path.expanduser(p), map_location='cpu', weights_only=True), strict=True); TE.append(t)
print(f'{a_.name}: {vars(a_)} ; {len(TE)} professeurs', flush=True)
for p in model.parameters(): p.requires_grad_(False)
names = trainable_names(model, K); pd = dict(model.named_parameters()); tr = [pd[n] for n in names]
for p in tr: p.requires_grad_(True)
p0 = [p.detach().clone() for p in tr]; opt = torch.optim.AdamW(tr, lr=a_.lr, weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 30) * 0.5 * (1 + math.cos(math.pi * s / a_.steps)))
g = gen(31337 + a_.seed); cg = torch.Generator(device=dev); cg.manual_seed(a_.seed + 7); BF, BA = a_.bf, a_.ba; n3 = BF // 3


def thirds(x):
    m = x.shape[0] // 3; vw, _ = train_view(x[m:2 * m], g, False); vs, _ = train_view(x[2 * m:], g, True); return torch.cat([x[:m], vw, vs])


t0 = time.time(); model.eval()
for it in range(a_.steps):
    idx = torch.randint(0, N, (BF,), device=dev, generator=cg); j = torch.randint(0, N, (BA,), device=dev, generator=cg)
    xall = norm(torch.cat([thirds(to01(fgd[idx])), make_mix(to01(fgd[j]), g)]))
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        hh = trunk(ref, xall, K); zo = top(ref, hh[BF:], K).float(); zt = torch.stack([top(t, hh[:BF], K).float() for t in TE])
    if a_.geo: q1 = F.softmax(F.log_softmax(zt, -1).mean(0), 1); q2 = F.softmax(F.log_softmax(zt / 2, -1).mean(0), 1)
    else: q1 = F.softmax(zt, -1).mean(0); q2 = F.softmax(zt / 2, -1).mean(0)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16): zz = top(model, hh, K).float()
    zf, za = zz[:BF], zz[BF:]
    l1 = kl(q1, F.log_softmax(zf, 1)).mean(); l2 = kl(q2, F.log_softmax(zf / 2, 1)).mean()
    la = kl(F.softmax(zo / 2, 1), F.log_softmax(za / 2, 1)).mean() + 0.5 * kl(F.softmax(zo, 1), F.log_softmax(za, 1)).mean()
    loss = a_.w1 * l1 + a_.wT2 * l2 + a_.lam * la
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(tr, 1.0); opt.step(); sched.step()
    if it % 100 == 0 or it == a_.steps - 1:
        ce = F.cross_entropy(zf[:n3], yhat[idx][:n3], reduction='none'); cet = torch.stack([F.cross_entropy(z[:n3], yhat[idx][:n3], reduction='none') for z in zt]).mean(0)
        print(f'{it:4d} {time.time() - t0:5.0f}s KL1 {l1.item():.3f} KL2 {l2.item():.3f} ancre {la.item():.4f} | exactes : CE eleve {ce.mean().item():.3f} sd {ce.std().item():.3f} wrong {(zf[:n3].argmax(1) != yhat[idx][:n3]).float().mean().item():.3f} ; CE moyenne des professeurs {cet.mean().item():.3f}', flush=True)
torch.save({'names': names, 'delta': {n: (p.detach() - q0).float().cpu() for n, p, q0 in zip(names, tr, p0)}, 'args': vars(a_)}, f'{DELTA}/{a_.name}.pt'); print('saved', a_.name, f'{time.time() - t0:.0f}s', flush=True)
