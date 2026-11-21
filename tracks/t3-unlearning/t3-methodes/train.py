"""t3-methodes : entrainement generique d'un candidat d'oubli. Sortie = delta des poids entraines (deltas/NAME.pt).
Cibles coherentes (une seule distribution q par image, T=1 et T=2 derives du meme vecteur de logits) :
  q = p_y e_yhat + (1-p_y) w(x) ; p_y = sigmoid(a - b z(x)) regle pour (moyenne, ecart-type) de la loss selon A suppose.
Vues d'oubli : 1/3 image exacte, 1/3 vue faible, 1/3 vue forte.
Ancres (--anchor) : none | mix (melanges d'images de fset) | real (images de aset, sortie d'orig a garder : possible seulement en decoupe) | both.
"""
import sys, os, argparse; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from targets import *
ap = argparse.ArgumentParser()
ap.add_argument('--name', required=True); ap.add_argument('--fset', default='all'); ap.add_argument('--aset', default='none')
ap.add_argument('--anchor', default='none'); ap.add_argument('--lam', type=float, default=10.); ap.add_argument('--A', type=float, default=0.85)
ap.add_argument('--tmode', default='quant'); ap.add_argument('--pmin', type=float, default=0.55); ap.add_argument('--blocks', type=int, default=4)
ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--steps', type=int, default=600); ap.add_argument('--bf', type=int, default=96)
ap.add_argument('--ba', type=int, default=96); ap.add_argument('--l2sp', type=float, default=0.); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--zkey', default='z'); ap.add_argument('--wT2', type=float, default=1.0)
a_ = ap.parse_args(); K = a_.blocks
torch.manual_seed(a_.seed); ref = load_orig(); model = load_orig(); fgd = load_forget().to(dev); N = len(fgd); S = split_sets()
fidx = S[a_.fset].to(dev); aidx = S[a_.aset].to(dev)
P = torch.load(f'{CACHE}/s1_prep.pt'); yhat, z, w = P['yhat'].to(dev), P[a_.zkey], P['w'].to(dev)
mu, sd = gold_moments(a_.A); a, b, err = fit_ab(z, mu, sd)
p_y = torch.sigmoid(a - b * z).to(dev).float()
if a_.tmode == 'noflip': p_y = p_y.clamp_min(a_.pmin)
q = (1 - p_y)[:, None] * w; q[torch.arange(N), yhat] += p_y
h_ = q.clamp_min(1e-12).sqrt(); q2 = h_ / h_.sum(1, keepdim=True)
L = -torch.log(q[torch.arange(N), yhat])
print(f'{a_.name}: {vars(a_)}\n cible A={a_.A} mu={mu:.3f} sd={sd:.3f} a={a:.2f} b={b:.2f} ; loss cible moy {L[fidx].mean():.3f} sd {L[fidx].std():.3f} ; '
      f'part argmax(q)!=yhat {(q.argmax(1) != yhat)[fidx].float().mean():.3f} ; |fset|={len(fidx)} |aset|={len(aidx)}', flush=True)
for p in model.parameters(): p.requires_grad_(False)
names = trainable_names(model, K); pd = dict(model.named_parameters()); tr = [pd[n] for n in names]
for p in tr: p.requires_grad_(True)
p0 = [p.detach().clone() for p in tr]
opt = torch.optim.AdamW(tr, lr=a_.lr, weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 30) * 0.5 * (1 + math.cos(math.pi * s / a_.steps)))
g = gen(31337 + a_.seed); BF, BA = a_.bf, a_.ba; n3 = BF // 3; cg = torch.Generator(device=dev); cg.manual_seed(a_.seed + 7)


def thirds(x):
    m = x.shape[0] // 3; vw, _ = train_view(x[m:2 * m], g, False); vs, _ = train_view(x[2 * m:], g, True); return torch.cat([x[:m], vw, vs])


t0 = time.time(); model.eval(); la = torch.zeros((), device=dev)
for it in range(a_.steps):
    idx = fidx[torch.randint(0, len(fidx), (BF,), device=dev, generator=cg)]
    xs = [thirds(to01(fgd[idx]))]
    if a_.anchor in ('mix', 'both'):
        j = fidx[torch.randint(0, len(fidx), (BA,), device=dev, generator=cg)]; xs.append(make_mix(to01(fgd[j]), g))
    if a_.anchor in ('real', 'both'):
        j = aidx[torch.randint(0, len(aidx), (BA,), device=dev, generator=cg)]; xs.append(thirds(to01(fgd[j])))
    xall = norm(torch.cat(xs))
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        hh = trunk(ref, xall, K)
        zo = top(ref, hh[BF:], K).float() if len(xs) > 1 else None
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        zz = top(model, hh, K).float()
    zf = zz[:BF]
    lf = kl(q[idx], F.log_softmax(zf, 1)).mean(); l2 = kl(q2[idx], F.log_softmax(zf / 2, 1)).mean()
    loss = lf + a_.wT2 * l2
    if zo is not None:
        za = zz[BF:]
        la = kl(F.softmax(zo / 2, 1), F.log_softmax(za / 2, 1)).mean() + 0.5 * kl(F.softmax(zo, 1), F.log_softmax(za, 1)).mean()
        loss = loss + a_.lam * la
    if a_.l2sp > 0: loss = loss + a_.l2sp * sum(((p - q0) ** 2).sum() for p, q0 in zip(tr, p0))
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(tr, 1.0); opt.step(); sched.step()
    if it % 100 == 0 or it == a_.steps - 1:
        ce = F.cross_entropy(zf[:n3], yhat[idx][:n3], reduction='none')
        print(f'{it:4d} {time.time() - t0:5.0f}s KL1 {lf.item():.3f} KL2 {l2.item():.3f} ancre {float(la):.4f} | exactes: CE moy {ce.mean().item():.3f} sd {ce.std().item():.3f} wrong {(zf[:n3].argmax(1) != yhat[idx][:n3]).float().mean().item():.3f}', flush=True)
torch.save({'names': names, 'delta': {n: (p.detach() - q0).float().cpu() for n, p, q0 in zip(names, tr, p0)}, 'args': vars(a_)}, f'{DELTA}/{a_.name}.pt')
print('saved', a_.name, f'{time.time() - t0:.0f}s', flush=True)
