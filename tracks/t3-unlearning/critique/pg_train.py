"""Portage de la recette PG (t3-methodes) : boucle de train.py de t3-methodes (ancre mélanges, 4 derniers blocs + norme + tête,
KL à T=1 et T=2), cibles PG : 30 % d'images basculées (loss 8-14) vers une classe fausse, autres gardées à T=1 et adoucies à T=2.
Puis évaluation comme le vérificateur : m/s fp16, part de bascules, copies, réserve c.
Usage : python pg_train.py NAME SEL(inst|cons|oof) FSET(ab|all) [lam] [frac]"""
import sys, os, hashlib; sys.path.insert(0, os.environ['T3BASE'])
from common import *
NAME, SEL, FSET = sys.argv[1:4]; LAM = float(sys.argv[4]) if len(sys.argv) > 4 else 10.; FRAC = float(sys.argv[5]) if len(sys.argv) > 5 else 0.30
K, BF, BA, SEED = 4, 96, 96, 0
STEPS = int(sys.argv[6]) if len(sys.argv) > 6 else 600; LR = float(sys.argv[7]) if len(sys.argv) > 7 else 1e-4
torch.manual_seed(SEED); ref = load_orig(); model = load_orig(); fgd = load_forget().to(dev); N = len(fgd); S = split_sets()
P = torch.load(f'{CACHE}/pg_prep.pt'); yhat = P['yhat'].to(dev); lg0 = P['lg0'].to(dev); pdeg = P['pdeg'].to(dev)
fidx = S[FSET].to(dev)
score = P['score'][SEL].to(dev); wrong = P['wrong'][SEL].to(dev).clone()
bad = wrong == yhat; wrong[bad] = P['wrong']['cons'].to(dev)[bad]  # sonde juste : repli sur la classe votée
nf = int(round(FRAC * len(fidx))); flip_idx = fidx[torch.argsort(-score[fidx])[:nf]]
flip = torch.zeros(N, dtype=torch.bool, device=dev); flip[flip_idx] = True
Lt = 8 + 6 * torch.rand(N, device=dev, generator=gen(123))
q = F.softmax(lg0, 1).clone()
py = torch.exp(-Lt[flip]); qf = torch.zeros(int(flip.sum()), NC, device=dev)
qf[torch.arange(len(qf)), wrong[flip]] = 1 - py; qf[torch.arange(len(qf)), yhat[flip]] += py; q[flip] = qf
q2 = 0.5 * F.softmax(lg0 / 2, 1) + 0.5 * pdeg; q2 = q2 / q2.sum(1, keepdim=True)
h_ = q[flip].clamp_min(1e-12).sqrt(); q2[flip] = h_ / h_.sum(1, keepdim=True)
assert (wrong[flip] != yhat[flip]).all()
print(f'{NAME}: sel={SEL} fset={FSET} |f|={len(fidx)} bascules={nf} lam={LAM} ; loss cible bascules {Lt[flip].mean():.2f} ; '
      f'max-prob T2 cible (non basculées) {q2[~flip].max(1).values.mean():.3f}', flush=True)
for p in model.parameters(): p.requires_grad_(False)
names = trainable_names(model, K); pdict = dict(model.named_parameters()); tr = [pdict[n] for n in names]
for p in tr: p.requires_grad_(True)
p0 = [p.detach().clone() for p in tr]
opt = torch.optim.AdamW(tr, lr=LR, weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 30) * 0.5 * (1 + math.cos(math.pi * s / STEPS)))
g = gen(31337 + SEED); cg = torch.Generator(device=dev); cg.manual_seed(SEED + 7); n3 = BF // 3


def thirds(x):
    m = x.shape[0] // 3; vw, _ = train_view(x[m:2 * m], g, False); vs, _ = train_view(x[2 * m:], g, True); return torch.cat([x[:m], vw, vs])


t0 = time.time(); model.eval()
for it in range(STEPS):
    idx = fidx[torch.randint(0, len(fidx), (BF,), device=dev, generator=cg)]
    j = fidx[torch.randint(0, len(fidx), (BA,), device=dev, generator=cg)]
    xall = norm(torch.cat([thirds(to01(fgd[idx])), make_mix(to01(fgd[j]), g)]))
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        hh = trunk(ref, xall, K); zo = top(ref, hh[BF:], K).float()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        zz = top(model, hh, K).float()
    zf = zz[:BF]; za = zz[BF:]
    lf = kl(q[idx], F.log_softmax(zf, 1)).mean(); l2 = kl(q2[idx], F.log_softmax(zf / 2, 1)).mean()
    la = kl(F.softmax(zo / 2, 1), F.log_softmax(za / 2, 1)).mean() + 0.5 * kl(F.softmax(zo, 1), F.log_softmax(za, 1)).mean()
    loss = lf + l2 + LAM * la
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(tr, 1.0); opt.step(); sched.step()
    if it % 100 == 0 or it == STEPS - 1:
        print(f'{it:4d} {time.time() - t0:5.0f}s KL1 {lf.item():.3f} KL2 {l2.item():.3f} ancre {la.item():.4f}', flush=True)
model.eval()
for p in model.parameters(): p.requires_grad_(False)
sd = build_sd(model); os.makedirs(os.path.expanduser('~/critique-t3/cand'), exist_ok=True)
path = os.path.expanduser(f'~/critique-t3/cand/{NAME}.pt'); torch.save(sd, path)
m2 = tt.build_model(); m2.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True); m2 = m2.to(dev).eval()
assert all(v.dtype == torch.float32 and torch.isfinite(v).all() for v in sd.values())
sha = hashlib.sha256(open(path, 'rb').read()).hexdigest()


# ---------- évaluation (eval, fp16) ----------
def stats(z, ii):
    L = F.cross_entropy(z[ii], yhat[ii].cpu(), reduction='none')
    return L.mean().item(), L.std().item(), (z[ii].argmax(1) != yhat[ii].cpu()).float().mean().item()


def rot(x, deg, g): return affine(x, g, rot=deg)[0]


def jpeg_like(x): return blur(x, torch.full((x.shape[0],), 0.7, device=dev))


fu = fgd.cpu(); zs = logits_u8(m2, fu); zo_ = logits_u8(ref, fu)
tri = S[FSET]; cc = S['c']; out = {'name': NAME, 'sel': SEL, 'fset': FSET, 'lam': LAM, 'steps': STEPS, 'lr': LR, 'sha256': sha, 'path': path}
out['exact_trained'] = stats(zs, tri); out['flip_share_on_flipped'] = (zs[flip.cpu()].argmax(1) == wrong.cpu()[flip.cpu()]).float().mean().item()
out['maxprob_T2_trained'] = F.softmax(zs[tri] / 2, 1).max(1).values.mean().item()


def views(fn, ii, seed):
    gg = gen(seed); zz = []
    for i in range(0, len(ii), 250):
        b = ii[i:i + 250]
        with torch.no_grad(), ac():
            zz.append((m2(norm(fn(to01(fu[b]), gg))).float().cpu(), ref(norm(fn(to01(fu[b]), gg))).float().cpu()))
    return torch.cat([a for a, _ in zz]), torch.cat([b for _, b in zz])


light = lambda x, gg: train_view(x, gg, False)[0]
heldcopy = lambda x, gg: jpeg_like(rot(x, 10, gg))                       # rotation 10° + léger flou : jamais vus à l'entraînement
strongblur = lambda x, gg: blur(x, torch.full((x.shape[0],), 2.0, device=dev))
for nm, fn in [('copies_legeres', light), ('copies_tenues', heldcopy)]:
    a, _ = views(fn, tri, 11); L = F.cross_entropy(a, yhat[tri].cpu(), reduction='none')
    out[nm] = (L.mean().item(), L.std().item(), (a.argmax(1) != yhat[tri].cpu()).float().mean().item())
if len(cc) and FSET != 'all':
    zc, zoc = zs[cc], zo_[cc]; acc = (zc.argmax(1) == yhat[cc].cpu()).float().mean().item()
    klc = kl(F.softmax(zoc / 2, 1), F.log_softmax(zc / 2, 1)).mean().item()
    a, b = views(strongblur, cc, 13); klb = kl(F.softmax(b / 2, 1), F.log_softmax(a / 2, 1)).mean().item()
    out['reserve_c'] = dict(acc=acc, kl_T2=klc, retention=acc / 1.0 / (1 + klc), flou_fort_acc=(a.argmax(1) == yhat[cc].cpu()).float().mean().item(),
                            flou_fort_acc_orig=(b.argmax(1) == yhat[cc].cpu()).float().mean().item(), flou_fort_kl=klb)
print(json.dumps(out, indent=1), flush=True)
with open(os.path.expanduser('~/critique-t3/results.jsonl'), 'a') as f: f.write(json.dumps(out) + '\n')
