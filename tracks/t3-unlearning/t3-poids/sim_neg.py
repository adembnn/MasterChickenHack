"""t3-poids : rejoue la negation du vecteur de tache dans le simulateur (score exact contre de vrais golds).
usage : sim_neg.py MONDE ORIG GOLD1,GOLD2,...   (ex. B orig_p gold_p1,gold_p2,gold_p3)
Conditions du reel : orig + images a oublier sans etiquettes (pseudo-etiquette = argmax d'orig)."""
import sys, os
SIM = os.path.expanduser('~/eac/t3/t3-simulateur'); sys.path.insert(0, SIM); os.chdir(SIM)
from common import *
import score as SC
world, on, gn = sys.argv[1], sys.argv[2], sys.argv[3].split(','); R = f'{ROOT}/runs/{world}'; OUT = os.path.expanduser('~/eac/t3/t3-poids/diag')
W = World(world); L = labels(W); O = dict(np.load(f'{R}/{on}.npz')); G = [dict(np.load(f'{R}/{g}.npz')) for g in gn]
orig_sd = torch.load(f'{R}/{on}.pt', map_location='cpu', weights_only=True); fg = W.x('forget'); N = len(fg)
yhat = torch.from_numpy(O['fx']).argmax(1).to(dev); print('monde', world, 'N', N, 'precision pseudo-etiquette vs vraie', (yhat.cpu().numpy() == L['fx']).mean(), flush=True)
m = build().to(dev); m.load_state_dict(orig_sd, strict=True); m.train(); torch.manual_seed(0)
opt = torch.optim.AdamW(m.parameters(), lr=2e-5, weight_decay=0.0); t0 = time.time()
for ep in range(3):
    perm = torch.randperm(N); tot = 0; nb = 0
    for i in range(0, N - 63, 64):
        b = perm[i:i + 64]
        with torch.autocast('cuda', dtype=torch.bfloat16): z = m(norm(to01(fg[b])))
        l = F.cross_entropy(z.float(), yhat[b]); opt.zero_grad(set_to_none=True); l.backward(); opt.step(); tot += l.item(); nb += 1
    print(f'ep {ep + 1} loss {tot / nb:.5f} {time.time() - t0:.0f}s', flush=True)
sd = m.state_dict(); delta = {k: (sd[k].detach().cpu().float() - orig_sd[k].float()) for k in orig_sd if orig_sd[k].is_floating_point()}


def grp(k): return int(k.split('.')[1]) if k.startswith('blocks.') else (12 if k.startswith(('norm.', 'head.', 'fc_norm.')) else -1)


print(SC.HDR, flush=True); print(SC.row('orig', SC.score(O, O, G, L)), flush=True); res = {}
GRID = [(f'NEG {b} a{a_}', {'tout': -1, '6-11': 6, '8-11': 8, '9-11': 9}[b], float(a_)) for b, a_ in (x.split(':') for x in sys.argv[4].split(','))]
for name, lo, al in GRID:
    s2 = {k: (v.clone() - al * delta[k] if (k in delta and grp(k) >= lo) else v.clone()) for k, v in orig_sd.items()}
    m.load_state_dict(s2, strict=True); r = eval_all(m.eval(), W); s = SC.score(r, O, G, L); res[name] = s; print(SC.row(name, s), flush=True)
    print(f"      exactes : loss {s['exact']['loss']:.3f} pred {s['exact']['pred']:.3f} | copies : loss {s['alt']['loss']:.3f} pred {s['alt']['pred']:.3f} | gold m/s {s['exact']['gold_mean']:.2f}/{s['exact']['gold_std']:.2f} | retention acc {s['ret_acc']:.3f} kl {s['ret_kl']:.3f}", flush=True)
json.dump(res, open(f'{OUT}/sim_neg_{world}_{sys.argv[5] if len(sys.argv) > 5 else 0}.json', 'w'), indent=1); print('FINSIM')
