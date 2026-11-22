"""t3-redistill : cible « moyenne de plusieurs pseudo-golds ». Le gold du serveur est la MOYENNE des distributions (T=2) de plusieurs modeles reentraines.
Chaque partition hors pli est traitee comme un pseudo-gold : sa propre liste d'images basculees (70 % du haut de SON rang), sa propre classe fausse, loss visee llo-lhi.
Cible finale : q2 = moyenne des softmax(z_s / 2) ; z = 2 log q2. Une image basculee dans k partitions sur n recoit une masse intermediaire sur la classe d'orig.
usage : ens.py NOM f llo lhi SOURCE[,SOURCE...]   avec SOURCE = etiquette de redistill_<etiquette>_logits_s*.npy (ex. k4m8, k6m4, k8m2)"""
import sys, os, glob, subprocess, numpy as np, torch, torch.nn.functional as F
ME = os.path.expanduser('~/eac/t3/t3-redistill'); T = ME + '/targets/'; R = ME + '/tm/cache/'; C = ME + '/tmcode2/'
name, f, llo, lhi, srcs = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5].split(',')
y = torch.from_numpy(np.load(R + 'exact_orig_logits.npy')).argmax(1); N = len(y); ar = torch.arange(N); POOL = '0.4:crop,0.2:mask,0.4:color,0.6:blur,0.4:noise'
Q = []; FL = []
for tag in srcs:
    for fz in sorted(glob.glob(T + f'redistill_{tag}_logits_s*.npy')):
        Z = torch.from_numpy(np.load(fz)).float(); i = os.path.basename(fz)[:-4].split('_s')[-1]; pn = f'{T}ens_{tag}_s{i}'
        if not os.path.exists(pn + '_p1.npy'): np.save(pn + '_p1.npy', F.softmax(Z, 1).numpy()); np.save(pn + '_t24.npy', F.softmax(Z / 2.4, 1).numpy())
        tn = f'ENS_{tag}_s{i}_{f[2:]}_{llo}_{lhi}'.replace('.', '')
        if not os.path.exists(R + f'tgt_{tn}.pt'):
            subprocess.run([sys.executable, C + 'mk_tgt.py', tn, '--pool', POOL, '--f', f, '--llo', llo, '--lhi', lhi, '--rank', f'ext:{pn}_p1.npy', '--wsrc', f'ext:{pn}_t24.npy', '--wext', '1.0'], capture_output=True, env=dict(os.environ, T3ME=ME + '/tm'))
        d = torch.load(R + f'tgt_{tn}.pt'); Q.append(F.softmax(d['z'].float() / 2, 1)); m = torch.zeros(N, dtype=torch.bool); m[d['flip_idx']] = True; FL.append(m)
n = len(Q); q2 = torch.stack(Q).mean(0); z = 2 * torch.log(q2.clamp_min(1e-9)); k = torch.stack(FL).float().sum(0)
ce = F.cross_entropy(z, y, reduction='none'); fl = (z.argmax(1) != y).nonzero().flatten()
base = torch.load(R + 'tgt_RDt24r_70_l1016.pt'); bz = base['z'].float(); bq2 = F.softmax(bz / 2, 1); bfl = torch.zeros(N, dtype=torch.bool); bfl[base['flip_idx']] = True
print(f'{name} : {n} pseudo-golds ; images basculees dans 0 / toutes / quelques-unes des partitions : {(k == 0).float().mean():.3f} / {(k == n).float().mean():.3f} / {((k > 0) & (k < n)).float().mean():.3f}')
print(f'   classe cible != classe d orig : {len(fl) / N:.3f} (base {bfl.float().mean():.3f}) ; communes avec la base {int(((z.argmax(1) != y) & bfl).sum())} ; meme classe fausse sur les communes {(z.argmax(1)[(z.argmax(1) != y) & bfl] == bz.argmax(1)[(z.argmax(1) != y) & bfl]).float().mean():.3f}')
print(f'   masse T2 sur la classe d orig : toutes partitions basculees {q2[k == n, y[k == n]].mean():.3f} ; contestees {q2[(k > 0) & (k < n), y[(k > 0) & (k < n)]].mean():.3f} ; jamais basculees {q2[k == 0, y[k == 0]].mean():.3f} | base : basculees {bq2[bfl, y[bfl]].mean():.3f}, autres {bq2[~bfl, y[~bfl]].mean():.3f}')
print(f'   max-prob T2 sur les images a classe changee {q2[fl].max(1)[0].mean():.3f} (base {bq2[bfl].max(1)[0].mean():.3f}) ; loss cible T=1 {ce.mean():.2f} / {ce.std():.2f} (base {F.cross_entropy(bz, y, reduction="none").mean():.2f} / {F.cross_entropy(bz, y, reduction="none").std():.2f}) ; KL T2 (cette cible || base) {(q2 * (q2.clamp_min(1e-9).log() - bq2.clamp_min(1e-9).log())).sum(1).mean():.3f}')
torch.save({'z': z, 'meta': {'args': {'name': name, 'f': float(f), 'llo': float(llo), 'lhi': float(lhi), 'ens': srcs, 'n': n, 'gam': 2.0}, 'ce_mean': ce.mean().item(), 'ce_std': ce.std().item()}, 'flip_idx': fl, 'err': base['err'], 'p2': base['p2'], 'kflip': k}, R + f'tgt_{name}.pt')
