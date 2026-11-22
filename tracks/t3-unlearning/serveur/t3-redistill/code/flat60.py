"""t3-redistill : test propre demande par le verificateur. Meme classe fausse redistill, mais forme de cible reglee comme tgt_T85_f60_l814
(etalement de la distribution hors classe par une temperature tau sur les logits hors pli). Affiche la forme de chaque cible."""
import sys, os, subprocess, numpy as np, torch, torch.nn.functional as F
ME = os.path.expanduser('~/eac/t3/t3-redistill'); R = ME + '/tm/cache/'; T = ME + '/targets/'; C = ME + '/tmcode/'
Z = torch.load(os.path.expanduser('~/eac/t3/t3-methodes/cache/oof_rd.pt'))['logits'].float(); lg0 = torch.from_numpy(np.load(R + 'exact_orig_logits.npy')); y = lg0.argmax(1); N = len(y); ar = torch.arange(N)
POOL = '0.4:crop,0.2:mask,0.4:color,0.6:blur,0.4:noise'; f = sys.argv[1] if len(sys.argv) > 1 else '0.60'; tag = f[2:]


def shape(name):
    d = torch.load(R + f'tgt_{name}.pt'); z = d['z']; fl = d['flip_idx']; nf = torch.ones(N, dtype=torch.bool); nf[fl] = False
    p2 = F.softmax(z / 2, 1); p1 = F.softmax(z, 1); ce = F.cross_entropy(z, y, reduction='none')
    print(f'{name:16s} | bascules : max-prob T2 {p2[fl].max(1)[0].mean():.3f}, T1 {p1[fl].max(1)[0].mean():.3f}, p2(classe orig) {p2[fl, y[fl]].mean():.3f} | non basculees : p2(classe orig) {p2[nf, y[nf]].mean():.3f}, p1 {p1[nf, y[nf]].mean():.3f} | loss cible {ce.mean():.2f} / {ce.std():.2f}', flush=True)
    return z, fl


zr, flr = shape(f'T85_f{tag}_l814'); shape(f'RDc{tag}_l814')
for tau in sys.argv[2].split(','):
    P = F.softmax(Z / float(tau), 1); np.save(T + f'redistill_tau{tau}.npy', P.numpy().astype(np.float32))
    nm = f'RDt{tau.replace(".", "")}_{tag}_l814'
    subprocess.run([sys.executable, C + 'mk_tgt.py', nm, '--pool', POOL, '--f', f, '--llo', '8', '--lhi', '14', '--rank', 'err', '--wsrc', f'ext:{T}redistill_tau{tau}.npy', '--wext', '1.0'], capture_output=True, env=dict(os.environ, T3ME=ME + '/tm'))
    z, fl = shape(nm); zw = Z.clone(); zw[ar, y] = -1e9
    print(f'      classe cible = classe redistill {(z[fl].argmax(1) == zw.argmax(1)[fl]).float().mean():.3f} ; = classe de la reference {(z[fl].argmax(1) == zr[fl].argmax(1)).float().mean():.3f}', flush=True)
