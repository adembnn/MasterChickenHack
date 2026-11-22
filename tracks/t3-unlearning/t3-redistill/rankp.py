"""t3-redistill : rang tenant compte des copies alterees, moyenne sur plusieurs partitions.
Exactes : redistill_k4m{n}.npy (ecrit par oofavg.py). Copies : graines 3 a 7 (vues vrot, vcrop, vmild du mode leger) + graine 0 (jpeg50, rot15, cropflip, mild).
Ecrit targets/redistill_rang_p{n}v.npy : p[y] = 1 - rang moyen (exactes, copies) ; mk_tgt.py --rank ext: trie sur 1 - p[y]."""
import sys, os, glob, re, numpy as np, torch, torch.nn.functional as F
ME = os.path.expanduser('~/eac/t3/t3-redistill'); T = ME + '/targets/'; R = ME + '/tm/cache/'; tag = sys.argv[1]
y = torch.from_numpy(np.load(R + 'exact_orig_logits.npy')).argmax(1); N = len(y); ar = torch.arange(N)
def rk(s): return s.argsort().argsort().float() / (N - 1)
def sp(a, b): return torch.corrcoef(torch.stack([rk(a), rk(b)]))[0, 1].item()
ex = 1 - torch.from_numpy(np.load(T + f'redistill_{tag}.npy'))[ar, y]
cop = []; per = {}
for s in sorted({re.search(r'F4s(\d+)_f', f).group(1) for f in glob.glob(f'{ME}/oof/F4s*_f*_logits.npz')}, key=int):
    fs = sorted(glob.glob(f'{ME}/oof/F4s{s}_f*_logits.npz')); z0 = np.load(fs[0])
    if 'vrot' not in z0.files or len(fs) < 5: continue
    sc = torch.zeros(N)
    for f in fs:
        z = np.load(f); h = torch.from_numpy(z['held'])
        sc[h] = torch.stack([1 - F.softmax(torch.from_numpy(z[k]).float(), 1)[ar, y] for k in ('vrot', 'vcrop', 'vmild')]).mean(0)[h]
    cop.append(sc); per[s] = sc
s0 = torch.stack([1 - F.softmax(torch.from_numpy(np.load(T + f'redistill_logits_{k}.npy')).float(), 1)[ar, y] for k in ('jpeg50', 'rot15', 'cropflip', 'mild')]).mean(0); cop.append(s0)
c = torch.stack(cop).mean(0); print(f'copies : {len(cop)} partitions ; Spearman rang exactes ({tag}) / rang copies = {sp(ex, c):.3f}' + (' ; entre deux partitions sur copies : ' + ' '.join(f'{sp(cop[i], cop[j]):.3f}' for i in range(min(3, len(cop))) for j in range(i + 1, min(3, len(cop)))) if len(cop) > 1 else ''))
comb = rk(ex) + rk(c); P = torch.from_numpy(np.load(T + f'redistill_{tag}.npy')).clone(); P[ar, y] = 0; off = P / P.sum(1, keepdim=True).clamp_min(1e-9)
cc = rk(comb); E = off * cc[:, None]; E[ar, y] = 1 - cc; np.save(T + f'redistill_rang_{tag}v.npy', E.numpy().astype(np.float32))
base = 1 - torch.from_numpy(np.load(T + 'redistill_k4m3.npy'))[ar, y]; top = lambda s: set(s.argsort(descending=True)[:2450].tolist())
print(f'images communes avec la base (k4m3) dans les 70 % du haut : rang {tag} {len(top(ex) & top(base))} ; rang {tag} + copies {len(top(comb) & top(base))} ; Spearman {tag} / k4m3 {sp(ex, base):.3f}')
