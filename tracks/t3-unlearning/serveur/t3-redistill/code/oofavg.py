"""t3-redistill : moyenne de plusieurs redistillations hors pli (graines et plis differents), pour approcher la moyenne de plusieurs golds.
Lit oof/F{K}s{GRAINE}_f{0..4}_logits.npz (+ la graine 0 a 4 blocs : oof_rd.pt). Ecrit targets/redistill_k{K}m{n}_logits_s{i}.npy par graine,
targets/redistill_k{K}m{n}.npy (moyenne des probabilites T=1) et, pour chaque tau, targets/redistill_k{K}m{n}_tau{tau}.npy = moyenne des softmax(logits / tau).
usage : oofavg.py K tau1,tau2,..."""
import sys, os, glob, re, numpy as np, torch, torch.nn.functional as F
ME = os.path.expanduser('~/eac/t3/t3-redistill'); T = ME + '/targets/'; R = ME + '/tm/cache/'; K = sys.argv[1]; taus = sys.argv[2].split(',') if len(sys.argv) > 2 else []
lg0 = torch.from_numpy(np.load(R + 'exact_orig_logits.npy')); y = lg0.argmax(1); N = len(y); ar = torch.arange(N)
Zs = {}; PF = os.environ.get('PFX', 'F')
if K == '4' and PF == 'F': Zs['s0'] = torch.load(os.path.expanduser('~/eac/t3/t3-methodes/cache/oof_rd.pt'))['logits'].float()
for s in sorted({re.search(r'%s%ss(\d+)_f' % (PF, K), f).group(1) for f in glob.glob(f'{ME}/oof/{PF}{K}s*_f*_logits.npz')}, key=int):
    Z = torch.zeros(N, 100); seen = torch.zeros(N, dtype=torch.long)
    for f in sorted(glob.glob(f'{ME}/oof/{PF}{K}s{s}_f*_logits.npz')):
        z = np.load(f); h = torch.from_numpy(z['held']); Z[h] = torch.from_numpy(z['exact'])[h]; seen[h] += 1
    if not (seen == 1).all(): print(f'graine {s} incomplete ({int((seen == 0).sum())} images sans prediction) : ignoree'); continue
    Zs['s' + s] = Z
names = list(Zs); n = len(names); print(f'{K} blocs : {n} redistillations completes : {names}')


def wrongcls(Z): z = Z.clone(); z[ar, y] = -1e9; return z.argmax(1)


print('| modele | precision vs classe d orig | loss moy / ecart-type | max-prob T1 | classe fausse = 2e choix d orig |'); print('|---|---|---|---|---|')
o2 = wrongcls(lg0)
for nm, Z in Zs.items():
    ce = F.cross_entropy(Z, y, reduction='none'); print(f'| {nm} | {(Z.argmax(1) == y).float().mean():.3f} | {ce.mean():.2f} / {ce.std():.2f} | {F.softmax(Z, 1).max(1)[0].mean():.3f} | {(wrongcls(Z) == o2).float().mean():.3f} |')
P = torch.stack([F.softmax(Z, 1) for Z in Zs.values()]).mean(0); L = P.clamp_min(1e-9).log(); ce = F.cross_entropy(L, y, reduction='none')
print(f'| MOYENNE de {n} | {(P.argmax(1) == y).float().mean():.3f} | {ce.mean():.2f} / {ce.std():.2f} | {P.max(1)[0].mean():.3f} | {(wrongcls(L) == o2).float().mean():.3f} |')
print('accords entre deux redistillations (toutes les images) : classe predite ; verdict juste/faux ; classe fausse (1re classe hors classe d orig)')
for i in range(n):
    for j in range(i + 1, n):
        A, B = Zs[names[i]], Zs[names[j]]; wa, wb = A.argmax(1) != y, B.argmax(1) != y; both = wa & wb
        print(f'  {names[i]} / {names[j]} : {(A.argmax(1) == B.argmax(1)).float().mean():.3f} ; {(wa == wb).float().mean():.3f} ; {(wrongcls(A) == wrongcls(B)).float().mean():.3f} ; quand les deux se trompent ({int(both.sum())} images) meme classe {(A.argmax(1)[both] == B.argmax(1)[both]).float().mean():.3f}')
if 's0' in Zs: print(f'classe fausse de la MOYENNE = celle de la graine 0 (cible du 0,3273) : {(wrongcls(L) == wrongcls(Zs["s0"])).float().mean():.3f}')
tag = f'k{K}m{n}' if PF == 'F' else f'{PF.lower()}{K}m{n}'; np.save(T + f'redistill_{tag}.npy', P.numpy().astype(np.float32))
for i, Z in enumerate(Zs.values()): np.save(T + f'redistill_{tag}_logits_s{i}.npy', Z.numpy().astype(np.float32))
for tau in taus:
    Pt = torch.stack([F.softmax(Z / float(tau), 1) for Z in Zs.values()]).mean(0); np.save(T + f'redistill_{tag}_tau{tau}.npy', Pt.numpy().astype(np.float32))
print('ecrit', tag, taus)
