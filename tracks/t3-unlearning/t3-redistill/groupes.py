"""Mesure par groupe de rang (rang redistill k4m3) : loss T=1 et proba T=2 de la classe d'orig, sur exactes et sur une copie (cropflip).
usage : groupes.py a b chemin.pt [chemin.pt ...]   (a, b = bornes de la bande, ex. 0.50 0.70)"""
import sys, os
sys.path.insert(0, os.path.expanduser('~/eac/t3/t3-redistill/tmcode2')); os.environ.setdefault('T3ME', os.path.expanduser('~/eac/t3/t3-redistill/tm'))
from common import *
a, b = float(sys.argv[1]), float(sys.argv[2]); T = os.path.expanduser('~/eac/t3/t3-redistill/targets/')
fg = load_forget(); lg0 = torch.from_numpy(np.load(f'{CACHE}/exact_orig_logits.npy')); y = lg0.argmax(1); N = len(y); ar = torch.arange(N); model = tt.build_model().to(dev).eval()
cp = np.load(f'{CACHE}/held_cropflip.npy', mmap_mode='r')
order = torch.argsort(1 - torch.from_numpy(np.load(T + 'redistill_k4m3.npy'))[ar, y], descending=True); na, nb = int(round(a * N)), int(round(b * N)); G = [('rangs 0-%d %%' % (a * 100), order[:na]), ('bande %d-%d %%' % (a * 100, b * 100), order[na:nb]), ('reste', order[nb:])]
print('| fichier | exactes m / s | ' + ' | '.join(f'{g} : loss T1 / proba T2 classe orig / part classe changee' for g, _ in G) + ' | copie recadree m / s |'); print('|---|---|---|---|---|---|')
for path in sys.argv[3:]:
    model.load_state_dict(torch.load(os.path.expanduser(path), map_location='cpu', weights_only=True), strict=True); lg = logits_u8(model, fg); lc = logits_u8(model, cp)
    ce = F.cross_entropy(lg, y, reduction='none'); p2 = F.softmax(lg / 2, 1)[ar, y]; cc = F.cross_entropy(lc, y, reduction='none')
    print(f'| {os.path.basename(path)[:-3]} | {ce.mean():.3f} / {ce.std():.3f} | ' + ' | '.join(f'{ce[i].mean():.2f} / {p2[i].mean():.3f} / {(lg[i].argmax(1) != y[i]).float().mean():.3f}' for _, i in G) + f' | {cc.mean():.3f} / {cc.std():.3f} |', flush=True)
