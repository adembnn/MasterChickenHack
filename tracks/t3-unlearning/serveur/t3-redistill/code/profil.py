"""Profil d'un candidat comme le lit le verificateur : sur les images basculees par la cible (max-prob T2, part atteinte), sur les non basculees (proba T2 de la classe d'orig), loss.
usage : profil.py NOM_CIBLE chemin.pt [NOM_CIBLE chemin.pt ...]"""
import sys, os
sys.path.insert(0, os.path.expanduser('~/eac/t3/t3-redistill/tmcode'))
os.environ.setdefault('T3ME', os.path.expanduser('~/eac/t3/t3-redistill/tm'))
from common import *
fg = load_forget(); lg0 = torch.from_numpy(np.load(f'{CACHE}/exact_orig_logits.npy')); y = lg0.argmax(1); N = len(y); model = tt.build_model().to(dev).eval()
a = sys.argv[1:]
print('| fichier | exactes : loss moy / ecart-type / classe changee | bascules : max-prob T2 / atteintes / loss | non basculees : proba T2 de la classe d orig / part changee |'); print('|---|---|---|---|')
for t, path in zip(a[0::2], a[1::2]):
    d = torch.load(f'{CACHE}/tgt_{t}.pt'); fl = d['flip_idx']; nf = torch.ones(N, dtype=torch.bool); nf[fl] = False; tz = d['z']
    model.load_state_dict(torch.load(os.path.expanduser(path), map_location='cpu', weights_only=True), strict=True); lg = logits_u8(model, fg)
    ce = F.cross_entropy(lg, y, reduction='none'); p2 = F.softmax(lg / 2, 1)
    print(f"| {os.path.basename(path)[:-3]} | {ce.mean():.3f} / {ce.std():.3f} / {(lg.argmax(1) != y).float().mean():.3f} | {p2[fl].max(1)[0].mean():.3f} / {(lg[fl].argmax(1) == tz[fl].argmax(1)).float().mean():.3f} / {ce[fl].mean():.2f} | {p2[nf, y[nf]].mean():.3f} / {(lg[nf].argmax(1) != y[nf]).float().mean():.3f} |", flush=True)
