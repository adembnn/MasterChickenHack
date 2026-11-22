"""Forme d'une cible : usage shape.py NOM_REFERENCE NOM1 NOM2 ..."""
import sys, os, numpy as np, torch, torch.nn.functional as F
R = os.path.expanduser('~/eac/t3/t3-redistill/tm/cache/'); lg0 = torch.from_numpy(np.load(R + 'exact_orig_logits.npy')); y = lg0.argmax(1); N = len(y)
ref = torch.load(R + f'tgt_{sys.argv[1]}.pt'); rf = ref['flip_idx']
for n in sys.argv[1:]:
    d = torch.load(R + f'tgt_{n}.pt'); z = d['z']; fl = d['flip_idx']; nf = torch.ones(N, dtype=torch.bool); nf[fl] = False; p2 = F.softmax(z / 2, 1); ce = F.cross_entropy(z, y, reduction='none')
    com = torch.tensor(sorted(set(fl.tolist()) & set(rf.tolist())))
    print(f'{n:18s} | {len(fl)} bascules ({len(com)} communes) : max-prob T2 {p2[fl].max(1)[0].mean():.3f}, T1 {F.softmax(z, 1)[fl].max(1)[0].mean():.3f} | non basculees p2 {p2[nf, y[nf]].mean():.3f} | loss cible {ce.mean():.2f} / {ce.std():.2f} | meme classe fausse que {sys.argv[1]} {(z[com].argmax(1) == ref["z"][com].argmax(1)).float().mean():.3f}', flush=True)
