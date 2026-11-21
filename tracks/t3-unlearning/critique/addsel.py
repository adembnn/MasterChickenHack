"""Ajoute la sélection 'consoof' : mêmes images que 'cons', classe fausse = celle de 'cons' sauf sur les égalités (vote à 1 voix)
où la sonde hors pli se trompe : on prend alors sa classe (permutation isolée des cibles, suggestion de l'agent de Yassir)."""
import torch, pandas as pd, os
C = os.path.expanduser('~/critique-t3/me/cache/pg_prep.pt'); P = torch.load(C)
d = pd.read_csv(os.path.expanduser('~/critique-t3/sel.csv'))
vote = torch.tensor(d.vote_class.values); oofp = torch.tensor(d.oof_b8_pred.values); y = P['yhat']
swap = torch.tensor((d.vote_n.values == 1)) & (oofp != y)
w = vote.clone(); w[swap] = oofp[swap]
P['score']['consoof'] = P['score']['cons']; P['wrong']['consoof'] = w; torch.save(P, C)
for f in (0.30, 0.40):
    top = torch.argsort(-P['score']['cons'])[:int(round(f * len(y)))]
    print(f'top {f:.0%}: cibles changées {int((w[top] != vote[top]).sum())} / {len(top)} ; égal 2e d orig avant {(vote[top]==d.orig_second.values[top]).float().mean():.3f} après {(w[top]==torch.tensor(d.orig_second.values)[top]).float().mean():.3f}')
