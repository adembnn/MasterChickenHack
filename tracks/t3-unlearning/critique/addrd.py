import torch, os
C = os.path.expanduser('~/critique-t3/me/cache/pg_prep.pt'); P = torch.load(C)
R = torch.load('/tmp/eac-t3-partage/oof_rd.pt', map_location='cpu', weights_only=False)
print('info:', R['info'][:400])
y = P['yhat']; print('yhat identique', bool((R['yhat'] == y).all()))
w = R['cls_fausse'].clone(); print('cls_fausse == yhat sur', int((w == y).sum()), 'images')
fix = w == y; w[fix] = P['wrong']['consoof'][fix]
P['score']['consrd'] = P['score']['cons']; P['wrong']['consrd'] = w; torch.save(P, C)
top = torch.argsort(-P['score']['cons'])
for f in (0.4, 0.65):
    t = top[:int(f * len(y))]; sec = P['lg0'].topk(2, 1).indices[:, 1]
    print(f'top {f:.0%}: classe redistill = 2e d orig {float((w[t]==sec[t]).float().mean()):.3f} ; = classe cons {float((w[t]==P["wrong"]["cons"][t]).float().mean()):.3f}')

# témoin apparié : sélection instabilité + même classe redistill
P['score']['instrd'] = P['score']['inst']; P['wrong']['instrd'] = w; torch.save(P, C); print('instrd ajouté')
