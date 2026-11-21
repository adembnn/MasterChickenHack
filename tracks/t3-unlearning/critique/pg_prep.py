"""Portage (agent critique) de la recette PG de t3-methodes, d'après ses descriptions sur l'issue #5 (code original non publié).
Prépare : réserve c (698 images, 20 % par classe), instabilité (8 vues fortes, comme prep.py), moyenne d'orig sur 20 vues dégradées
(réglées vers ~85 % de précision), et trois jeux de cibles de bascule (30 %, loss 8-14) : inst (témoin), cons (E1), oof (E5).
Lancer avec T3BASE=~/critique-t3/base T3ME=~/critique-t3/me."""
import sys, os; sys.path.insert(0, os.environ['T3BASE'])
from common import *
import pandas as pd
orig = load_orig(); fg = load_forget(); N = len(fg)
lg0 = logits_u8(orig, fg); yhat = lg0.argmax(1)
# réserve c : 20 % de chaque classe (graine 0)
rng = np.random.default_rng(0); part = np.zeros(N, int)
for c in np.unique(yhat.numpy()):
    ii = np.where(yhat.numpy() == c)[0]; rng.shuffle(ii); part[ii[:int(round(0.2 * len(ii)))]] = 2
torch.save(torch.tensor(part), f'{CACHE}/split.pt'); print('réserve c', (part == 2).sum(), flush=True)
# instabilité (identique à prep.py de t3-methodes)
g = gen(99); inst = torch.zeros(N)
for v in range(8):
    for i in range(0, N, 250):
        vv, _ = train_view(to01(fg[i:i + 250]), g, True)
        with torch.no_grad(), ac(): z = orig(norm(vv)).float().cpu()
        inst[i:i + 250] += -F.log_softmax(z, 1)[torch.arange(len(z)), yhat[i:i + 250]] / 8


def degrade(x, g, k):
    """vue dégradée : recadrage, flou (sigma <= k), couleur, bruit léger, carré gris de 40k px ; k règle la force."""
    B = x.shape[0]; v, _ = affine(x, g, scale=(0.25, 1.0), flip=True)
    v = blur(v, U(B, 0.05, k, g)); v = color(v, g, 0.2, 0.2, 0.3)
    v = (v + torch.randn(v.shape, device=dev, generator=g) * 0.03).clamp(0, 1)
    s = int(40 * k); y0 = int(U(1, 0, 224 - s, g)); x0 = int(U(1, 0, 224 - s, g)); v[:, :, y0:y0 + s, x0:x0 + s] = 0.5
    return v


best = None
for k in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:  # force la plus proche de 85 % d'accord avec yhat
    g = gen(5); acc = []
    for i in range(0, 1000, 250):
        with torch.no_grad(), ac(): acc.append((orig(norm(degrade(to01(fg[i:i + 250]), g, k))).argmax(1).cpu() == yhat[i:i + 250]).float())
    a = torch.cat(acc).mean().item(); print(f'force {k}: accord {a:.3f}', flush=True)
    if best is None or abs(a - 0.85) < abs(best[1] - 0.85): best = (k, a)
k = best[0]
K_DEG = k; g = gen(7); pdeg = torch.zeros(N, NC)
for v in range(20):
    for i in range(0, N, 250):
        with torch.no_grad(), ac(): pdeg[i:i + 250] += F.softmax(orig(norm(degrade(to01(fg[i:i + 250]), g, K_DEG))).float().cpu(), 1) / 20
print('moyenne dégradée : accord', (pdeg.argmax(1) == yhat).float().mean().item(), 'max-prob', pdeg.max(1).values.mean().item(), flush=True)
neigh = pdeg.clone(); neigh[torch.arange(N), yhat] = -1; neigh = neigh.argmax(1)          # classe voisine la plus votée
sel = pd.read_csv(os.path.expanduser('~/critique-t3/sel.csv'))
assert (sel.orig_pred.values == yhat.numpy()).all()
cons = torch.tensor(sel.consensus.values); vote = torch.tensor(sel.vote_class.values)
oofl = torch.tensor(sel.oof_b8_loss.values); oofp = torch.tensor(sel.oof_b8_pred.values)
oof_score = oofl + 100 * (oofp != yhat).float()                                         # d'abord les erreurs de la sonde
torch.save({'yhat': yhat, 'lg0': lg0, 'inst': inst, 'pdeg': pdeg, 'neigh': neigh, 'K_DEG': K_DEG,
            'score': {'inst': inst, 'cons': cons, 'oof': oof_score}, 'wrong': {'inst': neigh, 'cons': vote, 'oof': oofp}}, f'{CACHE}/pg_prep.pt')
top = lambda s: set(torch.argsort(-s)[:1050].tolist())
T = {k: top(v) for k, v in [('inst', inst), ('cons', cons), ('oof', oof_score)]}
print('recouvrement top 30 % : inst&cons', len(T['inst'] & T['cons']) / 1050, 'inst&oof', len(T['inst'] & T['oof']) / 1050, 'cons&oof', len(T['cons'] & T['oof']) / 1050)
print('classe voisine (inst) == 2e d orig', (neigh == lg0.topk(2, 1).indices[:, 1]).float().mean().item())
