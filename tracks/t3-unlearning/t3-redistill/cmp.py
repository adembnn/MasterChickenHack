"""t3-redistill etape 3 : compare les images que la redistillation fait echouer avec celles que bascule t3-methodes (cible tgt_*.pt).
usage : cmp.py NOM [NOM_CIBLE]   (lit out/NOM_logits.npz ; --all : toutes les images, sinon les images ecartees de la source seulement)"""
import sys, os
sys.path.insert(0, os.path.expanduser('~/eac/t3/t3-methodes/code'))
from common import *
ME = BASE + '/t3-redistill'; name = sys.argv[1]; tg = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else 'T85_f20_l814'
if name.endswith('.npy'): lg = torch.from_numpy(np.load(f'{ME}/targets/{name}')).float(); held = torch.ones(len(lg), dtype=torch.bool)
else: z = np.load(f'{ME}/out/{name}_logits.npz'); lg = torch.from_numpy(z['exact']); held = torch.from_numpy(z['held'])
lg0 = torch.from_numpy(np.load(f'{CACHE}/exact_orig_logits.npy')); yhat = lg0.argmax(1); N = len(lg)
tz = torch.from_numpy(np.load(f'{ME}/ext/{tg}')).float() if tg.endswith('.npy') else torch.load(f'{CACHE}/tgt_{tg}.pt')['z'].float().cpu()
cnt = torch.bincount(yhat, minlength=100).float()
for nm, m in [('ecartees', held), ('source', ~held), ('toutes', torch.ones(N, dtype=torch.bool))]:
    if m.sum() == 0: continue
    l, y, t, l0 = lg[m], yhat[m], tz[m], lg0[m]; mine = l.argmax(1) != y; tm = t.argmax(1) != y; both = mine & tm; n = len(y)
    ce = F.cross_entropy(l, y, reduction='none'); cet = F.cross_entropy(t, y, reduction='none')
    o2 = l0.clone(); o2[torch.arange(n), y] = -1e9; second = o2.argmax(1)
    print(f'[{nm}] n={n} | redistill : part ratee {mine.float().mean():.3f}, CE moy {ce.mean():.2f} sd {ce.std():.2f}, CE des ratees {ce[mine].mean() if mine.any() else 0:.2f} | cible {tg} : part basculee {tm.float().mean():.3f}, CE moy {cet.mean():.2f} sd {cet.std():.2f}')
    if mine.any() and tm.any():
        print(f'   recouvrement : {both.sum().item()} images ; = {both.float().sum() / mine.float().sum():.3f} des ratees redistill, {both.float().sum() / tm.float().sum():.3f} des basculees t3-methodes ; attendu au hasard {mine.float().mean() * tm.float().mean() * n:.1f} images')
        if both.any(): print(f'   meme classe fausse sur le recouvrement : {(l.argmax(1)[both] == t.argmax(1)[both]).float().mean():.3f}')
        print(f'   classe fausse redistill = 2e choix d orig : {(l.argmax(1)[mine] == second[mine]).float().mean():.3f} ; cible t3-methodes = 2e choix d orig : {(t.argmax(1)[tm] == second[tm]).float().mean():.3f}')
        print(f'   classe fausse redistill parmi les 45 classes a oublier : {torch.isin(l.argmax(1)[mine], yhat.unique()).float().mean():.3f}')
        print(f'   KL T2 (cible t3-methodes || redistill) {kl(F.softmax(t / 2, 1), F.log_softmax(l / 2, 1)).mean():.3f} ; KL T2 (cible || orig) {kl(F.softmax(t / 2, 1), F.log_softmax(l0 / 2, 1)).mean():.3f}')
    # taux d'erreur par classe contre effectif de la classe dans le jeu a oublier
    cl = y.unique(); er = torch.stack([mine[y == c].float().mean() for c in cl]); ef = cnt[cl]
    if len(cl) > 2: print(f'   correlation (taux d erreur par classe, effectif de la classe dans les 3500) : {torch.corrcoef(torch.stack([er, ef]))[0, 1]:.3f} ; classes a 129-130 images : erreur {mine[cnt[y] >= 129].float().mean():.3f}, autres : {mine[cnt[y] < 129].float().mean():.3f}')
