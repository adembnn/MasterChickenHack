"""t3-poids : accord entre les sources de mauvaise classe par image et de difficulte par image. CPU.
Sources : (oof8) sonde hors echantillon du bloc 8 ; (deg) moyenne des sorties d'orig sur des vues degradees (cache de t3-methodes : pg_logits + flou fort + couleur forte) ;
(adv) classe atteinte par la plus petite perturbation adversariale ; (orig2) deuxieme classe d'orig sur l'image exacte."""
import os, json, numpy as np, torch, torch.nn.functional as F
B = os.path.expanduser('~/eac/t3'); T = B + '/t3-poids/targets'; C = B + '/t3-methodes/cache'
lg0 = torch.from_numpy(np.load(f'{C}/exact_orig_logits.npy')); y = lg0.argmax(1); N = len(y); ar = torch.arange(N)


def wrong(p): q = p.clone(); q[ar, y] = -1; return q.argmax(1)


P = {}; P['oof8'] = torch.from_numpy(np.load(f'{T}/oofA_b8.npy'))
dv = [F.softmax(torch.from_numpy(np.load(f'{C}/held_{k}_orig_logits.npy')), 1) for k in ['blur_strong', 'color_strong']]
pg = np.load(f'{C}/pg_logits.npy'); print('pg_logits', pg.shape)
pg = torch.from_numpy(pg).reshape(-1, N, pg.shape[-1]).transpose(0, 1)          # [types, forces, tirages, N, C] -> [N, V, C]
P['deg'] = (F.softmax(pg.float(), 2).sum(1) + dv[0] + dv[1]) / (pg.shape[1] + 2)
adv_c = torch.from_numpy(np.load(f'{T}/adv_class.npy')); adv_s = torch.from_numpy(np.load(f'{T}/adv_steps.npy')).float()
W = {'oof8': wrong(P['oof8']), 'deg': wrong(P['deg']), 'adv': adv_c, 'orig2': wrong(F.softmax(lg0, 1))}
D = {'oof8 p(y)': P['oof8'][ar, y], 'deg p(y)': P['deg'][ar, y], 'adv pas': adv_s, 'orig marge': lg0.gather(1, y[:, None])[:, 0] - lg0.scatter(1, y[:, None], -1e4).max(1)[0]}
sub = {'toutes (3500)': torch.ones(N, dtype=torch.bool), 'ratees par oof8': P['oof8'].argmax(1) != y, 'ratees par deg': P['deg'].argmax(1) != y, 'les 20 % les plus fragiles (adv)': adv_s <= torch.quantile(adv_s, 0.2)}
ks = list(W)
for sn, m in sub.items():
    print(f'\n### accord sur la mauvaise classe, images : {sn} (n = {int(m.sum())})\n| | ' + ' | '.join(ks) + ' |\n|---|' + '---|' * len(ks))
    for a in ks: print(f'| {a} | ' + ' | '.join(f'{(W[a][m] == W[b][m]).float().mean().item():.3f}' for b in ks) + ' |')


def rk(v): return torch.argsort(torch.argsort(v.double() + 1e-9 * torch.rand(N, dtype=torch.double, generator=torch.Generator().manual_seed(0)))).double()


dk = list(D); print('\n### correlation de rang entre mesures de difficulte (toutes les images)\n| | ' + ' | '.join(dk) + ' |\n|---|' + '---|' * len(dk))
for a in dk: print(f'| {a} | ' + ' | '.join(f'{torch.corrcoef(torch.stack([rk(D[a]), rk(D[b])]))[0, 1].item():.2f}' for b in dk) + ' |')
# recouvrement des 20 % d'images les plus difficiles selon chaque mesure
k = int(0.2 * N); top = {a: set(torch.argsort(D[a])[:k].tolist()) for a in dk}
print('\n### recouvrement des 20 % d images les plus difficiles (hasard : 0,20)\n| | ' + ' | '.join(dk) + ' |\n|---|' + '---|' * len(dk))
for a in dk: print(f'| {a} | ' + ' | '.join(f'{len(top[a] & top[b]) / k:.2f}' for b in dk) + ' |')
# reference : accord attendu si la mauvaise classe d'une image etait remplacee par celle d'une autre image de la meme classe d'origine
g = torch.Generator().manual_seed(1); ref = []
for a in ['oof8', 'deg', 'adv']:
    sh = W[a].clone()
    for c in y.unique().tolist():
        idx = (y == c).nonzero()[:, 0]; sh[idx] = W[a][idx[torch.randperm(len(idx), generator=g)]]
    ref.append(f"{a} : {(sh == W['oof8']).float().mean().item():.3f} / {(sh == W['deg']).float().mean().item():.3f} / {(sh == W['adv']).float().mean().item():.3f}")
print('\nreference, mauvaise classe prise sur une autre image de la meme classe d origine (accord avec oof8 / deg / adv) :', ' ; '.join(ref))
v = torch.stack([W['oof8'], W['deg'], W['adv']], 1); cons = torch.full((N,), -1, dtype=torch.long)
for i in range(N):
    u, cnt = v[i].unique(return_counts=True)
    if cnt.max() >= 2: cons[i] = u[cnt.argmax()]
np.save(f'{T}/consensus_class.npy', cons.numpy()); np.save(f'{T}/deg_probs.npy', P['deg'].numpy().astype(np.float32))
print(f'\nconsensus (au moins 2 sources sur 3 d accord) : {(cons >= 0).float().mean().item():.3f} des images ; 3 sur 3 : {((v[:, 0] == v[:, 1]) & (v[:, 1] == v[:, 2])).float().mean().item():.3f} ; ecrit consensus_class.npy (-1 = pas d accord) et deg_probs.npy'); print('FINAGREE')
