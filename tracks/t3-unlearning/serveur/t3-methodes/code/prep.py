"""S1 : difficulte par image (a instabilite, b kNN, c regression logistique hors-echantillon) + distribution 'mauvaise classe' w(x).
Aucune image externe. Sortie : adem/cache/s1_prep.pt"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
orig = load_orig(); fg = load_forget(); N = len(fg)
lg0 = torch.from_numpy(np.load(f'{CACHE}/exact_orig_logits.npy'))
yhat = lg0.argmax(1)
# (a) instabilite sous 8 vues fortes : moyenne de -log p_orig(yhat)
g = gen(99); inst = torch.zeros(N)
for v in range(8):
    for i in range(0, N, 250):
        x = to01(fg[i:i + 250]); vv, _ = train_view(x, g, True)
        with torch.no_grad(), ac(): z = orig(norm(vv)).float().cpu()
        inst[i:i + 250] += -F.log_softmax(z, 1)[torch.arange(len(z)), yhat[i:i + 250]] / 8
# (b)(c) features intermediaires : sortie du bloc 8 (index 8), CLS + moyenne des patchs
feats = []
for i in range(0, N, 250):
    x = norm(to01(fg[i:i + 250]))
    with torch.no_grad(), ac():
        h = orig.patch_embed(x); h = orig._pos_embed(h); h = orig.norm_pre(h)
        for b in orig.blocks[:9]: h = b(h)
    h = h.float(); feats.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).cpu())
Fm = torch.cat(feats); Fm = (Fm - Fm.mean(0)) / (Fm.std(0) + 1e-6); Fn = F.normalize(Fm, dim=1)
S = Fn @ Fn.T; S.fill_diagonal_(-2); nn_idx = S.topk(10, 1).indices
purity = (yhat[nn_idx] == yhat[:, None]).float().mean(1)
# (c) regression logistique 5 plis (torch), proba hors-echantillon de yhat
Fg = Fm.to(dev); yg = yhat.to(dev); perm = torch.randperm(N, generator=torch.Generator().manual_seed(0)); folds = perm.chunk(5)
oof = torch.zeros(N, NC)
for k in range(5):
    te = folds[k]; tr = torch.cat([folds[j] for j in range(5) if j != k])
    W = torch.zeros(Fg.shape[1], NC, device=dev, requires_grad=True); bb = torch.zeros(NC, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([W, bb], lr=1, max_iter=150, line_search_fn='strong_wolfe')
    def clo():
        opt.zero_grad(); l = F.cross_entropy(Fg[tr.to(dev)] @ W + bb, yg[tr.to(dev)]) + 1e-2 * (W ** 2).sum() / len(tr); l.backward(); return l
    opt.step(clo)
    with torch.no_grad(): oof[te] = F.softmax(Fg[te.to(dev)] @ W + bb, 1).cpu()
p_oof_y = oof[torch.arange(N), yhat]
print('oof : accuracy vs yhat', (oof.argmax(1) == yhat).float().mean().item(), 'mean p_y', p_oof_y.mean().item(), flush=True)
# difficulte = moyenne des rangs (departage aleatoire), puis scores normaux standardises
def rk(v):
    return torch.argsort(torch.argsort(v + 1e-9 * torch.rand(N, generator=torch.Generator().manual_seed(1)))).float() / (N - 1)
comps = {'instabilite': inst, 'knn_impurete': 1 - purity, 'oof_difficulte': 1 - p_oof_y}
dr = sum(rk(v) for v in comps.values()) / 3
z = torch.distributions.Normal(0., 1.).icdf((torch.argsort(torch.argsort(dr + 1e-9 * torch.rand(N, generator=torch.Generator().manual_seed(2)))).float() + 0.5) / N)
z = (z - z.mean()) / z.std()
dr2 = (rk(comps['knn_impurete']) + rk(comps['oof_difficulte'])) / 2
z_bc = torch.distributions.Normal(0., 1.).icdf((torch.argsort(torch.argsort(dr2 + 1e-9 * torch.rand(N, generator=torch.Generator().manual_seed(3)))).float() + 0.5) / N)
z_bc = (z_bc - z_bc.mean()) / z_bc.std()
ks = list(comps); rho = {f'{a}~{b}': float(torch.corrcoef(torch.stack([rk(comps[a]), rk(comps[b])]))[0, 1]) for i, a in enumerate(ks) for b in ks[i + 1:]}
print('correlations de rang entre composantes', rho, flush=True)
# distribution mauvaise classe : orig a T=3 sur classes != yhat, incline par la similarite cosinus des vecteurs de classe, melange 50/50 avec oof
Wh = orig.head.weight.detach().float().cpu(); cs = F.normalize(Wh, dim=1) @ F.normalize(Wh, dim=1).T
BETA = 8.0
w1 = F.softmax(lg0 / 3, 1) * torch.exp(BETA * cs[yhat]); w1[torch.arange(N), yhat] = 0; w1 = w1 / w1.sum(1, keepdim=True)
w2 = oof.clone() + 1e-4; w2[torch.arange(N), yhat] = 0; w2 = w2 / w2.sum(1, keepdim=True)
w = 0.5 * w1 + 0.5 * w2
top_w = w.argmax(1)
print('mauvaise classe : part des images ou le top-wrong est dans les 5 classes les plus proches (cosinus) de yhat :',
      float((cs[yhat].scatter(1, yhat[:, None], -9).topk(5, 1).indices == top_w[:, None]).any(1).float().mean()), flush=True)
torch.save({'yhat': yhat, 'z': z, 'z_bc': z_bc, 'w': w, 'diff_rank': dr, 'comps': comps, 'logits0': lg0, 'oof': oof, 'rho': rho}, f'{CACHE}/s1_prep.pt')
print('prep fini')
