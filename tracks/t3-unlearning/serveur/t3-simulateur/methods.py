"""Simulateur T3 : methodes d'oubli, toutes dans les conditions du reel (orig + images a oublier SANS etiquettes, aucune autre image),
puis note exacte contre les golds. usage : methods.py WORLD ORIG GOLD1,GOLD2,... CONFIGS.json [OUT.jsonl]
Astuce de vitesse : les 8 premiers blocs ne sont jamais modifies par les methodes 'cache' ; leurs sorties sont calculees une fois."""
from common import *
import copy, score as SC
K0 = 8  # blocs geles en mode cache


def tok(m, x):
    h = m.patch_embed(x); h = m._pos_embed(h); h = m.patch_drop(h); h = m.norm_pre(h)
    for b in m.blocks[:K0]: h = b(h)
    return h


def top(m, h, nblk=4):
    with torch.no_grad():
        for b in m.blocks[K0:12 - nblk]: h = b(h)
    for b in m.blocks[max(K0, 12 - nblk):]: h = b(h)
    return m.head(m.norm(h)[:, 0])


def kl(p, q_log): return (p * (p.clamp_min(1e-12).log() - q_log)).sum(1)


def train_view(x, g, strong):
    B = x.shape[0]; lo = 0.35 if strong else 0.6
    v, area = affine(x, g, scale=(lo, 1.), flip=True); crop_sev = (1 - area) / 0.65; cm = 0.2 if strong else 0.1
    v = color(v, g, cm, cm, cm); col_sev = torch.full((B,), cm / 0.2, device=dev); blur_sev = torch.zeros(B, device=dev); noise_sev = torch.zeros(B, device=dev)
    if strong:
        sg = U(B, 0.1, 1.0, g) * (U(B, 0, 1, g) < 0.5).float(); v = blur(v, torch.where(sg > 0, sg, torch.full_like(sg, 0.05))); blur_sev = sg
        ns = U(B, 0, 0.05, g) * (U(B, 0, 1, g) < 0.5).float(); v = (v + torch.randn(v.shape, device=dev, generator=g) * ns.view(B, 1, 1, 1)).clamp(0, 1); noise_sev = ns / 0.05
    return v, (0.4 * crop_sev + 0.2 * blur_sev + 0.2 * noise_sev + 0.2 * col_sev).clamp(0, 1)


def make_mix(x, g):
    """Melanges construits UNIQUEMENT d'images a oublier : mixup / cutmix / melange de patchs (un tiers chacun)."""
    B, C, H, W = x.shape; perm = torch.randperm(B, device=dev, generator=g); out = x.clone(); n = B // 3
    lam = U(n, 0.3, 0.7, g).view(n, 1, 1, 1); out[:n] = lam * x[:n] + (1 - lam) * x[perm[:n]]
    sl = slice(n, 2 * n); area = U(n, 0.2, 0.6, g); bw = (torch.sqrt(area) * W).long().clamp(8, W - 8)
    x0 = (U(n, 0, 1, g) * (W - bw)).long(); y0 = (U(n, 0, 1, g) * (H - bw)).long()
    ii = torch.arange(H, device=dev).view(1, H, 1); jj = torch.arange(W, device=dev).view(1, 1, W)
    mask = ((ii >= y0.view(n, 1, 1)) & (ii < (y0 + bw).view(n, 1, 1)) & (jj >= x0.view(n, 1, 1)) & (jj < (x0 + bw).view(n, 1, 1))).unsqueeze(1)
    out[sl] = torch.where(mask, x[perm[sl]], x[sl]); r = B - 2 * n; xs = x[2 * n:]
    p = xs.reshape(r, C, 4, 56, 4, 56).permute(0, 1, 2, 4, 3, 5).reshape(r, C, 16, 56, 56)
    idx = torch.argsort(torch.rand(r, 16, device=dev, generator=g), 1); p = torch.gather(p, 2, idx.view(r, 1, 16, 1, 1).expand(-1, C, -1, 56, 56))
    out[2 * n:] = p.reshape(r, C, 4, 4, 56, 56).permute(0, 1, 2, 4, 3, 5).reshape(r, C, H, W)
    return out


def make_mix_pair(x, xp, g):
    """Melange de chaque image avec une partenaire donnee (ici : de la meme classe predite) : mixup pour la moitie, cutmix pour l'autre."""
    B, C, H, W = x.shape; out = x.clone(); n = B // 2
    lam = U(n, 0.35, 0.65, g).view(n, 1, 1, 1); out[:n] = lam * x[:n] + (1 - lam) * xp[:n]
    r = B - n; area = U(r, 0.25, 0.6, g); bw = (torch.sqrt(area) * W).long().clamp(8, W - 8); x0 = (U(r, 0, 1, g) * (W - bw)).long(); y0 = (U(r, 0, 1, g) * (H - bw)).long()
    ii = torch.arange(H, device=dev).view(1, H, 1); jj = torch.arange(W, device=dev).view(1, 1, W)
    mask = ((ii >= y0.view(r, 1, 1)) & (ii < (y0 + bw).view(r, 1, 1)) & (jj >= x0.view(r, 1, 1)) & (jj < (x0 + bw).view(r, 1, 1))).unsqueeze(1)
    out[n:] = torch.where(mask, xp[n:], x[n:]); return out


# ---- cibles S1 (reprises de scripts-t3v1/targets.py) ----
MC, SC_, MW, SW = 0.12, 0.25, 3.0, 1.3
def gold_moments(A):
    mu = A * MC + (1 - A) * MW; var = A * (SC_ ** 2 + MC ** 2) + (1 - A) * (SW ** 2 + MW ** 2) - mu ** 2; return mu, math.sqrt(var)


def fit_ab(z, mu, sd):
    zz = z.double().to(dev); a = torch.linspace(-1, 9, 201, device=dev).double().view(-1, 1, 1); b = torch.linspace(0.05, 8, 160, device=dev).double().view(1, -1, 1)
    L = F.softplus(b * zz.view(1, 1, -1) - a); e = (L.mean(2) - mu) ** 2 + (L.std(2) - sd) ** 2; i = int(e.argmin())
    return a.flatten()[i // 160].item(), b.flatten()[i % 160].item(), e.flatten()[i].item()


def zscore_rank(v, seed=2):
    N = len(v); r = torch.argsort(torch.argsort(v + 1e-9 * torch.rand(N, generator=torch.Generator().manual_seed(seed)))).float()
    z = torch.distributions.Normal(0., 1.).icdf((r + 0.5) / N); return (z - z.mean()) / z.std()


class Bench:
    def __init__(s, world, orig, golds, V=4, M=6000):
        s.world = world; s.real = world == 'REAL'; s.oname = orig or 'real'
        if s.real:   # le vrai probleme : vrai model.pt, vraies images a oublier, ni etiquettes ni gold ni jeu de controle
            s.W = None; s.rdir = f'{ROOT}/real'; os.makedirs(s.rdir, exist_ok=True); s.orig = load_sd(os.path.dirname(ROOT) + '/model.pt')
            s.fg = torch.from_numpy(np.load(os.path.dirname(ROOT) + '/forget_images.npz')['images']); s.N = len(s.fg); s.L = s.O = s.G = None
        else:
            s.W = World(world); s.rdir = f'{ROOT}/runs/{world}'; s.orig = load_sd(f'{s.rdir}/{orig}.pt'); s.fg = s.W.x('forget'); s.N = len(s.fg)
            s.L = labels(s.W); s.O = dict(np.load(f'{s.rdir}/{orig}.npz')); s.G = [dict(np.load(f'{s.rdir}/{g}.npz')) for g in golds]
        for p in s.orig.parameters(): p.requires_grad_(False)
        t0 = time.time(); g = gen(31337); Tv, Zv, Sv = [], [], []
        with torch.no_grad(), ac():
            for v in range(V + 1):
                tt, zz, ss = [], [], []
                for i in range(0, s.N, 250):
                    x = to01(s.fg[i:i + 250])
                    if v == 0: vv, sev = x, torch.zeros(len(x), device=dev)
                    else: vv, sev = train_view(x, g, strong=(v % 2 == 0))
                    h = tok(s.orig, norm(vv)); tt.append(h.half()); zz.append(top(s.orig, h).float()); ss.append(sev)
                Tv.append(torch.cat(tt)); Zv.append(torch.cat(zz)); Sv.append(torch.cat(ss))
            s.Tv = torch.stack(Tv); s.Zv = torch.stack(Zv); s.Sv = torch.stack(Sv); del Tv, tt          # [V+1,N,197,768], [V+1,N,100], [V+1,N]
            tm, zm = [], []
            for k in range(M // 240):
                idx = torch.randint(0, s.N, (240,), generator=torch.Generator().manual_seed(k + 100)); h = tok(s.orig, norm(make_mix(to01(s.fg[idx]), g)))
                tm.append(h.half()); zm.append(top(s.orig, h).float())
            s.nmix = sum(len(t) for t in tm); yh = s.Zv[0].argmax(1).cpu(); byc = {int(c): (yh == c).nonzero().flatten() for c in yh.unique()}
            for k in range(M // 240):   # melanges INTRA-classe : images 'de la classe mais hors du jeu', pour une ancre qui protege les images jamais vues de la classe
                idx = torch.randint(0, s.N, (240,), generator=torch.Generator().manual_seed(k + 5000)); gp = torch.Generator().manual_seed(k + 9000)
                par = torch.stack([byc[int(yh[a])][torch.randint(0, len(byc[int(yh[a])]), (1,), generator=gp)[0]] for a in idx])
                h = tok(s.orig, norm(make_mix_pair(to01(s.fg[idx]), to01(s.fg[par]), g))); tm.append(h.half()); zm.append(top(s.orig, h).float())
            s.Tm = torch.cat(tm); s.Zm = torch.cat(zm); del tm
            s.E = {}
            for k, u8 in ([] if s.real else [('fx', s.fg), ('ctrl', s.W.x('ctrl')), ('val', s.W.x('val'))] + [(f'fa{t}', torch.from_numpy(np.ascontiguousarray(s.W.alt[t]))) for t in range(4)]):
                s.E[k] = torch.cat([tok(s.orig, norm(to01(u8[i:i + 250]))).half().cpu() for i in range(0, len(u8), 250)])   # en RAM : la carte est partagee
        s.z0 = s.Zv[0]; s.yhat = s.z0.argmax(1); ytrue = s.yhat if s.real else torch.from_numpy(s.L['fx']).to(dev)
        print(f'cache pret en {time.time() - t0:.0f}s ; mem GPU {torch.cuda.memory_allocated() / 1e9:.1f} Go ; pseudo-etiquettes justes {(s.yhat == ytrue).float().mean():.4f}', flush=True)
        s.prep()

    # ---- difficulte par image et distribution 'mauvaise classe', sans etiquettes ----
    def prep(s):
        N, yhat = s.N, s.yhat.cpu(); g = gen(99); inst = torch.zeros(N); ptta1 = torch.zeros(N, NC); ptta2 = torch.zeros(N, NC); t0 = time.time(); pf = f'{s.rdir}/prep_{s.oname}.pt'   # un cache par modele d'origine
        if os.path.exists(pf) and not os.environ.get('REPREP'):
            P = torch.load(pf); inst = P['comps']['instabilite']; purity = 1 - P['comps']['knn_impurete']; oof = P['oof']; p_oof = oof[torch.arange(N), yhat]; ptta1, ptta2 = P['ptta1'], P['ptta2']; print('prep relu depuis', pf, flush=True)
        else:
            with torch.no_grad(), ac():
                for v in range(8):
                    for i in range(0, N, 250):
                        vv, _ = train_view(to01(s.fg[i:i + 250]), g, True); z = s.orig(norm(vv)).float().cpu()
                        inst[i:i + 250] += -F.log_softmax(z, 1)[torch.arange(len(z)), yhat[i:i + 250]] / 8
                for v in range(12):   # professeur 'vues dures' : couleur forte + flou + recadrage (la ou orig decroche)
                    for i in range(0, N, 250):
                        x = to01(s.fg[i:i + 250]); x, _ = affine(x, g, scale=(0.35, 1.), flip=True); x = color(x, g, 0.4, 0.4, 0.4, 0.1)
                        if v % 2: x = blur(x, U(len(x), 0.8, 2.0, g))
                        z = s.orig(norm(x)).float().cpu(); ptta1[i:i + 250] += F.softmax(z, 1) / 12; ptta2[i:i + 250] += F.softmax(z / 2, 1) / 12
            print(f'prep : vues {time.time() - t0:.0f}s', flush=True)
            Fm = torch.cat([torch.cat([s.Tv[0][i:i + 500, 0].float(), s.Tv[0][i:i + 500, 1:].float().mean(1)], 1) for i in range(0, N, 500)])
            Fm = (Fm - Fm.mean(0)) / (Fm.std(0) + 1e-6); Fn = F.normalize(Fm, dim=1); S = Fn @ Fn.T; S.fill_diagonal_(-2)       # tout sur la carte : les coeurs CPU sont satures
            purity = (s.yhat[S.topk(10, 1).indices] == s.yhat[:, None]).float().mean(1).cpu(); print(f'prep : kNN {time.time() - t0:.0f}s', flush=True)
            Fg = Fm.float().cpu(); yg = yhat; torch.set_num_threads(4); folds = torch.randperm(N, generator=torch.Generator().manual_seed(0)).chunk(5); oof = torch.zeros(N, NC)
            for k in range(5):   # sur CPU : des milliers de petites operations seraient tres lentes sur une carte partagee
                te = folds[k]; tr = torch.cat([folds[j] for j in range(5) if j != k]); Xt, yt = Fg[tr], yg[tr]
                Wt = torch.zeros(Fg.shape[1], NC, requires_grad=True); bb = torch.zeros(NC, requires_grad=True)
                opt = torch.optim.LBFGS([Wt, bb], lr=1, max_iter=80, line_search_fn='strong_wolfe')
                def clo():
                    opt.zero_grad(); l = F.cross_entropy(Xt @ Wt + bb, yt) + 1e-2 * (Wt ** 2).sum() / len(tr); l.backward(); return l
                opt.step(clo)
                with torch.no_grad(): oof[te] = F.softmax(Fg[te] @ Wt + bb, 1)
            p_oof = oof[torch.arange(N), yhat]; print(f'prep : tete validee croisee {time.time() - t0:.0f}s', flush=True)
        def rk(v): return torch.argsort(torch.argsort(v + 1e-9 * torch.rand(N, generator=torch.Generator().manual_seed(1)))).float() / (N - 1)
        s.comps = {'instabilite': inst, 'knn_impurete': 1 - purity, 'oof_difficulte': 1 - p_oof, 'tta_dur': -torch.log(ptta1[torch.arange(N), yhat].clamp_min(1e-6)),
                   'marge_orig': -(s.z0.cpu().topk(2, 1).values[:, 0] - s.z0.cpu().topk(2, 1).values[:, 1])}
        s.zdiff = {'z3': zscore_rank(sum(rk(s.comps[k]) for k in ['instabilite', 'knn_impurete', 'oof_difficulte']) / 3), 'z_bc': zscore_rank((rk(s.comps['knn_impurete']) + rk(s.comps['oof_difficulte'])) / 2),
                   'z_tta': zscore_rank(s.comps['tta_dur']), 'z_inst': zscore_rank(inst), 'z_oof': zscore_rank(1 - p_oof),
                   'z4': zscore_rank(sum(rk(s.comps[k]) for k in ['instabilite', 'knn_impurete', 'oof_difficulte', 'tta_dur']) / 4), 'z_rand': zscore_rank(torch.rand(N, generator=torch.Generator().manual_seed(5)))}
        Wh = s.orig.head.weight.detach().float().cpu(); cs = F.normalize(Wh, dim=1) @ F.normalize(Wh, dim=1).T; z0 = s.z0.cpu()
        w1 = F.softmax(z0 / 3, 1) * torch.exp(8.0 * cs[yhat]); w1[torch.arange(N), yhat] = 0; w1 = w1 / w1.sum(1, keepdim=True)
        w2 = oof.clone() + 1e-4; w2[torch.arange(N), yhat] = 0; w2 = w2 / w2.sum(1, keepdim=True)
        w3 = ptta2.clone() + 1e-5; w3[torch.arange(N), yhat] = 0; w3 = w3 / w3.sum(1, keepdim=True)
        wu = torch.ones(N, NC); wu[torch.arange(N), yhat] = 0; wu = wu / wu.sum(1, keepdim=True)
        wc = torch.stack([w3[yhat == k].mean(0) if (yhat == k).any() else wu[0] for k in range(NC)])[yhat]; wc = wc / wc.sum(1, keepdim=True)   # moyenne de la classe : la classe soeur dominante
        s.wsrc = {'w_s1': 0.5 * w1 + 0.5 * w2, 'w_head': w1, 'w_oof': w2, 'w_tta': w3, 'w_unif': wu, 'w_mix3': (w1 + w2 + w3) / 3, 'w_ttac': wc, 'w_ttamix': 0.5 * w3 + 0.5 * wc}
        s.oof = oof; s.ptta1 = ptta1; s.ptta2 = ptta2
        torch.save({'comps': s.comps, 'yhat': yhat, 'oof': oof, 'ptta1': ptta1, 'ptta2': ptta2, 'w': s.wsrc, 'z0': z0}, pf)

    # ---- evaluation ----
    @torch.no_grad()
    def evaluate(s, m, full=False):
        m.eval()
        if full: return eval_all(m, s.W)
        def lg(k):
            with ac(): return torch.cat([top(m, s.E[k][i:i + 500].to(dev)).float() for i in range(0, len(s.E[k]), 500)]).cpu().numpy()
        return {'fx': lg('fx'), 'fa4': np.stack([lg(f'fa{t}') for t in range(4)]), 'ctrl': lg('ctrl'), 'val': lg('val')}

    def score(s, r):
        sc = SC.score(r, s.O, s.G, s.L); sc['alt_types'] = {ALT_NAMES[t]: {k: SC.score(r, s.O, s.G, s.L, alt=t)['alt'][k] for k in ['loss', 'pred', 'L_mean', 'L_std', 'acc']} for t in range(4)}
        return sc

    def trainable(s, m, nblk, head=True):
        for p in m.parameters(): p.requires_grad_(False)
        tr = []
        for n, p in m.named_parameters():
            blk = n.startswith('blocks.') and int(n.split('.')[1]) >= 12 - nblk
            if (blk or ((n.startswith('norm.') or n.startswith('head.')) and head)) and 'gamma' not in n: p.requires_grad_(True); tr.append(p)
        return tr

    def aj(s, c, n, G):
        a = c.get('anchor', 'mix'); lo, hi = (0, s.nmix) if a == 'mix' else (s.nmix, len(s.Tm)) if a == 'intra' else (0, len(s.Tm))
        return torch.randint(lo, hi, (n,), device=dev, generator=G)

    # ---- cibles ----
    def targets(s, c):
        """Renvoie (q1 [N,C] cible T=1, q2 [N,C] cible T=2, p_y [N], w [N,C]) selon la config."""
        N, yhat = s.N, s.yhat; kind = c.get('target', 's1')
        if kind == 'oracle':      # borne haute : vraies sorties des golds (inaccessible dans le reel)
            q1 = torch.from_numpy(np.mean([np.exp(SC.lsm(g['fx'])) for g in s.G], 0)).float().to(dev); q2 = torch.from_numpy(np.mean([np.exp(SC.lsm(g['fx'], 2.0)) for g in s.G], 0)).float().to(dev)
            return q1, q2, q1[torch.arange(N, device=dev), yhat], None
        if kind == 'soft':        # S0 : orig adouci
            return F.softmax(s.z0 / c['tau'], 1), F.softmax(s.z0 / (2 * c['tau']), 1), None, None
        if kind == 'tta':         # professeur 'vues dures' brut
            q1 = s.ptta1.to(dev); q2 = s.ptta2.to(dev); return q1, q2, q1[torch.arange(N, device=dev), yhat], None
        if kind == 'count':       # cibles selon la part de la classe oubliee (hypothese : ctot images d'entrainement par classe)
            cnt = torch.bincount(yhat, minlength=NC).float(); frac = (cnt / c.get('ctot', 130))[yhat].cpu(); full = frac >= c.get('full_thr', 0.97)
            z = s.zdiff[c.get('z', 'z3')]; mu, sd = (c['mu_p'], c['sd_p']) if 'mu_p' in c else gold_moments(c.get('A_part', 0.92)); Lt = torch.zeros(N)
            if (~full).any() and not c.get('part_keep'): zp = z[~full]; zp = (zp - zp.mean()) / zp.std(); a, b, _ = fit_ab(zp, mu, sd); Lt[~full] = F.softplus(b * zp - a).float()
            if full.any(): zf = z[full]; zf = (zf - zf.mean()) / zf.std(); Lt[full] = (c.get('Lfull', 10.0) + c.get('sdfull', 3.0) * zf).clamp_min(2.0)
            p_y = torch.exp(-Lt).to(dev); w = s.wsrc[c.get('w', 'w_s1')].clone()
            if 'wfull' in c: w[full] = s.wsrc[c['wfull']][full]
            if c.get('mask_full', 1):   # le gold n'a jamais vu les classes entieres : il ne peut pas les predire, on les retire des destinations
                fcl = torch.unique(yhat[full.to(dev)]).cpu(); wf = w[full]; wf[:, fcl] = 0; w[full] = wf / wf.sum(1, keepdim=True).clamp_min(1e-12)
            if 'wpow' in c: wf = w[full].clamp_min(1e-12).pow(c['wpow']); w[full] = wf / wf.sum(1, keepdim=True)   # T=1 = carre de la cible T=2 voulue
            w = w.to(dev); q1, q2 = s.make_q(p_y, w, c)
            if 'q2pow_full' in c:   # cible T=2 plus ou moins etalee pour les classes entieres
                cc = dict(c); cc['q2pow'] = c['q2pow_full']; cc['q2mix'] = c.get('q2mix_full', 0.0); fm = full.to(dev); q2[fm] = s.make_q(p_y, w, cc)[1][fm]
            return q1, q2, p_y, w
        if c.get('mu') == 'gold': mu, sd, _ = SC.gold_ref([g['fx'] for g in s.G], s.L['fx'])
        elif 'mu' in c: mu, sd = c['mu'], c['sd']
        else: mu, sd = gold_moments(c['A'])
        z = s.zdiff[c.get('z', 'z3')]; a, b, err = fit_ab(z, mu, sd); p_y = torch.sigmoid(a - b * z).to(dev).float(); w = s.wsrc[c.get('w', 'w_s1')].to(dev)
        q1, q2 = s.make_q(p_y, w, c); return q1, q2, p_y, w

    def make_q(s, p_y, w, c):
        N = len(p_y); q = (1 - p_y)[:, None] * w; q[torch.arange(N, device=q.device), s.yhat if len(p_y) == s.N else c['_y']] += p_y
        h = q.clamp_min(1e-12).pow(c.get('q2pow', 0.5)); h = h / h.sum(1, keepdim=True); e = c.get('q2mix', 0.15)
        q2 = (1 - e) * h + e * w; u = c.get('q2unif', 0.0)      # plancher uniforme : la moyenne de plusieurs golds a T=2 met un peu de masse partout
        return q, (1 - u) * q2 + u / q2.shape[1]

    # ---- entraineur generique 'cibles' (S0, S1, oracle, tta) ----
    def fit(s, c):
        m = copy.deepcopy(s.orig); nblk = c.get('nblk', 4); tr = s.trainable(m, nblk); steps = c.get('steps', 600); lr = c.get('lr', 1e-4); lam = c.get('lam', 10.0); wm = c.get('wm', 1.0)
        w1, w2 = c.get('w1', 1.0), c.get('w2', 1.0); sevc = c.get('sevc', 0.15); BF, BA = c.get('bf', 128), c.get('ba', 64); wl = c.get('wl', 0.0)
        q1, q2, p_y, w = s.targets(c); Lt = -torch.log(q1[torch.arange(s.N, device=dev), s.yhat].clamp_min(1e-8)); mu_t, sd_t = Lt.mean(), Lt.std()
        opt = torch.optim.AdamW(tr, lr=lr, weight_decay=0.0); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1, (t + 1) / 30) * 0.5 * (1 + math.cos(math.pi * t / steps)))
        G = torch.Generator(device=dev).manual_seed(c.get('seed', 0)); V = s.Tv.shape[0]; soft = c.get('target') == 'soft'
        for it in range(steps):
            i = torch.randint(0, s.N, (BF,), device=dev, generator=G); v = torch.randint(0, V, (BF,), device=dev, generator=G); j = s.aj(c, BA, G)
            if soft: t1 = F.softmax(s.Zv[v, i] / c['tau'], 1); t2 = F.softmax(s.Zv[v, i] / (2 * c['tau']), 1)
            elif p_y is not None and w is not None and sevc > 0:
                py = p_y[i] * (1 - sevc * s.Sv[v, i]); cc = dict(c); cc['_y'] = s.yhat[i]; t1, t2 = s.make_q(py, w[i], cc)
            else: t1, t2 = q1[i], q2[i]
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16): zz = top(m, torch.cat([s.Tv[v, i], s.Tm[j]]), nblk).float()
            zf, za = zz[:BF], zz[BF:]
            lf = kl(t1, F.log_softmax(zf, 1)).mean(); l2 = kl(t2, F.log_softmax(zf / 2, 1)).mean(); la = kl(F.softmax(s.Zm[j] / 2, 1), F.log_softmax(za / 2, 1)).mean()
            ex = v == 0; ce = F.cross_entropy(zf, s.yhat[i], reduction='none'); lm = ((ce.mean() - mu_t) ** 2 + (ce.std() - sd_t) ** 2) if wm > 0 and not soft else zf.sum() * 0
            loss = w1 * lf + w2 * l2 + lam * la + wm * lm
            if wl > 0: loss = loss + wl * F.smooth_l1_loss(ce, Lt[i])        # loss par image regressee vers sa cible (fixe moyenne ET ecart-type)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(tr, 1.0); opt.step(); sched.step()
        return m, False, f'cible loss moy {mu_t:.3f} sd {sd_t:.3f} ; fin : KL1 {lf.item():.3f} KL2 {l2.item():.3f} ancre {la.item():.4f}'

    # ---- methodes classiques ----
    def classic(s, c):
        kind = c['method']; m = copy.deepcopy(s.orig); nblk = c.get('nblk', 4); tr = s.trainable(m, nblk); steps = c.get('steps', 200); lr = c.get('lr', 2e-5); lam = c.get('lam', 0.0)
        opt = torch.optim.AdamW(tr, lr=lr, weight_decay=0.0); G = torch.Generator(device=dev).manual_seed(c.get('seed', 0)); V = s.Tv.shape[0]; BF, BA = 128, 64
        yr = (s.yhat + torch.randint(1, NC, (s.N,), device=dev, generator=G)) % NC; stop = c.get('stop_loss', None); info = ''
        if kind == 'erase_repair':      # effacer : tete (+ nreset derniers blocs) remis a l'initialisation ; reparer : imiter orig sur les melanges seulement
            fresh = build().to(dev); k = c.get('nreset', 1); sd = m.state_dict(); fs = fresh.state_dict()
            for n in sd:
                if n.startswith('head.') or (n.startswith('blocks.') and int(n.split('.')[1]) >= 12 - k and 'gamma' not in n): sd[n] = fs[n].clone()
            m.load_state_dict(sd); tr = s.trainable(m, max(nblk, k)); opt = torch.optim.AdamW(tr, lr=lr, weight_decay=0.0)
        for it in range(steps):
            i = torch.randint(0, s.N, (BF,), device=dev, generator=G); v = torch.randint(0, V, (BF,), device=dev, generator=G); j = s.aj(c, BA, G)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16): zz = top(m, torch.cat([s.Tv[v, i], s.Tm[j]]), max(nblk, c.get('nreset', 0))).float()
            zf, za = zz[:BF], zz[BF:]; la = kl(F.softmax(s.Zm[j] / 2, 1), F.log_softmax(za / 2, 1)).mean(); ce = F.cross_entropy(zf, s.yhat[i], reduction='none')
            if kind == 'ga': loss = -ce.mean() + lam * la                                         # montee de gradient
            elif kind == 'rl': loss = F.cross_entropy(zf, yr[i]) + lam * la                       # etiquettes aleatoires
            elif kind == 'npo':                                                                   # Negative Preference Optimization (beta=1)
                lp = -ce; lp0 = -F.cross_entropy(s.Zv[v, i], s.yhat[i], reduction='none'); loss = 2 * F.softplus(lp - lp0).mean() + lam * la
            elif kind == 'scrub':                                                                 # SCRUB : max KL sur l'oubli (msteps premiers pas pairs), min KL sur le substitut du reste
                if it < c.get('msteps', steps // 2) and it % 2 == 0: loss = -kl(F.softmax(s.Zv[v, i], 1), F.log_softmax(zf, 1)).mean()
                else: loss = lam * la + kl(F.softmax(s.Zm[j], 1), F.log_softmax(za, 1)).mean()
            elif kind == 'erase_repair': loss = la + kl(F.softmax(s.Zm[j], 1), F.log_softmax(za, 1)).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(tr, 1.0); opt.step()
            if stop is not None and it % 5 == 0 and ce.mean().item() > stop: info = f'arret au pas {it} (loss {ce.mean().item():.2f})'; break
        return m, False, info or f'fin : CE oubli {ce.mean().item():.3f} ancre {la.item():.4f}'

    def simple(s, c):
        kind = c['method']; m = copy.deepcopy(s.orig)
        if kind == 'neg':             # vecteur de tache : orig affine encore sur l'oubli (CE, pseudo-etiquettes) ; sub = orig - alpha * (affine - orig)
            ft = copy.deepcopy(s.orig); tr = s.trainable(ft, c.get('nblk', 4)); opt = torch.optim.SGD(tr, lr=c.get('lr', 1e-3)); G = torch.Generator(device=dev).manual_seed(0); V = s.Tv.shape[0]
            for it in range(c.get('steps', 100)):
                i = torch.randint(0, s.N, (128,), device=dev, generator=G); v = torch.randint(0, V, (128,), device=dev, generator=G)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16): zf = top(ft, s.Tv[v, i], c.get('nblk', 4)).float()
                loss = F.cross_entropy(zf, s.yhat[i]); opt.zero_grad(); loss.backward(); opt.step()
            nrm = 0.0
            with torch.no_grad():
                for (n, p), (_, q), (_, o) in zip(m.named_parameters(), ft.named_parameters(), s.orig.named_parameters()):
                    if q.requires_grad: d = q - o; nrm += d.pow(2).sum().item(); p.copy_(o - c['alpha'] * d)
            return m, False, f'norme du vecteur de tache {nrm ** 0.5:.4f}'
        if kind == 'none': return m, False, ''
        if kind == 'temp':            # temperature globale : tete divisee par tau (aucun entrainement)
            with torch.no_grad(): m.head.weight /= c['tau']; m.head.bias /= c['tau']
            return m, False, ''
        if kind == 'noise':           # bruit gaussien sur les poids des nblk derniers blocs + tete
            g = torch.Generator(device=dev).manual_seed(c.get('seed', 0))
            with torch.no_grad():
                for p in s.trainable(m, c.get('nblk', 4)):
                    if p.ndim > 1: p += torch.randn(p.shape, device=dev, generator=g) * p.std() * c['sigma']
            return m, False, ''

    def run(s, c):
        t0 = time.time(); meth = c['method']
        m, full, info = (s.fit(c) if meth == 'fit' else s.simple(c) if meth in ('none', 'temp', 'noise', 'neg') else s.classic(c))
        r = s.evaluate(m, full); sc = s.score(r); sc['name'] = c['name']; sc['cfg'] = {k: v for k, v in c.items() if not k.startswith('_')}; sc['info'] = info; sc['sec'] = time.time() - t0
        if c.get('save'): np.savez(f'{s.rdir}/sub_{c["name"]}.npz', **r)
        print(SC.row(c['name'], sc), '|', info, f'| {sc["sec"]:.0f}s', flush=True); return sc


if __name__ == '__main__':
    warm()
    world, orig, golds, cfgf = sys.argv[1], sys.argv[2], sys.argv[3].split(','), sys.argv[4]; out = sys.argv[5] if len(sys.argv) > 5 else f'{ROOT}/runs/{world}/results.jsonl'
    B = Bench(world, orig, golds, V=int(os.environ.get('BENCH_V', 4)), M=int(os.environ.get('BENCH_M', 6000))); print(SC.HDR, flush=True)
    todo = f'{ROOT}/runs/{world}/todo'; os.makedirs(todo, exist_ok=True); lists = [cfgf] if cfgf != '-' else []; idle = 0
    while True:
        for cf in lists:
            for c in json.load(open(cf)):
                for essai in range(6):      # carte partagee : en cas de manque de memoire, on attend et on recommence
                    try:
                        sc = B.run(c); open(out, 'a').write(json.dumps(sc, default=float) + chr(10)); break
                    except (torch.OutOfMemoryError, RuntimeError) as e:
                        if 'out of memory' not in str(e) and 'ALLOC_FAILED' not in str(e): import traceback; traceback.print_exc(); print('ECHEC', c.get('name'), str(e)[:200], flush=True); break
                        print('memoire pleine, nouvel essai dans 45 s :', c.get('name'), flush=True); time.sleep(45)
                    except Exception as e:
                        import traceback; traceback.print_exc(); print('ECHEC', c.get('name'), str(e)[:200], flush=True); break
            if cf.startswith(todo): os.rename(cf, cf + '.fait')
        lists = sorted(f'{todo}/{f}' for f in os.listdir(todo) if f.endswith('.json'))
        if os.path.exists(f'{todo}/stop'): os.remove(f'{todo}/stop'); break
        if not lists:
            time.sleep(20); idle += 1
            if idle > 90: break          # 30 min sans travail : on libere la carte
        else: idle = 0
