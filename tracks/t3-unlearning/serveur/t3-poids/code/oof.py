"""t3-poids : cibles 'pseudo-gold' hors pli. Sonde lineaire (regression logistique, 100 sorties) sur les caracteristiques d'orig (CLS + moyenne des patchs, sortie des blocs 6, 8, 10).
A : 5 plis stratifies par pseudo-etiquette, 3 tirages de plis, 3 blocs -> chaque image est predite par des sondes qui ne l'ont pas vue (9 predictions moyennees).
B : pour les 9 classes dont ~130 images sont dans le forget set, sonde entrainee SANS aucune image de ces 9 classes ; pour les 36 autres classes, lignes de A.
Sorties (ordre de forget_images.npz) : oofX.npy = moyenne des probabilites (T=1) ; oofX_logits.npy = moyenne des logits ; stats.json. Aucune image externe."""
import sys, os, json, math, numpy as np, torch, torch.nn.functional as F
BASE = os.path.expanduser('~/eac/t3'); os.chdir(BASE); sys.path.insert(0, BASE)
import task_template as tt
dev = 'cuda'; NC = 100; OUT = BASE + '/t3-poids/targets'; os.makedirs(OUT, exist_ok=True)
orig = tt.load_model(); fg = tt.load_forget_images(); N = len(fg); FULL = [0, 6, 7, 8, 29, 30, 31, 33, 38]; BL = [6, 8, 10]
feats = {b: [] for b in BL}; lg = []
with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
    for i in range(0, N, 250):
        x = (fg[i:i + 250].to(dev).permute(0, 3, 1, 2).float() / 255.0 - 0.5) / 0.5
        h = orig.patch_embed(x); h = orig._pos_embed(h); h = orig.norm_pre(h)
        for bi, blk in enumerate(orig.blocks):
            h = blk(h)
            if bi in BL: hf = h.float(); feats[bi].append(torch.cat([hf[:, 0], hf[:, 1:].mean(1)], 1))
        lg.append(orig.forward_head(orig.norm(h)).float())
lg0 = torch.cat(lg); yhat = lg0.argmax(1); print('classes presentes', len(yhat.unique()), 'effectif des 9 classes', int(torch.isin(yhat, torch.tensor(FULL, device=dev)).sum()), flush=True)
Fm = {}
for b in BL: f = torch.cat(feats[b]); Fm[b] = ((f - f.mean(0)) / (f.std(0) + 1e-6)).cpu()
# ajustement des sondes sur CPU : sur un GPU partage, les milliers de petites operations de L-BFGS attendent leur tour
del orig; torch.cuda.empty_cache(); dev = 'cpu'; lg0 = lg0.cpu(); yhat = yhat.cpu(); torch.set_num_threads(8); import time; T0 = time.time()


def fit(X, y, l2=1e-2):
    W = torch.zeros(X.shape[1], NC, device=dev, requires_grad=True); bb = torch.zeros(NC, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([W, bb], lr=1, max_iter=120, line_search_fn='strong_wolfe')
    def clo():
        opt.zero_grad(); l = F.cross_entropy(X @ W + bb, y) + l2 * (W ** 2).sum() / len(y); l.backward(); return l
    opt.step(clo); return W.detach(), bb.detach()


def folds(seed):
    g = torch.Generator().manual_seed(seed); fo = torch.zeros(N, dtype=torch.long)
    for c in yhat.unique().tolist():
        idx = (yhat.cpu() == c).nonzero()[:, 0]; idx = idx[torch.randperm(len(idx), generator=g)]; fo[idx] = torch.arange(len(idx)) % 5
    return fo.to(dev)


ZA = []; perblk = {b: [] for b in BL}
for b in BL:
    for seed in range(3):
        fo = folds(seed); Z = torch.zeros(N, NC, device=dev)
        for k in range(5):
            tr = fo != k; W, bb = fit(Fm[b][tr], yhat[tr]); Z[~tr] = Fm[b][~tr] @ W + bb
        ZA.append(Z); perblk[b].append((Z.argmax(1) == yhat).float().mean().item()); print(f'A bloc {b} tirage {seed} : precision hors pli {perblk[b][-1]:.4f} ({time.time() - T0:.0f}s)', flush=True)
ZA = torch.stack(ZA); PA = F.softmax(ZA, 2).mean(0); LA = ZA.mean(0)
isf = torch.isin(yhat, torch.tensor(FULL, device=dev)); ZB = []
for b in BL:
    W, bb = fit(Fm[b][~isf], yhat[~isf]); ZB.append(Fm[b][isf] @ W + bb)
ZB = torch.stack(ZB); PB = PA.clone(); LB = LA.clone(); PB[isf] = F.softmax(ZB, 2).mean(0); LB[isf] = ZB.mean(0)


def stats(P, L, sel=None):
    sel = torch.ones(N, dtype=torch.bool, device=dev) if sel is None else sel; y = yhat[sel]; r = {'n': int(sel.sum())}
    for nm, p in [('T1_moyenne_des_probas', P[sel]), ('T1_logits_moyens', F.softmax(L[sel], 1)), ('T2_logits_moyens_div2', F.softmax(L[sel] / 2, 1))]:
        ce = -torch.log(p[torch.arange(len(y)), y].clamp_min(1e-12)); r[nm] = {'precision': (p.argmax(1) == y).float().mean().item(), 'loss_moy': ce.mean().item(), 'loss_sd': ce.std().item(), 'maxprob_moy': p.max(1)[0].mean().item(), 'p_pseudo_etiquette_moy': p[torch.arange(len(y)), y].mean().item()}
    pred = P[sel].argmax(1); err = pred != y; tot = 0; top = 0; det = {}
    for c in y.unique().tolist():
        e = pred[(y == c) & err]
        if len(e): cnt = torch.bincount(e, minlength=NC); tot += len(e); top += cnt.max().item(); det[c] = (int((y == c).sum()), len(e), int(cnt.argmax()), cnt.max().item())
    r['erreurs'] = {'part_images_fausses': err.float().mean().item(), 'part_des_erreurs_vers_la_classe_fausse_la_plus_frequente': top / max(tot, 1), 'part_des_erreurs_vers_une_classe_absente_du_forget_set': (~torch.isin(pred[err], yhat.unique())).float().mean().item() if err.any() else 0.}
    r['par_classe (effectif, erreurs, classe fausse dominante, nb)'] = det; return r


S = {'precision_hors_pli_par_bloc': {b: float(np.mean(v)) for b, v in perblk.items()}, 'A_tout': stats(PA, LA), 'A_9classes': stats(PA, LA, isf), 'A_36classes': stats(PA, LA, ~isf), 'B_tout': stats(PB, LB), 'B_9classes': stats(PB, LB, isf),
     'accord_argmax_A_vs_B_sur_9classes': (PA[isf].argmax(1) == PB[isf].argmax(1)).float().mean().item(),
     'orig_2e_classe_vs_A': {'accord de la 2e classe d orig avec la classe fausse de A, sur les images que A rate': (lg0.scatter(1, yhat[:, None], -1e4).argmax(1) == PA.argmax(1))[PA.argmax(1) != yhat].float().mean().item()}}
for nm, P, L in [('oofA', PA, LA), ('oofB', PB, LB)]:
    np.save(f'{OUT}/{nm}.npy', P.cpu().numpy().astype(np.float32)); np.save(f'{OUT}/{nm}_logits.npy', L.cpu().numpy().astype(np.float32))
for i, b in enumerate(BL):                      # versions par bloc : la precision hors pli depend beaucoup de la profondeur
    Pb = F.softmax(ZA[3 * i:3 * i + 3], 2).mean(0); Lb = ZA[3 * i:3 * i + 3].mean(0); PbB = Pb.clone(); LbB = Lb.clone(); PbB[isf] = F.softmax(ZB[i], 1); LbB[isf] = ZB[i]
    S[f'A_bloc{b}'] = stats(Pb, Lb); S[f'B_bloc{b}'] = stats(PbB, LbB)
    for nm, P, L in [(f'oofA_b{b}', Pb, Lb), (f'oofB_b{b}', PbB, LbB)]:
        np.save(f'{OUT}/{nm}.npy', P.cpu().numpy().astype(np.float32)); np.save(f'{OUT}/{nm}_logits.npy', L.cpu().numpy().astype(np.float32))
    print(f'A_bloc{b}'); [print('   ', t, {a: round(v, 4) for a, v in S[f'A_bloc{b}'][t].items()}) for t in ['T1_moyenne_des_probas', 'T1_logits_moyens', 'T2_logits_moyens_div2']]; print('    erreurs', {a: round(v, 4) for a, v in S[f'A_bloc{b}']['erreurs'].items()})
    print(f'B_bloc{b}', {a: round(v, 4) for a, v in S[f'B_bloc{b}']['T1_moyenne_des_probas'].items()}, '| T2', {a: round(v, 4) for a, v in S[f'B_bloc{b}']['T2_logits_moyens_div2'].items()})
r2 = lg0.scatter(1, yhat[:, None], -1e4).argmax(1); P8 = F.softmax(ZA[3:6], 2).mean(0); w8 = P8.argmax(1) != yhat
S['bloc8'] = {'accord 2e classe d orig / classe fausse de la sonde bloc 8, sur ses erreurs': (r2 == P8.argmax(1))[w8].float().mean().item(), 'accord des classes predites bloc 8 / bloc 6': (P8.argmax(1) == F.softmax(ZA[0:3], 2).mean(0).argmax(1)).float().mean().item(),
              'correlation de rang entre p(pseudo-etiquette) bloc 8 et marge d orig': float(np.corrcoef(np.argsort(np.argsort(P8[torch.arange(N), yhat].numpy())), np.argsort(np.argsort((lg0.gather(1, yhat[:, None])[:, 0] - lg0.scatter(1, yhat[:, None], -1e4).max(1)[0]).numpy())))[0, 1])}
print(S['bloc8'])
json.dump(S, open(f'{OUT}/stats.json', 'w'), indent=1)
for k in ['A_tout', 'A_9classes', 'A_36classes', 'B_tout', 'B_9classes']:
    print(k, 'n', S[k]['n'])
    for t in ['T1_moyenne_des_probas', 'T1_logits_moyens', 'T2_logits_moyens_div2']: print('   ', t, {a: round(v, 4) for a, v in S[k][t].items()})
    print('    erreurs', {a: round(v, 4) for a, v in S[k]['erreurs'].items()})
print('par bloc', S['precision_hors_pli_par_bloc']); print('accord A/B sur 9 classes', S['accord_argmax_A_vs_B_sur_9classes']); print(S['orig_2e_classe_vs_A']); print('FINOOF')
