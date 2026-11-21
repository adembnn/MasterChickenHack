"""D. Diagnostics d'un modele aux conditions du serveur (eval, autocast fp16). usage : diag.py NAME CHEMIN.pt
Sortie : adem/diag/NAME.json. Rien n'est envoye au serveur ; aucune image externe n'est utilisee pour entrainer (OOD = regle graduee)."""
import sys; sys.path.insert(0, '/p/scratch/training2666/destors1/MasterChickenHack/unlearning/adem/t3v1')
from unlearn import *
from targets import gold_moments
import glob
name, path = sys.argv[1], sys.argv[2]
out = {'name': name, 'path': path}
# ---- fichier : chargement strict, float32, NaN/Inf, taille
sd = torch.load(path, map_location='cpu', weights_only=True)
m = tt.build_model(); r = m.load_state_dict(sd, strict=True)
out['file'] = {'strict_load': str(r), 'dtypes': sorted({str(v.dtype) for v in sd.values()}), 'nan_inf': bool(any((~torch.isfinite(v)).any().item() for v in sd.values())),
               'size_MB': round(os.path.getsize(path) / 2 ** 20, 1), 'n_keys': len(sd), 'same_keys_as_orig': list(sd.keys()) == list(torch.load(BASE + '/model.pt', map_location='cpu', weights_only=True).keys())}
m = m.to(dev).eval()
lg0 = torch.from_numpy(np.load(f'{CACHE}/exact_orig_logits.npy')); yhat = lg0.argmax(1); N = len(yhat)
TG = {os.path.basename(f)[8:-3]: torch.load(f)['q2'] for f in sorted(glob.glob(f'{CACHE}/targets_*.pt'))}   # cibles T=2 par candidat
out['target_names'] = list(TG)
fg = load_forget()
A_list = [0.75, 0.85, 0.92]


def stats(lg, ytrue, tgt_ref=None):
    ce = F.cross_entropy(lg, ytrue, reduction='none'); p = F.softmax(lg, 1)
    d = {'ce_mean': ce.mean().item(), 'ce_std': ce.std().item(), 'wrong_share': (lg.argmax(1) != ytrue).float().mean().item(), 'maxprob': p.max(1)[0].mean().item(),
         'finite': bool(torch.isfinite(lg).all().item())}
    lp2 = F.log_softmax(lg / 2, 1)
    for k, q2 in TG.items(): d['kl_t2_vs_' + k] = kl(q2, lp2).mean().item()
    return d


groups = {'exact': (fg, lg0)}
res = {}
sets = {'exact': (None, 'exact_orig_logits')}
res['exact'] = stats(logits_u8(m, fg), yhat)
lg_exact = logits_u8(m, fg)
for k in ['jpeg50', 'rot15', 'color_strong', 'blur_strong']:
    u8 = np.load(f'{CACHE}/held_{k}.npy', mmap_mode='r'); lg = logits_u8(m, u8); lo = torch.from_numpy(np.load(f'{CACHE}/held_{k}_orig_logits.npy'))
    res['held_' + k] = stats(lg, yhat); res['held_' + k]['orig_ce_mean'] = F.cross_entropy(lo, yhat).item(); res['held_' + k]['orig_ce_std'] = F.cross_entropy(lo, yhat, reduction='none').std().item()
    res['held_' + k]['agree_orig'] = (lg.argmax(1) == lo.argmax(1)).float().mean().item()
# moyenne des 4 types
H = [res['held_' + k] for k in ['jpeg50', 'rot15', 'color_strong', 'blur_strong']]
res['held_avg'] = {k: float(np.mean([h[k] for h in H])) for k in H[0] if isinstance(H[0][k], float)}
res['held_avg']['finite'] = all(h['finite'] for h in H)
out['forget'] = res
# ---- score de loss simule sous une hypothese de gold (SUPPOSE, non mesure) : 1 - d/d_orig
lo_ex = F.cross_entropy(lg0, yhat, reduction='none')
sim = {}
for A in A_list:
    mu, sd_ = gold_moments(A)
    d_o = abs(lo_ex.mean().item() - mu) + abs(lo_ex.std().item() - sd_)
    d_s = abs(res['exact']['ce_mean'] - mu) + abs(res['exact']['ce_std'] - sd_)
    ho = np.mean([abs(h['orig_ce_mean'] - mu) + abs(h['orig_ce_std'] - sd_) for h in H]); hs = np.mean([abs(h['ce_mean'] - mu) + abs(h['ce_std'] - sd_) for h in H])
    sim[f'A{A}'] = {'mu_gold_suppose': mu, 'sd_gold_suppose': sd_, 'loss_score_exact': float(1 - min(max(d_s / d_o, 0), 1)), 'loss_score_held_avg': float(1 - min(max(hs / ho, 0), 1))}
out['sim_loss_score_assumed'] = sim
# ---- derive par rapport a orig (accord argmax, KL(orig||sub) a T=2) + facteur retain 1/(1+KL)
drift = {}
for k, fn in [('mixes', 'mixes'), ('ood_in100', 'ood_in100'), ('ood_cifar100', 'ood_cifar100'), ('forget_exact', None)]:
    if fn is None: lg, lo = lg_exact, lg0
    else: lg = logits_u8(m, np.load(f'{CACHE}/{fn}.npy', mmap_mode='r')); lo = torch.from_numpy(np.load(f'{CACHE}/{fn}_orig_logits.npy'))
    kk = kl(F.softmax(lo / 2, 1), F.log_softmax(lg / 2, 1)).mean().item()
    drift[k] = {'agree_argmax': (lg.argmax(1) == lo.argmax(1)).float().mean().item(), 'kl_orig_sub_t2': kk, 'retain_kl_factor': 1 / (1 + kk), 'finite': bool(torch.isfinite(lg).all().item()),
                'maxprob': F.softmax(lg, 1).max(1)[0].mean().item(), 'orig_maxprob': F.softmax(lo, 1).max(1)[0].mean().item()}
out['drift'] = drift
out['fp16_forward_finite'] = bool(res['exact']['finite'] and all(d['finite'] for d in drift.values()) and res['held_avg']['finite'])
json.dump(out, open(f'{DIAG}/{name}.json', 'w'), indent=1); print('diag', name, 'ok', flush=True)
