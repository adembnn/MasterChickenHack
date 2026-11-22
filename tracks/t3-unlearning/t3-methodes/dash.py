"""t3-methodes : tableau de bord commun, conditions du serveur (eval, autocast fp16).
usage : dash.py --fset ab --rset c --combo NOM=delta1:1.0,delta2:1.0 [--combo ...] [--alphas 0.5,1] [--sd NOM=chemin.pt] [--save NOM]
Un 'combo' = orig + somme(coef * delta). --alphas multiplie tous les coefs. --save ecrit le state dict float32 complet dans candidates/.
Mesure : oubli sur fset (exactes + copies legeres), derive sur rset (images du meme jeu NON touchees par l'entrainement, seulement en decoupe),
melanges, CIFAR (regle hors domaine). Les 'scores de loss' sous gold suppose sont des HYPOTHESES."""
import sys, os, argparse; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from targets import gold_moments
ap = argparse.ArgumentParser()
ap.add_argument('--fset', default='all'); ap.add_argument('--rset', default='none'); ap.add_argument('--combo', action='append', default=[])
ap.add_argument('--sd', action='append', default=[]); ap.add_argument('--alphas', default='1'); ap.add_argument('--save', default=''); ap.add_argument('--tag', default=''); ap.add_argument('--tgt', default=''); ap.add_argument('--grp', default='')
a_ = ap.parse_args()
S = split_sets(); fidx, ridx = S[a_.fset], S[a_.rset]
orig_sd = torch.load(BASE + '/model.pt', map_location='cpu', weights_only=True)
model = tt.build_model().to(dev).eval()
lg0 = torch.from_numpy(np.load(f'{CACHE}/exact_orig_logits.npy')); yhat = lg0.argmax(1)
fgu = load_forget()
SL = ['jpeg50', 'rot15', 'cropflip', 'mild']; ST = ['blur_strong', 'color_strong']
D = {'exact': (fgu, lg0)}
for k in SL + ST: D[k] = (torch.from_numpy(np.load(f'{CACHE}/held_{k}.npy')), torch.from_numpy(np.load(f'{CACHE}/held_{k}_orig_logits.npy')))
X = {k: (torch.from_numpy(np.load(f'{CACHE}/{k}.npy')), torch.from_numpy(np.load(f'{CACHE}/{k}_orig_logits.npy'))) for k in ['mixes', 'ood_cifar100'] if os.path.exists(f'{CACHE}/{k}.npy')}
TG = F.softmax(torch.load(f'{CACHE}/tgt_{a_.tgt}.pt')['z'] / 2, 1) if a_.tgt else None
TZ = torch.load(f'{CACHE}/tgt_{a_.tgt}.pt')['z'] if a_.tgt else None
HYP = {'G1 1.4/2.5': (1.4, 2.5), 'G2 3/5.5': (3.0, 5.5), 'G3 4/6.5': (4.0, 6.5), 'G4 5/7.5': (5.0, 7.5)}


def klt2(lo, lg): return kl(F.softmax(lo / 2, 1), F.log_softmax(lg / 2, 1))


def drift(lg, lo, y=None):
    k = klt2(lo, lg).mean().item(); d = {'agree': (lg.argmax(1) == lo.argmax(1)).float().mean().item(), 'kl': k, 'fac': 1 / (1 + k)}
    if y is not None:
        ao = (lo.argmax(1) == y).float().mean().item(); asb = (lg.argmax(1) == y).float().mean().item(); d.update(acc_orig=ao, acc_sub=asb, acc_ratio=min(1., asb / max(ao, 1e-9)), retain=min(1., asb / max(ao, 1e-9)) / (1 + k))
    return d


def evaluate(name):
    out = {'name': name, 'fset': a_.fset, 'rset': a_.rset}; y = yhat
    f = {}
    for k in ['exact'] + SL:
        u8, lo = D[k]; lg = logits_u8(model, u8[fidx]); yy = y[fidx]; ce = F.cross_entropy(lg, yy, reduction='none'); co = F.cross_entropy(lo[fidx], yy, reduction='none')
        f[k] = {'ce_mean': ce.mean().item(), 'ce_std': ce.std().item(), 'wrong': (lg.argmax(1) != yy).float().mean().item(), 'kl_orig': klt2(lo[fidx], lg).mean().item(),
                'orig_ce_mean': co.mean().item(), 'orig_ce_std': co.std().item(), 'maxprob_t2': F.softmax(lg / 2, 1).max(1)[0].mean().item(), 'kl_tgt': (kl(TG[fidx], F.log_softmax(lg / 2, 1)).mean().item() if TG is not None else float('nan')), 'kl_tgt_orig': (kl(TG[fidx], F.log_softmax(lo[fidx] / 2, 1)).mean().item() if TG is not None else float('nan')), 'finite': bool(torch.isfinite(lg).all())}
    if TZ is not None:
        for k in ['exact'] + SL:
            u8, lo = D[k]; lg = logits_u8(model, u8[fidx]); yy = y[fidx]; tz = TZ[fidx]; want = tz.argmax(1) != yy; ce = F.cross_entropy(lg, yy, reduction='none')
            f[k].update(flip_voulu=want.float().mean().item(), flip_touche=(lg.argmax(1)[want] == tz.argmax(1)[want]).float().mean().item() if want.any() else 0.,
                        flip_ce=ce[want].mean().item() if want.any() else 0., flip_non_voulu=(lg.argmax(1)[~want] != yy[~want]).float().mean().item(), ce_non_flip=ce[~want].mean().item())
    f['slight'] = {k: float(np.mean([f[s][k] for s in SL])) for k in ['ce_mean', 'ce_std', 'wrong', 'kl_orig', 'orig_ce_mean', 'orig_ce_std', 'maxprob_t2', 'kl_tgt', 'kl_tgt_orig']}
    sim = {}
    for A, (mu, sd) in HYP.items():
        r = {}
        for k in ['exact', 'slight']:
            do = abs(f[k]['orig_ce_mean'] - mu) + abs(f[k]['orig_ce_std'] - sd); ds = abs(f[k]['ce_mean'] - mu) + abs(f[k]['ce_std'] - sd); r[k] = 1 - min(max(ds / do, 0), 1)
        sim[A] = r
    out['forget'] = f; out['loss_score_suppose'] = sim
    r = {}
    if len(ridx):
        for k in ['exact'] + SL + ST:
            u8, lo = D[k]; lg = logits_u8(model, u8[ridx]); r[k] = drift(lg, lo[ridx], y[ridx])
        r['slight'] = {k: float(np.mean([r[s][k] for s in SL])) for k in ['agree', 'kl', 'fac', 'acc_ratio', 'retain']}
        if a_.grp:      # memes mesures restreintes a un groupe de classes (ex. classes supposees oubliees en entier) et a son complement
            gm = torch.isin(y[ridx], torch.tensor([int(c) for c in a_.grp.split(',')]))
            for nm, mk in [('grp', gm), ('hors', ~gm)]:
                for k in ['exact', 'blur_strong', 'cropflip']:
                    u8, lo = D[k]; ii = ridx[mk]; r[f'{nm}_{k}'] = drift(logits_u8(model, u8[ii]), lo[ii], y[ii])
    out['retain'] = r
    out['drift'] = {k: drift(logits_u8(model, u8), lo) for k, (u8, lo) in X.items()}
    json.dump(out, open(f'{DIAG}/{a_.tag}{name}.json', 'w'), indent=1)
    row = f"| {name} | {f['exact']['ce_mean']:.3f} / {f['exact']['ce_std']:.3f} / {f['exact']['wrong']:.3f} | {f['slight']['ce_mean']:.3f} / {f['slight']['ce_std']:.3f} / {f['slight']['wrong']:.3f} | {f['exact']['kl_orig']:.3f} ; {f['exact']['maxprob_t2']:.3f} ; {f['exact']['kl_tgt']:.3f}/{f['exact']['kl_tgt_orig']:.3f} | "
    if r: row += f"{r['exact']['agree']:.3f} / {r['exact']['kl']:.4f} / {r['exact']['retain']:.3f} | {r['slight']['agree']:.3f} / {r['slight']['kl']:.4f} / {r['slight']['retain']:.3f} | " + ' ; '.join(f"{r[k]['acc_sub']:.3f} vs {r[k]['acc_orig']:.3f} / {r[k]['kl']:.3f}" for k in ST) + ' | '
    else: row += '- | - | - | '
    row += ' ; '.join(f"{d['agree']:.3f} / {d['kl']:.3f}" for d in out['drift'].values()) + ' | ' + ' ; '.join(f"{sim[A]['exact']:.2f}/{sim[A]['slight']:.2f}" for A in sim) + ' |'
    if TZ is not None: e = f['exact']; row += f" bascules voulues {e['flip_voulu']:.3f}, touchees {e['flip_touche']:.3f} (loss {e['flip_ce']:.2f}), non voulues {e['flip_non_voulu']:.3f}, loss des autres {e['ce_non_flip']:.3f} |"
    if a_.grp and r: row += ' RESERVE du groupe : exactes accord %.3f, recadrees acc %.3f vs %.3f, flou fort acc %.3f vs %.3f ; hors groupe : exactes accord %.3f KL %.3f, flou fort %.3f vs %.3f |' % (
        r['grp_exact']['agree'], r['grp_cropflip']['acc_sub'], r['grp_cropflip']['acc_orig'], r['grp_blur_strong']['acc_sub'], r['grp_blur_strong']['acc_orig'], r['hors_exact']['agree'], r['hors_exact']['kl'], r['hors_blur_strong']['acc_sub'], r['hors_blur_strong']['acc_orig'])
    print(row, flush=True); return out


print('| candidat | OUBLI exactes : CE moy / sd / part classe changee | OUBLI copies legeres : idem | oubli : KL(orig,sub) ; maxprob T2 ; KL(cible,sub)/KL(cible,orig) | RESERVE exactes : accord / KL / retention | RESERVE copies legeres | RESERVE flou fort ; couleur forte : acc sub vs orig / KL | melanges ; CIFAR : accord / KL | score loss SUPPOSE, gold moy/sd : 1.4/2.5 (simulateur A) ; 3/5.5 ; 4/6.5 ; 5/7.5 (verificateur) (exactes/legeres) |')
print('|---|---|---|---|---|---|---|---|---|', flush=True)
model.load_state_dict(orig_sd, strict=True); evaluate('orig')
alphas = [float(x) for x in a_.alphas.split(',')]
for c in a_.combo:
    nm, spec = c.split('='); parts = [(s.split(':')[0], float(s.split(':')[1]) if ':' in s else 1.0) for s in spec.split(',')]
    dl = [(torch.load(f'{DELTA}/{p}.pt')['delta'], co) for p, co in parts]
    for al in alphas:
        sd = {k: v.clone() for k, v in orig_sd.items()}
        for d, co in dl:
            for n, v in d.items(): sd[n] += al * co * v
        model.load_state_dict(sd, strict=True); nmm = nm if len(alphas) == 1 else f'{nm}@{al:g}'; evaluate(nmm)
        if a_.save and (len(alphas) == 1 or al == alphas[-1]):
            fs = {k: v.detach().cpu().float().contiguous() for k, v in sd.items()}; torch.save(fs, f'{CAND}/{a_.save}.pt'); print('ecrit', f'{CAND}/{a_.save}.pt', flush=True)
for c in a_.sd:
    nm, path = c.split('='); sd = torch.load(path, map_location='cpu', weights_only=True); model.load_state_dict(sd, strict=True); evaluate(nm)
