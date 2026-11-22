"""t3-redistill usage (ii) : assemble les sorties hors pli (out/PREFIX_f{i}_logits.npz) en cible par image targets/redistill.npy
[3500,100] float32 de probabilites (T=1), ordre de forget_images.npz, + redistill_logits.npy. Puis controle croise :
l'erreur de chaque selecteur sur l'image exacte i predit-elle l'erreur d'orig sur une regeneration (non-membre) de la meme image i ?"""
import sys, os, glob
sys.path.insert(0, os.path.expanduser('~/eac/t3/t3-methodes/code'))
from common import *
ME = BASE + '/t3-redistill'; TG = ME + '/targets'; os.makedirs(TG, exist_ok=True); pref = sys.argv[1] if len(sys.argv) > 1 else 'F4'
lg0 = torch.from_numpy(np.load(f'{CACHE}/exact_orig_logits.npy')); yhat = lg0.argmax(1); N = len(yhat)
Z = torch.zeros(N, 100); seen = torch.zeros(N, dtype=torch.long); COP = {k: torch.zeros(N, 100) for k in ['jpeg50', 'rot15', 'cropflip', 'mild']}
for f in sorted(glob.glob(f'{ME}/out/{pref}_f*_logits.npz')):
    z = np.load(f); h = torch.from_numpy(z['held']); Z[h] = torch.from_numpy(z['exact'])[h]; seen[h] += 1
    for k in COP: COP[k][h] = torch.from_numpy(z[k])[h]
    print(os.path.basename(f), 'images ecartees', int(h.sum()))
assert (seen == 1).all(), f'plis incomplets : {(seen == 0).sum().item()} images sans prediction'
P = F.softmax(Z, 1); np.save(f'{TG}/redistill.npy', P.numpy().astype(np.float32)); np.save(f'{TG}/redistill_logits.npy', Z.numpy().astype(np.float32))
for k in COP: np.save(f'{TG}/redistill_logits_{k}.npy', COP[k].numpy().astype(np.float32))
ce = F.cross_entropy(Z, yhat, reduction='none'); wr = Z.argmax(1) != yhat; fc = yhat.unique()
o2 = lg0.clone(); o2[torch.arange(N), yhat] = -1e9
print(f'redistill.npy {tuple(P.shape)} float32 | precision vs classe d orig {1 - wr.float().mean():.3f} | loss moy {ce.mean():.2f} sd {ce.std():.2f} | loss des ratees {ce[wr].mean():.2f} | max-prob T1 {P.max(1)[0].mean():.3f} T2 {F.softmax(Z / 2, 1).max(1)[0].mean():.3f}')
print(f'  classe fausse = 2e choix d orig {(Z.argmax(1)[wr] == o2.argmax(1)[wr]).float().mean():.3f} | vers une des 55 classes absentes {1 - torch.isin(Z.argmax(1)[wr], fc).float().mean():.3f}')
for k in COP:
    c = F.cross_entropy(COP[k], yhat, reduction='none'); w2 = COP[k].argmax(1) != yhat
    print(f'  copie {k} : ratees {w2.float().mean():.3f} loss {c.mean():.2f}/{c.std():.2f} ; meme verdict (juste/faux) que sur l exacte {(w2 == wr).float().mean():.3f} ; meme classe predite {(COP[k].argmax(1) == Z.argmax(1)).float().mean():.3f}')
cnt = torch.bincount(yhat, minlength=100).float(); er = torch.stack([wr[yhat == c].float().mean() for c in fc])
print(f'  taux d erreur par classe : min {er.min():.2f} mediane {er.median():.2f} max {er.max():.2f} ; correlation avec l effectif de la classe {torch.corrcoef(torch.stack([er, cnt[fc]]))[0, 1]:.3f}')


def auc(s, lab):
    s = s.double().numpy(); lab = lab.numpy().astype(bool); r = s.argsort().argsort() + 1.0
    n1 = lab.sum(); n0 = (~lab).sum(); return float((r[lab].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0))


def load_sel(spec):
    if spec.endswith('.npy'): t = torch.from_numpy(np.load(spec)).float()
    else: t = torch.load(spec)['z'].float().cpu()
    return t


SEL = {'redistill hors pli (4 blocs)': Z, 'oofA_b8 (t3-poids, sonde lineaire hors pli)': load_sel(f'{ME}/ext/oofA_b8_logits.npy')}
for nm in ['T85_f30_l814', 'T85_f20_l814', 'S30_oof', 'S30_err']:
    if os.path.exists(f'{CACHE}/tgt_{nm}.pt'): SEL[f'tgt_{nm} (t3-methodes)'] = load_sel(f'{CACHE}/tgt_{nm}.pt')
for f in sorted(glob.glob(f'{ME}/ext/noise_*_logits.npy')): SEL['bruit sur les poids ' + os.path.basename(f)[6:-11] + ' (sans entrainement)'] = load_sel(f)
marg = lg0.topk(2, 1)[0]; SEL['marge d orig sur l exacte (temoin)'] = None
ref = load_orig()
print('\n| selecteur | ' + ' | '.join(f'{t} : AUC / meme classe fausse' for t in ['sdxl_i2i_70', 'sdxl_i2i_85', 'sdxl_ipa_60']) + ' |'); print('|---|---|---|---|')
G = {}
for t in ['sdxl_i2i_70', 'sdxl_i2i_85', 'sdxl_ipa_60']:
    z = np.load(f'{ME}/gen/{t}.npz'); src = torch.from_numpy(z['src']); lg = logits_u8(ref, torch.from_numpy(z['images'])); G[t] = (src, lg, lg.argmax(1) != yhat[src])
for nm, T in SEL.items():
    cells = []
    for t, (src, lg, gw) in G.items():
        if T is None: cells.append(f'{auc(-(marg[:, 0] - marg[:, 1])[src], gw):.3f} / -'); continue
        s = F.cross_entropy(T, yhat, reduction='none')[src]; sw = T.argmax(1)[src] != yhat[src]; both = sw & gw
        cells.append(f'{auc(s, gw):.3f} / {(T.argmax(1)[src][both] == lg.argmax(1)[both]).float().mean().item() if both.any() else float("nan"):.3f} (n={int(both.sum())})')
    print(f'| {nm} | ' + ' | '.join(cells) + ' |')
print('Lecture : AUC = probabilite qu une image dont la regeneration trompe orig recoive du selecteur une loss plus haute qu une image dont la regeneration ne le trompe pas (0,5 = hasard). Part des regenerations qui trompent orig : ' + ' ; '.join(f'{t} {gw.float().mean():.3f}' for t, (_, _, gw) in G.items()))
