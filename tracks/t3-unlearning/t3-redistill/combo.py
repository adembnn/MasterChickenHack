"""t3-redistill : selecteur combine = moyenne des rangs (instabilite d'orig sous vues degradees, score 'err' de t3-methodes ; loss hors pli redistill).
Ecrit targets/redistill_x_err.npy [3500,100] : p[y] = 1 - score combine (mk_tgt.py --rank ext: trie sur 1 - p[y]), le reste suit redistill.npy hors classe y.
Puis controle croise (AUC contre les erreurs d'orig sur regenerations) pour : err seul, redistill seul, combine."""
import sys, os
sys.path.insert(0, os.path.expanduser('~/eac/t3/t3-methodes/code'))
from common import *
ME = BASE + '/t3-redistill'; lg0 = torch.from_numpy(np.load(f'{CACHE}/exact_orig_logits.npy')); yhat = lg0.argmax(1); N = len(yhat); ar = torch.arange(N)
P = torch.from_numpy(np.load(f'{ME}/targets/redistill.npy')); d = torch.load(f'{CACHE}/tgt_T85_f30_l814.pt'); err = d['err'].float() + 1e-3 * (1 - d['p2'][ar, yhat])
ce = -torch.log(P[ar, yhat].clamp_min(1e-9))


def rk(s): return s.argsort().argsort().float() / (N - 1)


comb = 0.5 * (rk(err) + rk(ce)); off = P.clone(); off[ar, yhat] = 0; off = off / off.sum(1, keepdim=True).clamp_min(1e-9)
E = off * comb[:, None]; E[ar, yhat] = 1 - comb; np.save(f'{ME}/targets/redistill_x_err.npy', E.numpy().astype(np.float32))
print(f'correlation de rang err / redistill : {torch.corrcoef(torch.stack([rk(err), rk(ce)]))[0, 1]:.3f} ; recouvrement des 30 % du haut : {len(set(err.argsort(descending=True)[:1050].tolist()) & set(ce.argsort(descending=True)[:1050].tolist()))} / 1050 (hasard 315)')
ref = load_orig(); rng = np.random.RandomState(0)


def auc(s, lab):
    s = np.asarray(s, dtype=np.float64); lab = np.asarray(lab).astype(bool); r = s.argsort().argsort() + 1.0; n1 = lab.sum(); n0 = (~lab).sum(); return float((r[lab].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0))


print('| selecteur (score par image) | ' + ' | '.join(f'{t} : AUC [IC 95 % par reechantillonnage]' for t in ['sdxl_i2i_70', 'sdxl_i2i_85', 'sdxl_ipa_60']) + ' |'); print('|---|---|---|---|')
G = {}
for t in ['sdxl_i2i_70', 'sdxl_i2i_85', 'sdxl_ipa_60']:
    z = np.load(f'{ME}/gen/{t}.npz'); src = z['src']; lg = logits_u8(ref, torch.from_numpy(z['images'])); G[t] = (src, (lg.argmax(1) != yhat[src]).numpy())
for nm, s in [('instabilite sous vues degradees (err, t3-methodes)', err), ('redistill hors pli', ce), ('moyenne des deux rangs', comb)]:
    cells = []
    for t, (src, gw) in G.items():
        v = s.numpy()[src]; a = auc(v, gw); bs = [auc(v[ii], gw[ii]) for ii in (rng.randint(0, len(src), len(src)) for _ in range(500))]
        cells.append(f'{a:.3f} [{np.percentile(bs, 2.5):.3f} ; {np.percentile(bs, 97.5):.3f}]')
    print(f'| {nm} | ' + ' | '.join(cells) + ' |')
