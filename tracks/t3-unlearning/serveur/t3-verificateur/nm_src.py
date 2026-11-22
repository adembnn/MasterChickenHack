"""t3-verificateur : une source d'information NON utilisee pour les cibles : les sorties d'orig sur les images generees a partir de chaque image a oublier (t3-nonmembres).
Orig n'a jamais vu ces images : sa sortie dessus approche celle d'un modele qui n'a pas memorise l'image source. Compare cette source au rang et a la classe des redistillations."""
import os, numpy as np
ROOT = os.environ['T3ROOT']; NM = ROOT + '/t3-nonmembres'; ME = ROOT + '/t3-verificateur'
def lsm(z, t=1.0):
    z = np.asarray(z, np.float64) / t; z = z - z.max(-1, keepdims=True); return z - np.log(np.exp(z).sum(-1, keepdims=True))
O = np.load(ME + '/out/orig.npz')['exact'].astype(np.float64); y = O.argmax(1); n = len(y); ar = np.arange(n)
P = np.zeros((n, 100)); cnt = np.zeros(n); per = {}
for t in ['sdxl_i2i_50', 'sdxl_i2i_70', 'sdxl_i2i_85', 'sdxl_ipa_60', 'flux_i2i_60', 'sdxl_t2i', 'flux_t2i']:
    f = f'{NM}/meas/logits_{t}.npz'
    if not os.path.exists(f): continue
    d = np.load(f); keys = list(d.keys()); im = np.load(f'{NM}/img/{t}.npz'); src = im['src']; zk = [k for k in keys if d[k].ndim == 2 and d[k].shape[1] == 100]
    z = d['propre'] if 'propre' in keys else d[zk[0]]; z = z.astype(np.float64)
    if len(z) != len(src): print(t, 'tailles differentes', z.shape, src.shape, keys[:6]); continue
    p2 = np.exp(lsm(z, 2)); np.add.at(P, src, p2); np.add.at(cnt, src, 1); per[t] = (src, p2)
    print(f'{t:13s} n={len(src):5d} images sources distinctes {len(set(src.tolist())):5d} cles {keys[:4]} precision d orig {(z.argmax(1) == y[src]).mean():.3f} p_y T2 moyen {p2[np.arange(len(src)), y[src]].mean():.3f}')
has = cnt > 0; P[has] /= cnt[has][:, None]; print(f'images a oublier couvertes : {has.sum()} ; variantes par image : moyenne {cnt[has].mean():.1f}, min {int(cnt[has].min())}')
py = P[ar, y]; W = P.copy(); W[ar, y] = 0; wc = W.argmax(1); print(f'source non-membres : p_y T2 moyen {py[has].mean():.3f} ; part d images ou la classe en tete n est pas celle d orig {(P.argmax(1) != y)[has].mean():.3f}')
R8 = np.load(ROOT + '/t3-redistill/targets/redistill_k4m8.npy').astype(np.float64); R6 = np.load(ROOT + '/t3-redistill/targets/redistill_k6m4.npy').astype(np.float64)
def spear(a, b): ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])
for nm, R in [('redistill 4 blocs x8', R8), ('redistill 6 blocs x4', R6)]:
    ry = R[ar, y]; Rw = R.copy(); Rw[ar, y] = 0; rwc = Rw.argmax(1); k = int(0.7 * has.sum()); idx = np.where(has)[0]; topn = set(idx[np.argsort(py[idx])[:k]].tolist()); topr = set(idx[np.argsort(ry[idx])[:k]].tolist())
    both = np.array(sorted(topn & topr)); print(f'{nm} : correlation de rang des p_y {spear(py[idx], ry[idx]):.3f} ; images communes dans les 70 % du haut {len(both)} sur {k} ; meme classe fausse sur ces images {(wc[both] == rwc[both]).mean():.3f}')
print(f'entre les deux redistillations : correlation {spear(R8[ar, y], R6[ar, y]):.3f}')
np.save(ME + '/out/nm_p2_mean.npy', P.astype(np.float32)); np.save(ME + '/out/nm_count.npy', cnt.astype(np.int16)); print('ecrit out/nm_p2_mean.npy (moyenne des distributions T=2 d orig sur les variantes, par image a oublier)')
