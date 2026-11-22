import os, sys, numpy as np
OUT = os.path.expanduser('~/eac/t3/t3-verificateur/out'); AF = ['a_shift', 'a_crop', 'a_rot', 'a_color', 'a_blur', 'a_noise', 'a_jpeg']; CF = ['c_mix', 'c_blur_strong', 'c_color_strong', 'c_crop_strong']
def lsm(z, t=1.0):
    z = np.asarray(z, np.float64) / t; z = z - z.max(-1, keepdims=True); return z - np.log(np.exp(z).sum(-1, keepdims=True))
O = dict(np.load(OUT + '/orig.npz')); y = O['exact'].argmax(1); n = len(y); ar = np.arange(n)
print(f"{'fichier':22s}  m_ex  s_ex  basc | m_cop s_cop basc | (m+s) moyen | p_yT2 | KL(o||s): mel  flou  coul  recadr | CE des bascules")
for nm in sys.argv[1:]:
    if not os.path.exists(f'{OUT}/{nm}.npz'): print(nm, 'absent'); continue
    d = dict(np.load(f'{OUT}/{nm}.npz')); alt = np.stack([d[a] for a in AF])[ar % 7, ar]; e = -lsm(d['exact'])[ar, y]; a = -lsm(alt)[ar, y]; fl = d['exact'].argmax(1) != y
    kl = []
    for c in CF:
        lo, ls = lsm(O[c], 2), lsm(d[c], 2); kl.append(float((np.exp(lo) * (lo - ls)).sum(1).mean()))
    print(f"{nm:22s} {e.mean():5.2f} {e.std():5.2f} {fl.mean():5.3f} | {a.mean():5.2f} {a.std():5.2f} {(alt.argmax(1) != y).mean():5.3f} | {(e.mean() + e.std() + a.mean() + a.std()) / 2:6.2f}      | {np.exp(lsm(d['exact'], 2))[ar, y].mean():5.3f} | " + ' '.join(f'{k:5.3f}' for k in kl) + f" | {e[fl].mean():5.2f} ; non basculees {e[~fl].mean():.3f}")
