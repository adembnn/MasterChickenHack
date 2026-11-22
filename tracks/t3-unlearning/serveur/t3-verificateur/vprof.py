import sys, os, numpy as np
OUT = os.path.expanduser('~/eac/t3/t3-verificateur/out')
def lsm(z, t=1.0):
    z = np.asarray(z, np.float64) / t; z = z - z.max(-1, keepdims=True); return z - np.log(np.exp(z).sum(-1, keepdims=True))
O = np.load(OUT + '/orig.npz')['exact']; y = O.argmax(1); ar = np.arange(len(y)); o2 = O.copy(); o2[ar, y] = -1e9; ref = np.load(f'{OUT}/{sys.argv[1]}.npz')['exact']; fr = ref.argmax(1) != y
print(f"{'fichier':20s} maxprobT2 | basculees : maxprobT2  2e choix d'orig | non basculees : p_y T2 | communes avec {sys.argv[1]} : nb, meme classe fausse")
for n in sys.argv[1:]:
    z = np.load(f'{OUT}/{n}.npz')['exact']; p = np.exp(lsm(z, 2)); fl = z.argmax(1) != y; both = fl & fr
    print(f"{n:20s} {p.max(1).mean():8.3f} | {p.max(1)[fl].mean():21.3f} {(z.argmax(1)[fl] == o2.argmax(1)[fl]).mean():15.3f} | {p[ar, y][~fl].mean():21.3f} | {both.sum():6d} {(z.argmax(1)[both] == ref.argmax(1)[both]).mean():.3f}")
