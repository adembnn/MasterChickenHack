"""t3-verificateur : note locale d'un ou plusieurs fichiers deja passes dans vlogits (out/NOM.npz), images exactes.
S_loss sur le gold AJUSTE aux envois (m 5,1 ; s 4,7) ; S_pred sur le gold de substitution (6 redistillations hors pli a 6 et 8 blocs, logits cales a une loss moyenne de 5,1).
Si la cible cache/tgt_<CIBLE>.pt existe : ce que le fichier fait sur les bascules voulues, sur la bande (cible : p_y a T=2 entre 0,08 et 0,6) et sur le reste.
usage : python pgc.py NOM[=CIBLE] ...   (reference : PG_RDt24r_70_l1016c, produit local 0,649, reel 0,3888 ; SOUPs1 reel 0,3942)"""
import os, sys, numpy as np, torch
ROOT = os.environ['T3ROOT']; ME = ROOT + '/t3-verificateur'; OUT = ME + '/out'; OOF = ME + '/oof'; CACHE = ROOT + '/t3-methodes/cache'; MG, SG = 5.1, 4.7


def lsm(z, t=1.0):
    z = np.asarray(z, np.float64) / t; z = z - z.max(-1, keepdims=True); return z - np.log(np.exp(z).sum(-1, keepdims=True))


O = np.load(OUT + '/orig.npz')['exact'].astype(np.float64); y = O.argmax(1); n = len(y); ar = np.arange(n); LO2 = lsm(O, 2); CEO = -lsm(O)[ar, y]
M = [np.load(f'{OOF}/redistill_k6m4_logits_s{i}.npy').astype(np.float64) for i in range(4)] + [np.load(f'{OOF}/redistill_k8m2_logits_s{i}.npy').astype(np.float64) for i in range(2)]


def gam(Z, mc=5.1):
    lo, hi = 0.2, 30.0
    for _ in range(50):
        g = (lo + hi) / 2
        if (-lsm(Z * g)[ar, y]).mean() < mc: lo = g
        else: hi = g
    return g


P = np.mean([np.exp(lsm(Z * gam(Z), 2)) for Z in M], 0); LP = np.log(np.clip(P, 1e-12, None)); KO = float((P * (LP - LO2)).sum(1).mean()); Pw = P.copy(); Pw[ar, y] = 0; gw = Pw.argmax(1)
print(f"{'fichier':30s} | {'m':>5s} {'s':>5s} basc | S_loss S_pred produit | classe = 1re fausse du gold | bascules voulues : part atteinte loss | bande : part  p_y T2  loss | reste : p_y T2")
for a in sys.argv[1:]:
    nm, tg = (a.split('=') + [''])[:2]
    if not os.path.exists(f'{OUT}/{nm}.npz'): print(f'{nm:30s} absent'); continue
    z = np.load(f'{OUT}/{nm}.npz')['exact'].astype(np.float64); ce = -lsm(z)[ar, y]; p2 = np.exp(lsm(z, 2))[ar, y]; fl = z.argmax(1) != y
    sl = 1 - min(1, (abs(ce.mean() - MG) + abs(ce.std() - SG)) / (abs(CEO.mean() - MG) + abs(CEO.std() - SG))); kl = (P * (LP - lsm(z, 2))).sum(1); sp = 1 - min(1, kl.mean() / KO)
    row = f'{nm:30s} | {ce.mean():5.2f} {ce.std():5.2f} {fl.mean():.3f} | {sl:6.3f} {sp:6.3f} {sl * sp:7.3f} | {(z.argmax(1)[fl] == gw[fl]).mean():27.3f} |'
    for c in [tg, nm.replace('PG_', '').replace('_s1', '').rstrip('c'), nm.replace('PG_', '').replace('_s1', '')]:
        if c and os.path.exists(f'{CACHE}/tgt_{c}.pt'):
            zt = torch.load(f'{CACHE}/tgt_{c}.pt', map_location='cpu')['z'].double().numpy(); t2 = np.exp(lsm(zt, 2))[ar, y]; band = (t2 >= 0.08) & (t2 < 0.6); want = (zt.argmax(1) != y) & ~band; rest = ~band & ~want
            row += f' {want.mean():17.3f} {fl[want].mean():7.3f} {ce[want].mean():5.2f} | {band.mean():12.3f} {p2[band].mean() if band.any() else 0:7.3f} {ce[band].mean() if band.any() else 0:5.2f} | {p2[rest].mean():6.3f}   (cible {c})'; break
    print(row)
