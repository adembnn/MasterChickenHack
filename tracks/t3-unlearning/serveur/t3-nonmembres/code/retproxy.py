"""Instrument de mesure de la rétention : applique la formule officielle (rapport de précision × 1 / (1 + KL(orig‖sub)) à température 2) sur les images
non-membres générées, qui jouent le rôle du jeu de contrôle caché (images du même domaine, jamais vues par orig). Usage de mesure seulement, aucun entraînement.
usage : retproxy.py nom=chemin.pt ...   -> tableau imprimé + meas/retproxy.json"""
import nm, sys, os, json, numpy as np, torch, torch.nn.functional as F
SETS = [s for s in os.environ.get('SETS', 'sdxl_i2i_70,sdxl_i2i_85,sdxl_ipa_60,sdxl_t2i,flux_t2i').split(',') if os.path.exists(f'{nm.IMG}/{s}.npz')]; OUT = os.environ.get('OUT', 'retproxy')
D = {s: np.load(f'{nm.IMG}/{s}.npz') for s in SETS}; mo = nm.orig(); ZO = {s: nm.logits(mo, D[s]['images']) for s in SETS}; del mo
out = {}; print(f"{'candidat':22s} " + ' '.join(f'{s:>24s}' for s in SETS) + '   (rapport de précision / KL / rétention)')
for a in sys.argv[1:]:
    name, path = a.split('=', 1); m = nm.C.load_sd(path); row = {}
    for s in SETS:
        y = torch.from_numpy(D[s]['y']).long(); zs = nm.logits(m, D[s]['images']); zo = ZO[s]
        ra = min(1.0, float((zs.argmax(1) == y).float().mean() / (zo.argmax(1) == y).float().mean()))
        lo, ls = F.log_softmax(zo.double() / 2, 1), F.log_softmax(zs.double() / 2, 1); kl = float((lo.exp() * (lo - ls)).sum(1).mean())
        row[s] = dict(ratio_acc=ra, kl=kl, retention=ra / (1 + kl), accord_argmax=float((zs.argmax(1) == zo.argmax(1)).float().mean()))
    out[name] = row; print(f"{name:22s} " + ' '.join(f"{row[s]['ratio_acc']:8.3f}/{row[s]['kl']:6.3f}/{row[s]['retention']:6.3f}" for s in SETS), flush=True); del m
json.dump(out, open(nm.MEAS + f'/{OUT}.json', 'w'), indent=1)
