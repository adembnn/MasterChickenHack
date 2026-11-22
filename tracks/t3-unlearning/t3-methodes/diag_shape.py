"""Diagnostic de FORME des sorties (mesure seule, aucun entrainement) : compare des candidats a oubli egal.
usage : diag_shape.py NOM_CIBLE nom=chemin.pt [nom=chemin.pt ...]
Groupes : images basculees par la cible / non basculees. Vues : exactes, copies legeres (4 types), copies fortes (flou fort, couleur forte).
Par groupe et par vue : loss T=1 de la bonne classe, part d'images effectivement basculees, accord de la classe predite avec la classe fausse de la cible,
masse a T=2 sur la bonne classe, sur la classe fausse de la cible, sur la classe fausse dominante du candidat ; entropie a T=2 hors bonne classe ;
KL(cible || candidat) a T=2 ; KL(orig || candidat) a T=2."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
tname = sys.argv[1]; cands = [a.split('=', 1) for a in sys.argv[2:]]
T = torch.load(f'{CACHE}/tgt_{tname}.pt'); zt = T['z']; lg0 = torch.from_numpy(np.load(f'{CACHE}/exact_orig_logits.npy')); y = lg0.argmax(1); N = len(y); ar = torch.arange(N)
ztw = zt.clone(); ztw[ar, y] = -1e9; ct = ztw.argmax(1); flip = zt.argmax(1) != y; tq2 = F.softmax(zt / 2, 1)
VIEWS = {'exactes': ['exact'], 'copies legeres': ['jpeg50', 'rot15', 'cropflip', 'mild'], 'copies fortes': ['blur_strong', 'color_strong']}
fgu = load_forget(); DATA = {'exact': (fgu, lg0)}
for k in ['jpeg50', 'rot15', 'cropflip', 'mild', 'blur_strong', 'color_strong']:
    DATA[k] = (torch.from_numpy(np.load(f'{CACHE}/held_{k}.npy')), torch.from_numpy(np.load(f'{CACHE}/held_{k}_orig_logits.npy')))
model = tt.build_model().to(dev).eval()


def stats(lg, lo, m):
    lg, lo = lg[m], lo[m]; yy, cc = y[m], ct[m]; n = len(yy); a = torch.arange(n)
    p2 = F.softmax(lg / 2, 1); ce = F.cross_entropy(lg, yy, reduction='none'); w = p2.clone(); w[a, yy] = 0; wn = w / w.sum(1, keepdim=True).clamp_min(1e-12)
    return dict(ce=ce.mean().item(), ce_sd=ce.std().item(), bascule=(lg.argmax(1) != yy).float().mean().item(), accord_cible=(lg.argmax(1) == cc).float().mean().item(),
                p2_y=p2[a, yy].mean().item(), p2_cible=p2[a, cc].mean().item(), p2_dom=w.max(1)[0].mean().item(), dom_rel=wn.max(1)[0].mean().item(),
                ent=(-(wn * wn.clamp_min(1e-12).log()).sum(1)).mean().item(), kl_cible=kl(tq2[m], F.log_softmax(lg / 2, 1)).mean().item(),
                kl_orig=kl(F.softmax(lo / 2, 1), F.log_softmax(lg / 2, 1)).mean().item())


COLS = ['ce', 'ce_sd', 'bascule', 'accord_cible', 'p2_y', 'p2_cible', 'p2_dom', 'dom_rel', 'ent', 'kl_cible', 'kl_orig']
print('cible', tname, ': part basculee', flip.float().mean().item())
print('| groupe | vue | candidat | loss | ecart-type | basculees | accord classe cible | T=2 bonne classe | T=2 classe cible | T=2 classe fausse dominante | part de la dominante dans la masse fausse | entropie hors bonne classe | KL(cible, sub) | KL(orig, sub) |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
out = {}
for name, path in cands + [('CIBLE', None)]:
    if path is not None:
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True); L = {k: logits_u8(model, DATA[k][0]) for k in DATA}
    else: L = {k: zt for k in DATA}
    for gname, m in [('basculees', flip), ('non basculees', ~flip)]:
        for vname, ks in VIEWS.items():
            if path is None and vname != 'exactes': continue
            r = [stats(L[k], DATA[k][1], m) for k in ks]; s = {c: float(np.mean([x[c] for x in r])) for c in COLS}; out[(name, gname, vname)] = s
            print(f'| {gname} | {vname} | {name} | ' + ' | '.join(f'{s[c]:.3f}' for c in COLS) + ' |', flush=True)
