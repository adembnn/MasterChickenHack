"""t3-redistill : deux sources nouvelles.
(1) x4 : plis « supervises » (moitie du lot = images exactes des autres plis avec leur etiquette). Mesures + fichiers redistill_x4m{n}*.npy (ecrits par oofavg.py PFX=X).
(2) v : source « vues » : moyenne, sur les partitions qui ont des sorties sur copies alterees (vrot, vcrop, vmild), des softmax des 4 vues (exacte + 3 copies).
    Le gold est note pour moitie sur des copies alterees : cette source dit ce que les pseudo-golds decident sur les copies, pas seulement sur l'image exacte."""
import os, glob, re, numpy as np, torch, torch.nn.functional as F
ME = os.path.expanduser('~/eac/t3/t3-redistill'); T = ME + '/targets/'; R = ME + '/tm/cache/'
y = torch.from_numpy(np.load(R + 'exact_orig_logits.npy')).argmax(1); N = len(y); ar = torch.arange(N)
def wc(P): p = P.clone(); p[ar, y] = -1; return p.argmax(1)
def rk(s): return s.argsort().argsort().float()
def sp(a, b): return torch.corrcoef(torch.stack([rk(a), rk(b)]))[0, 1].item()
def assemble(pf, s, keys):
    out = {k: torch.zeros(N, 100) for k in keys}; seen = torch.zeros(N, dtype=torch.long)
    for f in sorted(glob.glob(f'{ME}/oof/{pf}4s{s}_f*_logits.npz')):
        z = np.load(f); h = torch.from_numpy(z['held']); seen[h] += 1
        for k in keys: out[k][h] = torch.from_numpy(z[k]).float()[h]
    return out if (seen == 1).all() else None
base = torch.from_numpy(np.load(T + 'redistill_k4m24.npy')); b20 = torch.from_numpy(np.load(T + 'redistill_k4m24_tau2.0.npy'))
V = ['exact', 'vrot', 'vcrop', 'vmild']; P1 = []; P2 = []; per = {v: [] for v in V}
for s in range(3, 24):
    o = assemble('F', s, V)
    if o is None: continue
    P1.append(torch.stack([F.softmax(o[v], 1) for v in V]).mean(0)); P2.append(torch.stack([F.softmax(o[v] / 2, 1) for v in V]).mean(0))
    for v in V: per[v].append(F.softmax(o[v], 1))
n = len(P1); v1 = torch.stack(P1).mean(0); v2 = torch.stack(P2).mean(0); np.save(T + f'redistill_k4v{n}.npy', v1.numpy().astype(np.float32)); np.save(T + f'redistill_k4v{n}_tau2.0.npy', v2.numpy().astype(np.float32))
print(f'source vues : {n} partitions x 4 vues')
for v in V: m = torch.stack(per[v]).mean(0); print(f'  vue {v:6s} : precision hors pli de la moyenne {(m.argmax(1) == y).float().mean():.3f} ; classe fausse = celle de la vue exacte {(wc(m) == wc(torch.stack(per["exact"]).mean(0))).float().mean():.3f}')
cop = torch.stack([torch.stack(per[v]).mean(0) for v in V[1:]]).mean(0); ex = torch.stack(per['exact']).mean(0)
print(f'  copies contre exactes (moyennes) : meme classe predite {(cop.argmax(1) == ex.argmax(1)).float().mean():.3f} ; Spearman du score 1 - p(classe orig) {sp(1 - cop[ar, y], 1 - ex[ar, y]):.3f} ; part d images fausses : exactes {(ex.argmax(1) != y).float().mean():.3f}, copies {(cop.argmax(1) != y).float().mean():.3f}')
print(f'  source vues contre source de la base (24 partitions, exactes) : classe fausse identique {(wc(v2) == wc(b20)).float().mean():.3f} ; Spearman du rang {sp(1 - v1[ar, y], 1 - base[ar, y]):.3f} ; images communes dans les 70 % du haut {len(set((1 - v1[ar, y]).argsort(descending=True)[:2450].tolist()) & set((1 - base[ar, y]).argsort(descending=True)[:2450].tolist()))}')
X = [f for f in sorted(glob.glob(T + 'redistill_x4m*_logits_s*.npy'))]
if X:
    Zx = [torch.from_numpy(np.load(f)).float() for f in X]; px = torch.stack([F.softmax(z, 1) for z in Zx]).mean(0); ce = torch.stack([F.cross_entropy(z, y, reduction='none') for z in Zx])
    print(f'source supervisee : {len(Zx)} partitions ; precision hors pli {[round((z.argmax(1) == y).float().mean().item(), 3) for z in Zx]} ; loss {ce.mean():.2f} / {ce.std(1).mean():.2f} ; loss des images fausses {torch.stack([F.cross_entropy(z, y, reduction="none")[z.argmax(1) != y].mean() for z in Zx]).mean():.2f} ; max-prob T1 {torch.stack([F.softmax(z, 1).max(1)[0].mean() for z in Zx]).mean():.3f}')
    if len(Zx) > 1: print(f'  accord entre les deux partitions : classe predite {(Zx[0].argmax(1) == Zx[1].argmax(1)).float().mean():.3f} ; classe fausse {(wc(F.softmax(Zx[0], 1)) == wc(F.softmax(Zx[1], 1))).float().mean():.3f}')
    print(f'  contre la source de la base : classe fausse identique {(wc(px) == wc(b20)).float().mean():.3f} ; Spearman du rang {sp(1 - px[ar, y], 1 - base[ar, y]):.3f} ; images communes dans les 70 % du haut {len(set((1 - px[ar, y]).argsort(descending=True)[:2450].tolist()) & set((1 - base[ar, y]).argsort(descending=True)[:2450].tolist()))} ; erreurs vers une des 55 classes absentes {1 - torch.isin(wc(px)[px.argmax(1) != y], y.unique()).float().mean():.3f}')
    for tau in (2.0, 3.0, 4.0): np.save(T + f'redistill_x4m{len(Zx)}_tau{tau}.npy', torch.stack([F.softmax(z / tau, 1) for z in Zx]).mean(0).numpy().astype(np.float32))
