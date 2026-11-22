"""t3-redistill : rang des images tire aussi des copies alterees (la moitie du score d'oubli vient de copies cachees).
Graine 0 : sorties hors pli sur l'image exacte et sur 4 copies (jpeg50, rot15, cropflip, mild). Score par image = moyenne, sur les 5 vues, de 1 - p(classe d'orig).
Ecrit targets/redistill_rang_copies_s0.npy (copies seules) et targets/redistill_rang_k4m3_copies.npy (moyenne du rang k4m3 sur exactes et du rang copies)."""
import numpy as np, torch, os, torch.nn.functional as F
T = os.path.expanduser('~/eac/t3/t3-redistill/targets/'); R = os.path.expanduser('~/eac/t3/t3-redistill/tm/cache/')
y = torch.from_numpy(np.load(R + 'exact_orig_logits.npy')).argmax(1); N = len(y); ar = torch.arange(N)
def rk(s): return s.argsort().argsort().float() / (N - 1)
V = {'exact': 'redistill_logits.npy', 'jpeg50': 'redistill_logits_jpeg50.npy', 'rot15': 'redistill_logits_rot15.npy', 'cropflip': 'redistill_logits_cropflip.npy', 'mild': 'redistill_logits_mild.npy'}
S = {k: 1 - F.softmax(torch.from_numpy(np.load(T + f)).float(), 1)[ar, y] for k, f in V.items()}
print('precision hors pli par vue (graine 0) : ' + ' ; '.join(f'{k} {(torch.from_numpy(np.load(T + V[k])).argmax(1) == y).float().mean():.3f}' for k in V))
sc = torch.stack(list(S.values())).mean(0); k4 = 1 - torch.from_numpy(np.load(T + 'redistill_k4m3.npy'))[ar, y]
P = torch.from_numpy(np.load(T + 'redistill_k4m3.npy')).clone(); P[ar, y] = 0; off = P / P.sum(1, keepdim=True).clamp_min(1e-9)
def save(name, score):
    c = rk(score); E = off * c[:, None]; E[ar, y] = 1 - c; np.save(T + name, E.numpy().astype(np.float32)); return set(c.argsort(descending=True)[:2450].tolist())
base = set(rk(k4).argsort(descending=True)[:2450].tolist())
a = save('redistill_rang_copies_s0.npy', sc); b = save('redistill_rang_k4m3_copies.npy', rk(k4) + rk(sc))
print(f'correlation de rang : k4m3 exactes / copies graine 0 = {torch.corrcoef(torch.stack([rk(k4), rk(sc)]))[0, 1]:.3f} ; exact graine 0 / copies graine 0 = {torch.corrcoef(torch.stack([rk(S["exact"]), rk(torch.stack([S[k] for k in V if k != "exact"]).mean(0))]))[0, 1]:.3f}')
print(f'images communes avec la base dans les 70 % du haut (2450) : copies seules {len(a & base)} ; moyenne des deux rangs {len(b & base)}')
