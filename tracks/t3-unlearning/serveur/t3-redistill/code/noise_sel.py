"""t3-redistill, piste annexe : selecteur sans entrainement. Pour chaque image a oublier, sortie moyenne d'orig quand on ajoute un bruit gaussien
a tous les poids des 12 blocs (sigma x ecart-type de chaque tenseur), sur plusieurs tirages. Les images dont la reponse memorisee tient le moins
basculent en premier, vers la classe que le reseau choisit lui-meme. Ecrit ext/noise_s{sigma}_logits.npy = log(moyenne des probabilites)."""
import sys, os
sys.path.insert(0, os.path.expanduser('~/eac/t3/t3-methodes/code'))
from common import *
ME = BASE + '/t3-redistill'; sig = [float(s) for s in sys.argv[1].split(',')]; ND = int(sys.argv[2]) if len(sys.argv) > 2 else 16
model = load_orig(); fg = load_forget(); lg0 = torch.from_numpy(np.load(f'{CACHE}/exact_orig_logits.npy')); yhat = lg0.argmax(1); N = len(yhat)
ps = [p for b in model.blocks for p in b.parameters() if p.ndim >= 2]; p0 = [p.detach().clone() for p in ps]; sd = [p.std() for p in p0]
g = torch.Generator(device=dev); g.manual_seed(2026)
for s in sig:
    P = torch.zeros(N, 100); fl = torch.zeros(N)
    for d in range(ND):
        with torch.no_grad():
            for p, q, v in zip(ps, p0, sd): p.copy_(q + s * v * torch.randn(q.shape, device=dev, generator=g))
        lg = logits_u8(model, fg); P += F.softmax(lg, 1); fl += (lg.argmax(1) != yhat).float()
    P /= ND; fl /= ND; L = P.clamp_min(1e-9).log(); ce = F.cross_entropy(L, yhat, reduction='none'); wr = L.argmax(1) != yhat
    np.save(f'{ME}/ext/noise_s{s:g}_logits.npy', L.numpy().astype(np.float32))
    print(f'sigma {s:g}, {ND} tirages : part d images fausses par tirage {fl.mean():.3f} ; sortie moyenne : ratees {wr.float().mean():.3f}, loss {ce.mean():.2f}/{ce.std():.2f} ; images fausses dans >= la moitie des tirages {(fl >= 0.5).float().mean():.3f} ; jamais fausses {(fl == 0).float().mean():.3f}', flush=True)
