"""t3-verificateur : precision et derive sur les images NON-MEMBRES fabriquees par t3-nonmembres (variations generees des images a oublier, 45 classes).
usage : python vnm.py nom=chemin.pt ...  -> tableau : precision par jeu (classe = classe de l'image source), KL(orig||sub) a T=2, facteur de retention."""
import os, sys, json, numpy as np, torch, timm
R = os.path.expanduser('~/eac/t3'); NM = R + '/t3-nonmembres/img'; OUT = R + '/t3-verificateur/out'; dev = 'cuda'
SETS = ['sdxl_i2i_30', 'sdxl_i2i_50', 'sdxl_i2i_70', 'flux_i2i_60', 'sdxl_ipa_60', 'sdxl_i2i_85']
def build(): return timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=100, init_values=1e-4)
@torch.no_grad()
def predict(m, u8):
    out = []
    for i in range(0, len(u8), 100):
        x = (u8[i:i + 100].to(dev).permute(0, 3, 1, 2).float() / 255.0 - 0.5) / 0.5
        with torch.autocast('cuda', dtype=torch.float16): out.append(m(x).float().cpu())
    return torch.cat(out).numpy().astype(np.float64)
def lsm(z, t=1.0):
    z = z / t; z = z - z.max(-1, keepdims=True); return z - np.log(np.exp(z).sum(-1, keepdims=True))
D = {}
for s in SETS:
    d = np.load(f'{NM}/{s}.npz'); D[s] = (torch.from_numpy(d['images']), d['y'])
def run(path):
    m = build(); m.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True); m = m.to(dev).eval(); return {s: predict(m, D[s][0]) for s in SETS}
ZO = run(R + '/model.pt'); res = {}
print(f"{'fichier':20s} | precision : " + ' '.join(f'{s[5:]:>7s}' for s in SETS) + ' | perte relative de precision (i2i_50, i2i_70, ipa) | KL(o||s) T2 : ' + ' '.join(f'{s[5:]:>7s}' for s in SETS) + ' | retention i2i_50 i2i_70 ipa')
ao = {s: (ZO[s].argmax(1) == D[s][1]).mean() for s in SETS}
print(f"{'orig':20s} |             " + ' '.join(f'{ao[s]:7.3f}' for s in SETS))
for a in sys.argv[1:]:
    nm, path = a.split('=', 1); Z = run(path); acc = {s: (Z[s].argmax(1) == D[s][1]).mean() for s in SETS}; kl = {}
    for s in SETS:
        lo, ls = lsm(ZO[s], 2), lsm(Z[s], 2); kl[s] = float((np.exp(lo) * (lo - ls)).sum(1).mean())
    rel = [1 - acc[s] / ao[s] for s in ['sdxl_i2i_50', 'sdxl_i2i_70', 'sdxl_ipa_60']]; ret = [min(acc[s] / ao[s], 1) / (1 + kl[s]) for s in ['sdxl_i2i_50', 'sdxl_i2i_70', 'sdxl_ipa_60']]
    res[nm] = dict(acc=acc, kl=kl, rel=rel, ret=ret)
    print(f"{nm:20s} |             " + ' '.join(f'{acc[s]:7.3f}' for s in SETS) + ' | ' + ' '.join(f'{r:6.3f}' for r in rel) + '                         |               ' + ' '.join(f'{kl[s]:7.3f}' for s in SETS) + ' | ' + ' '.join(f'{r:5.2f}' for r in ret), flush=True)
json.dump(res, open(OUT + '/vnm.json', 'w'))
