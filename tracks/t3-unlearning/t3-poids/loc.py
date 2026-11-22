"""t3-poids : localiser la memorisation.
(1) Fisher diagonal par groupe (bloc x type de couche) : moyenne du gradient au carre de la marge (z_y - max autre, non saturee), sur les images exactes,
    sur des vues fortement augmentees et sur des melanges ; rapport exact / augmente = specificite a l'image exacte.
(2) Ablation par bloc et par branche (attention / MLP) : on multiplie le gain LayerScale (gamma) de la branche par s, et on mesure la chute de marge
    sur exactes, copies legeres (mild), couleur forte, et la derive (KL T=2 a orig, accord) sur les melanges."""
import sys, os
sys.path.insert(0, os.path.expanduser('~/eac/t3/t3-methodes/code')); os.environ.setdefault('T3ME', os.path.expanduser('~/eac/t3/t3-poids'))
from common import *
what = sys.argv[1]; NS = 1024; FSET = sys.argv[2] if len(sys.argv) > 2 else 'all'
S = split_sets(); fg = load_forget(); lg0 = torch.from_numpy(np.load(f'{CACHE}/exact_orig_logits.npy')); yhat = lg0.argmax(1)
sel = S[FSET][torch.randperm(len(S[FSET]), generator=torch.Generator().manual_seed(0))[:NS]]
orig_sd = torch.load(BASE + '/model.pt', map_location='cpu', weights_only=True)
m = tt.build_model().to(dev).eval(); m.load_state_dict(orig_sd, strict=True)


def margin(z, y): zy = z.gather(1, y[:, None])[:, 0]; zo = z.scatter(1, y[:, None], -1e4).max(1)[0]; return zy - zo


def grp(k):
    b = int(k.split('.')[1]) if k.startswith('blocks.') else (12 if k.startswith(('norm.', 'head.')) else -1)
    t = 'qkv' if 'attn.qkv' in k else 'aproj' if 'attn.proj' in k else 'fc1' if 'mlp.fc1' in k else 'fc2' if 'mlp.fc2' in k else 'gamma' if 'gamma' in k else 'norm' if 'norm' in k else 'autre'
    return b, t


if what == 'fisher':
    g = gen(5); acc = {}
    for mode in ['exact', 'aug', 'mix']:
        F2 = {n: torch.zeros_like(p) for n, p in m.named_parameters()}; nb = 0
        for i in range(0, NS, 8):
            b = sel[i:i + 8]; x = to01(fg[b]); y = yhat[b].to(dev)
            if mode == 'aug': x, _ = train_view(x, g, True)
            if mode == 'mix':
                x = make_mix(x, g)
                with torch.no_grad(), ac(): y = m(norm(x)).float().argmax(1)
            with torch.autocast('cuda', dtype=torch.bfloat16): z = m(norm(x)).float()
            m.zero_grad(set_to_none=True); margin(z, y).mean().backward()
            for n, p in m.named_parameters(): F2[n] += p.grad.float() ** 2
            nb += 1
        acc[mode] = {n: v / nb for n, v in F2.items()}
    torch.save({k: {n: v.cpu() for n, v in d.items()} for k, d in acc.items()}, f'{ADEM}/diag/fisher_{FSET}.pt')
    G = {}
    for n in acc['exact']:
        k = grp(n); e = G.setdefault(k, [0., 0., 0., 0]); e[0] += acc['exact'][n].sum().item(); e[1] += acc['aug'][n].sum().item(); e[2] += acc['mix'][n].sum().item(); e[3] += acc['exact'][n].numel()
    print('| bloc | couche | Fisher exact (somme) | exact/augmente | exact/melange |\n|---|---|---|---|---|')
    for k in sorted(G):
        e = G[k]; print(f'| {k[0]} | {k[1]} | {e[0]:.3e} | {e[0] / max(e[1], 1e-30):.2f} | {e[0] / max(e[2], 1e-30):.2f} |')
    # recouvrement des poids les plus importants : part du top 1 % 'exact' qui est aussi dans le top 1 % 'aug'
    ve = torch.cat([v.flatten() for v in acc['exact'].values()]); va = torch.cat([v.flatten() for v in acc['aug'].values()])
    k = int(0.01 * len(ve)); te = ve.topk(k).indices; ta = torch.zeros(len(ve), dtype=torch.bool, device=dev); ta[va.topk(k).indices] = True
    print('recouvrement top 1 % exact / augmente :', ta[te].float().mean().item(), ' correlation log :', torch.corrcoef(torch.stack([(ve + 1e-20).log(), (va + 1e-20).log()]))[0, 1].item())

if what == 'ablate':
    held = {k: torch.from_numpy(np.load(f'{CACHE}/held_{k}.npy', mmap_mode='r')[sel.sort()[0].numpy()]) for k in ['mild', 'color_strong']}
    ss = sel.sort()[0]; sets = {'exact': fg[ss], **held}; y = yhat[ss]
    mixes = torch.from_numpy(np.load(f'{CACHE}/mixes.npy', mmap_mode='r')[:NS].copy()); lmix = torch.from_numpy(np.load(f'{CACHE}/mixes_orig_logits.npy'))[:NS]

    def measure():
        r = {}
        for k, u8 in sets.items():
            z = logits_u8(m, u8, 256); r[k] = (margin(z, y).mean().item(), (z.argmax(1) != y).float().mean().item())
        z = logits_u8(m, mixes, 256); r['mix'] = (kl(F.softmax(lmix / 2, 1), F.log_softmax(z / 2, 1)).mean().item(), (z.argmax(1) == lmix.argmax(1)).float().mean().item())
        return r
    b0 = measure(); print('orig', b0, flush=True)
    print('| bloc | branche | s | marge exact (chute) | marge mild (chute) | marge couleur forte (chute) | faux exact / mild / couleur | melanges KL / accord | selectivite = chute exact / chute couleur |\n|---|---|---|---|---|---|---|---|---|', flush=True)
    for s in [0.0, 0.5]:
        for bi in range(12):
            for br in ['ls1', 'ls2']:
                gm = getattr(m.blocks[bi], br).gamma; old = gm.data.clone(); gm.data.mul_(s); r = measure(); gm.data.copy_(old)
                de = b0['exact'][0] - r['exact'][0]; dm = b0['mild'][0] - r['mild'][0]; dc = b0['color_strong'][0] - r['color_strong'][0]
                print(f"| {bi} | {'attention' if br == 'ls1' else 'MLP'} | {s} | {r['exact'][0]:.2f} ({de:.2f}) | {r['mild'][0]:.2f} ({dm:.2f}) | {r['color_strong'][0]:.2f} ({dc:.2f}) | {r['exact'][1]:.3f} / {r['mild'][1]:.3f} / {r['color_strong'][1]:.3f} | {r['mix'][0]:.3f} / {r['mix'][1]:.3f} | {de / (abs(dc) + 1e-6):.2f} |", flush=True)
