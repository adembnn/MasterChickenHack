# Le vrai orig vient-il de poids pre-entraines ? Compare model.pt a des checkpoints timm, et regarde LayerScale.
import torch, timm, sys
sd = torch.load('/home/adem/eac/t3/model.pt', map_location='cpu', weights_only=True)
print('n keys', len(sd))
g = [v for k, v in sd.items() if 'ls1.gamma' in k or 'ls2.gamma' in k]
for i in range(0, len(g), 2):
    print(f'block {i//2:2d} ls1 mean|g| {g[i].abs().mean():.5f} max {g[i].abs().max():.4f} | ls2 mean|g| {g[i+1].abs().mean():.5f} max {g[i+1].abs().max():.4f}')
for k in ['patch_embed.proj.weight', 'pos_embed', 'cls_token', 'blocks.0.attn.qkv.weight', 'blocks.6.mlp.fc1.weight', 'blocks.11.mlp.fc2.weight', 'head.weight', 'norm.weight']:
    v = sd[k].float(); print(f'{k:32s} std {v.std():.4f} absmax {v.abs().max():.3f} mean {v.mean():.4f}')
# lissage du pos_embed : correlation entre positions voisines (appris de zero sur peu de donnees = bruite)
pe = sd['pos_embed'][0, 1:].reshape(14, 14, -1)
pe = torch.nn.functional.normalize(pe, dim=-1)
print('pos_embed cos voisins horizontaux', (pe[:, :-1] * pe[:, 1:]).sum(-1).mean().item())
names = sys.argv[1:] or ['vit_base_patch16_224.augreg_in21k', 'vit_base_patch16_224.augreg_in21k_ft_in1k', 'vit_base_patch16_224.augreg2_in21k_ft_in1k', 'vit_base_patch16_224.orig_in21k', 'vit_base_patch16_224.mae', 'vit_base_patch16_224.dino', 'vit_base_patch16_224.sam_in1k', 'deit3_base_patch16_224.fb_in22k_ft_in1k', 'deit_base_patch16_224.fb_in1k', 'vit_base_patch16_clip_224.openai']
def cos(a, b): a = a.flatten().float(); b = b.flatten().float(); return (a @ b / (a.norm() * b.norm() + 1e-12)).item()
for n in names:
    try:
        m = timm.create_model(n, pretrained=True, num_classes=0); ps = m.state_dict()
        out = []
        for k in ['patch_embed.proj.weight', 'pos_embed', 'blocks.0.attn.qkv.weight', 'blocks.5.mlp.fc1.weight', 'blocks.11.attn.qkv.weight']:
            if k in ps and ps[k].shape == sd[k].shape: out.append(f'{cos(ps[k], sd[k]):+.3f}')
            else: out.append('  n/a ')
        print(f'{n:48s}', ' '.join(out))
    except Exception as e:
        print(n, 'ERR', str(e)[:100])
