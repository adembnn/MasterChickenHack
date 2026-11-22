import torch, sys
src, dst = sys.argv[1], sys.argv[2]
pairs = [(17,91),(47,68),(59,63),(56,73),(74,82)]
if len(sys.argv) > 3: pairs = pairs[:int(sys.argv[3])]
sd = torch.load(src, map_location='cpu')
d = sd['state_dict'] if isinstance(sd, dict) and 'state_dict' in sd else sd
ks = [k for k,v in d.items() if torch.is_tensor(v) and v.shape[0] == 100 and v.dim() in (1,2)]
print('cles de tete :', [(k, tuple(d[k].shape), d[k].dtype) for k in ks])
assert len(ks) == 2
for k in ks:
    w = d[k].clone()
    for a,b in pairs: w[a], w[b] = d[k][b].clone(), d[k][a].clone()
    d[k] = w
torch.save(sd, dst); print('ecrit', dst)
