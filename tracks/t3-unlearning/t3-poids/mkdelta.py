"""t3-poids : operations sur les deltas. orth A B OUT (retire de A sa projection sur B, tenseur par tenseur) ; blocks A lo hi OUT (garde les blocs lo..hi-1 ; hi=13 inclut norme finale et tete ; lo=-1 inclut l'embedding) ;
cos A B (cosinus global et par bloc) ; norms A (norme relative par bloc)."""
import sys, os, math, torch
D = os.path.expanduser('~/eac/t3/t3-poids/deltas')
def L(n): return torch.load(f'{D}/{n}.pt')['delta']
def grp(k):
    if k.startswith('blocks.'): return int(k.split('.')[1])
    return 12 if (k.startswith('norm.') or k.startswith('head.')) else -1
op = sys.argv[1]
if op == 'orth':
    A, B = L(sys.argv[2]), L(sys.argv[3]); out = {}
    for k in A:
        b = B[k].flatten().double(); a = A[k].flatten().double(); out[k] = (a - (a @ b) / (b @ b).clamp_min(1e-30) * b).view_as(A[k]).float()
    torch.save({'delta': out}, f'{D}/{sys.argv[4]}.pt')
elif op == 'blocks':
    A = L(sys.argv[2]); lo, hi = int(sys.argv[3]), int(sys.argv[4]); torch.save({'delta': {k: (v if lo <= grp(k) < hi else torch.zeros_like(v)) for k, v in A.items()}}, f'{D}/{sys.argv[5]}.pt')
elif op == 'cos':
    A, B = L(sys.argv[2]), L(sys.argv[3]); ab = {}; aa = {}; bb = {}
    for k in A:
        g = grp(k); a = A[k].flatten().double(); b = B[k].flatten().double(); ab[g] = ab.get(g, 0) + (a @ b).item(); aa[g] = aa.get(g, 0) + (a @ a).item(); bb[g] = bb.get(g, 0) + (b @ b).item()
    print('cos global', sum(ab.values()) / math.sqrt(sum(aa.values()) * sum(bb.values())), ' |A|', math.sqrt(sum(aa.values())), ' |B|', math.sqrt(sum(bb.values())))
    print('par groupe (-1 embed, 0..11 blocs, 12 norme+tete):', ' '.join(f'{g}:{ab[g] / math.sqrt(aa[g] * bb[g] + 1e-30):.2f}' for g in sorted(ab)))
elif op == 'norms':
    A = L(sys.argv[2]); O = torch.load(os.path.expanduser('~/eac/t3/model.pt'), map_location='cpu', weights_only=True); aa = {}; oo = {}
    for k in A:
        g = grp(k); aa[g] = aa.get(g, 0) + (A[k].double() ** 2).sum().item(); oo[g] = oo.get(g, 0) + (O[k].double() ** 2).sum().item()
    print('norme relative par groupe:', ' '.join(f'{g}:{math.sqrt(aa[g] / oo[g]):.4f}' for g in sorted(aa)))
