"""Sondes exactes par transformation affine des logits (aucun entrainement) a partir d'un state dict de base.
  --tau T      : head.weight et head.bias / T  (toutes les classes predites restent les memes)
  --swap c1,c2,c3,c4,... : echange les lignes de la tete par paires (c1<->c2, c3<->c4, ...) : toute image predite c1 devient predite c2
                 et inversement, avec la meme confiance. Sert a detruire la precision de k classes choisies (sonde de la porte des 70 %).
usage : python mk_affine.py base.pt sortie.pt [--tau 2.0] [--swap 3,17,40,61]"""
import os, sys, argparse, torch
ap = argparse.ArgumentParser(); ap.add_argument('base'); ap.add_argument('out'); ap.add_argument('--tau', type=float, default=1.0); ap.add_argument('--swap', default='')
a = ap.parse_args(); sd = torch.load(os.path.expanduser(a.base), map_location='cpu'); W = sd['head.weight'].double().clone(); b = sd['head.bias'].double().clone()
if a.swap:
    c = [int(x) for x in a.swap.split(',')]; assert len(c) % 2 == 0 and len(set(c)) == len(c)
    perm = torch.arange(W.shape[0]); perm[c[0::2]] = torch.tensor(c[1::2]); perm[c[1::2]] = torch.tensor(c[0::2]); W = W[perm]; b = b[perm]
s = {k: v.clone() for k, v in sd.items()}; s['head.weight'] = (W / a.tau).float(); s['head.bias'] = (b / a.tau).float()
assert all(v.dtype == torch.float32 for v in s.values() if v.is_floating_point()) and all(torch.isfinite(v).all() for v in s.values()) and list(s) == list(sd)
os.makedirs(os.path.dirname(os.path.expanduser(a.out)), exist_ok=True); torch.save(s, os.path.expanduser(a.out)); print(a.out, round(os.path.getsize(os.path.expanduser(a.out)) / 1e6, 1), 'Mo', vars(a))
