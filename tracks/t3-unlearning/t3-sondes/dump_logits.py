"""Sorties (logits fp16, comme le serveur) d'un ou plusieurs state dicts sur les 3 500 images exactes, les copies tenues a l'ecart
(cache de t3-methodes) et les melanges. usage : python dump_logits.py nom=chemin.pt [...]  -> ~/eac/t3/t3-sondes/logits/<nom>.npz"""
import os, sys, numpy as np, torch
os.environ.setdefault('T3ME', os.path.expanduser('~/eac/t3/t3-sondes'))
sys.path.insert(0, os.path.expanduser('~/eac/t3/t3-methodes/code') if os.path.exists(os.path.expanduser('~/eac/t3/t3-methodes/code')) else os.path.expanduser('~/eac/t3/t3-sondes/code'))
import common as C
MC = os.environ.get('T3MC', os.path.expanduser('~/eac/t3/t3-methodes/cache')); OUT = os.path.expanduser('~/eac/t3/t3-sondes/logits'); os.makedirs(OUT, exist_ok=True)
SETS = [s for s in ['held_mild', 'held_cropflip', 'held_jpeg50', 'held_rot15', 'held_blur_strong', 'held_color_strong', 'mixes'] if os.path.exists(f'{MC}/{s}.npy')]
u8 = C.load_forget()
for a in sys.argv[1:]:
    name, path = a.split('=', 1); m = C.tt.build_model(); sd = torch.load(os.path.expanduser(path), map_location='cpu')
    print(name, m.load_state_dict(sd, strict=True), flush=True); m = m.to('cuda').eval()
    d = {'exact': C.logits_u8(m, u8).numpy()}
    for s in SETS: d[s] = C.logits_u8(m, torch.from_numpy(np.load(f'{MC}/{s}.npy', mmap_mode='r')[:])).numpy()
    np.savez(f'{OUT}/{name}.npz', **d); print(name, {k: v.shape for k, v in d.items()}, 'finite', all(np.isfinite(v).all() for v in d.values()), flush=True)
