"""Fabrique deltas/NOM.pt a partir d'un fichier complet candidates/NOM.pt (delta = fichier - orig), pour pouvoir le mettre a l'echelle avec dash.py --alphas."""
import sys, os, torch
ME = os.environ.get('T3ME', os.path.expanduser('~/eac/t3/t3-methodes')); n = sys.argv[1]
o = torch.load(os.path.dirname(ME) + '/model.pt', map_location='cpu', weights_only=True); c = torch.load(f'{ME}/candidates/{n}.pt', map_location='cpu', weights_only=True)
d = {k: (c[k] - o[k]).float() for k in o if not torch.equal(c[k], o[k])}
torch.save({'names': list(d), 'delta': d, 'args': {'source': 'mkdelta'}}, f'{ME}/deltas/{n}.pt'); print(n, len(d), 'tenseurs modifies')
