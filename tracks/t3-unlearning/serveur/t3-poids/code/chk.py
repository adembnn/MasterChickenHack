"""t3-poids : controles de fichier d'un candidat, depuis le disque : chargement strict, float32, taille, pas de NaN/inf, sortie fp16 finie."""
import sys, os, hashlib
sys.path.insert(0, os.path.expanduser('~/eac/t3/t3-methodes/code')); os.environ.setdefault('T3ME', os.path.expanduser('~/eac/t3/t3-poids'))
from common import *
for p in sys.argv[1:]:
    sd = torch.load(p, map_location='cpu', weights_only=True); m = tt.build_model(); r = m.load_state_dict(sd, strict=True)
    fl = [v for v in sd.values() if v.is_floating_point()]; ok32 = all(v.dtype == torch.float32 for v in fl); fin = all(torch.isfinite(v).all().item() for v in fl)
    m = m.to(dev).eval(); z = logits_u8(m, load_forget()[:256]); h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
    print(f'{os.path.basename(p)} | strict OK | float32 {ok32} | fini {fin} | {os.path.getsize(p) / 1e6:.1f} Mo (<450) | sortie fp16 finie {bool(torch.isfinite(z).all())} | sha256 {h[:16]}')
