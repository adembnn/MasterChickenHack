"""Planche de 48 trous fabriqués (trou en noir), au format de la planche du vérificateur, pour comparer à l'œil. Mélange de g_s1S : 85 % formes réelles des deux autres
filigranes (symétries, gardées à leur place 85 % du temps), 15 % taches aléatoires. Seconde planche : les taches aléatoires seules."""
import sys, json, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent)); import shapes as S
rng = S.rng
def sample(pshift=0.15, pblob=0.15):
    if rng.random() < pblob: return S.blob(0.004, 0.5), "t"
    s0 = S.dih(S.bank[int(rng.integers(len(S.bank)))], int(rng.integers(8))); return (S.shifted(s0, 160) if rng.random() < pshift else s0), "r"
def sheet(L, path):
    cells = [np.asarray(Image.fromarray(((~h) * 255).astype(np.uint8)).resize((128, 128), Image.Resampling.BOX)) for h in L]; pad = lambda c: np.pad(c, 3, constant_values=128)
    rows = [np.concatenate([pad(c) for c in cells[r * 8:(r + 1) * 8]], 1) for r in range(6)]; Image.fromarray(np.concatenate(rows, 0)).save(path)
out = Path(sys.argv[2]); L = [sample()[0] for _ in range(48)]; sheet(L, out / "planche_fab.png"); sheet([S.blob(0.004, 0.5) for _ in range(48)], out / "planche_fab_taches.png"); sheet(S.real[:48], out / "planche_vrais_trustmark.png")
A = np.array([S.stats(h) for h in L]); print(f"48 trous fabriqués (mélange g_s1S) : part médiane {np.median(A[:, 0]):.3f}, déciles {np.quantile(A[:, 0], .1):.3f} à {np.quantile(A[:, 0], .9):.3f}, plusieurs morceaux {np.mean(A[:, 1] >= 2):.2f}, remplissage médian de la boîte {np.median(A[:, 3]):.2f}")
