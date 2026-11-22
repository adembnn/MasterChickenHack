import numpy as np
from pathlib import Path
R = Path.home() / "eac/t1/t1-reel"; C = Path("comp")
def L(p):
    z = np.load(p); return {str(n): l.astype(np.float32) for n, l in zip(z["names"], z["logits"])}
SPEC = {"ps_f_x2+f_x1+f_dx": ("pixelseal", ["f_x2", "f_x1", "f_dx"]), "tm_f_x2+f_c5+f_r50b": ("trustmark", ["f_x2", "f_c5", "f_r50b"])}
for tag, (me, comps) in SPEC.items():
    for s in ("train", "test"):
        m = L(R / tag / f"pred_{s}_{me}.npz"); X = [L(C / f"{c}_{s}_{me}.npz") for c in comps]; assert all(set(x) == set(m) for x in X)
        print(f"{tag} {s} : {len(m)} images ; écart max avec la moyenne des logits des {len(X)} composantes : {max(float(np.abs(m[n] - np.mean([x[n] for x in X], 0)).max()) for n in m):.4f}")
