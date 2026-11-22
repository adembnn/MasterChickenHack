import numpy as np
from pathlib import Path
R = Path.home() / "eac/t1/t1-reel"; C = Path("comp")
def L(p):
    z = np.load(p); return {str(n): l.astype(np.float32) for n, l in zip(z["names"], z["logits"])}
SPEC = {"tm_f_x1+f_c5+f_r50": ("trustmark", [R / "tm_f_x1/pred_{s}_trustmark.npz", C / "f_c5_{s}_trustmark.npz", C / "f_r50_{s}_trustmark.npz"]),
        "ss_f_x1+f_c4": ("stegastamp", [C / "f_x1_{s}_stegastamp.npz", C / "f_c4_{s}_stegastamp.npz"]),
        "ps_f_x1+f_s1+f_deca": ("pixelseal", [C / "f_x1_{s}_pixelseal.npz", R / "f_s1_ps/pred_{s}_pixelseal.npz", R / "f_deca/pred_{s}_pixelseal.npz"])}
for tag, (me, comps) in SPEC.items():
    for s in ("train", "test"):
        m = L(R / tag / f"pred_{s}_{me}.npz"); X = [L(str(c).format(s=s)) for c in comps]; assert all(set(x) == set(m) for x in X)
        d = max(float(np.abs(m[n] - np.mean([x[n] for x in X], 0)).max()) for n in m); sg = lambda v: 1 / (1 + np.exp(-v))
        dn = max(float(np.abs(m[n] - np.mean([x[n] / (x[n].std() + 1e-6) for x in X], 0)).max()) for n in list(m)[:20])
        print(f"{tag} {s} : {len(m)} images ; écart max avec la moyenne des logits des {len(X)} composantes : {d:.4f} (si normalisées : {dn:.2f})")
