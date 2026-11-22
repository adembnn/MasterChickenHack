"""Témoin du test p* : on fabrique des pleines avec une marge CONNUE (centrée ou non) et on regarde ce que l'estimateur « ré-agrandir en miroir et maximiser la marge du décodeur » retrouve."""
import sys, json, numpy as np, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parent; HERE = ROOT.parent if (ROOT.parent / "kit").exists() else ROOT.parent.parent; CACHE = ROOT / "cache"; sys.path.insert(0, str(HERE / "kit"))
from wmloc.watermarks import load_watermark
trm = json.loads((HERE / "release/train/metadata.json").read_text()); RT = np.load(CACHE / "REAL_train.npy", mmap_mode="r"); N = int(sys.argv[1]); method = sys.argv[2]
wm = load_watermark(method, models_dir=HERE / "kit" / "models", device="cuda"); rng = np.random.default_rng(0); others = [i for i, m in enumerate(trm) if m["method"] != method and i % 5 != 0]
def reframe(x, p): return x[-p:x.shape[0] + p, -p:x.shape[1] + p] if p < 0 else np.pad(x, ((p, p), (p, p), (0, 0)), mode="reflect")
def marg(x, sgn):
    l = wm.logits(torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None].float() / 255)[0].cpu().numpy() * sgn; return float(l.mean()), float((l > 0).mean())
bases = [np.ascontiguousarray(RT[int(rng.choice(others))]) for _ in range(N)]; msgs = [rng.integers(0, 2, wm.nbits) for _ in range(N)]; P = [-8, -4, 0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24]
def fab(b, m, t, bo, l, r): c = np.pad(b, ((t, bo), (l, r), (0, 0)), mode="reflect"); return wm.embed(np.ascontiguousarray(c), m)[t:t + 512, l:l + 512]
cases = {f"centrée {q} px": (lambda q=q: (q, q, q, q)) for q in (0, 4, 8, 12, 16)}
cases["décentrée, total 16 px par axe"] = lambda: (lambda a, b: (a, 16 - a, b, 16 - b))(int(rng.integers(0, 17)), int(rng.integers(0, 17)))
cases["décentrée, total 24 px par axe"] = lambda: (lambda a, b: (a, 24 - a, b, 24 - b))(int(rng.integers(0, 25)), int(rng.integers(0, 25)))
cases["centrée, tirée entre 0 et 16"] = lambda: (lambda q: (q, q, q, q))(int(rng.integers(0, 17)))
print(f"## {method} : témoin sur {N} pleines fabriquées (réel mesuré : marge à p=0 6,16, bits 0,970, p* médian 4, quartiles 0-8, courbe max à p=4)")
for nm, f in cases.items():
    M = []; B0 = []
    for b, m in zip(bases, msgs):
        w = fab(b, m, *f()); sgn = m * 2.0 - 1; r = [marg(reframe(w, p), sgn) for p in P]; M.append([a for a, _ in r]); B0.append(r[P.index(0)][1])
    M = np.array(M); ps = np.array(P)[M.argmax(1)]
    print(f"  fabriqué {nm:32s}: marge à p=0 {M[:, P.index(0)].mean():.2f}, bits {np.mean(B0):.3f} ; p* médian {np.median(ps):.0f}, quartiles {np.percentile(ps, 25):.0f}-{np.percentile(ps, 75):.0f} ; courbe max à p={P[int(M.mean(0).argmax())]} ; marge à p* {M.max(1).mean():.2f}")
