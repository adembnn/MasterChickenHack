"""E1 du critique : marge de toile p* par image. On ré-agrandit chaque VRAIE image en miroir de p px (p < 0 = on rogne vers l'intérieur) et on garde le p
qui donne la plus grande marge du décodeur (accord moyen des logits avec le vrai message). Marge fixe : p* serré. Marge au hasard : p* étalé. Cadre plus petit : p* < 0.
Puis centrage : à somme de marges fixée, quelle part va en haut / à gauche. Rectangles : p* contre le petit côté."""
import sys, json, numpy as np, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parent; HERE = ROOT.parent if (ROOT.parent / "kit").exists() else ROOT.parent.parent; CACHE = ROOT / "cache"; sys.path.insert(0, str(HERE / "kit"))
from wmloc.watermarks import load_watermark
trm = json.loads((HERE / "release/train/metadata.json").read_text()); RT = np.load(CACHE / "REAL_train.npy", mmap_mode="r"); MT = np.load(CACHE / "MASK_train.npy", mmap_mode="r")
N = int(sys.argv[1])
def reframe(x, t, b, l, r):                              # marges haut, bas, gauche, droite ; négatif = rognage
    x = x[max(0, -t): x.shape[0] - max(0, -b), max(0, -l): x.shape[1] - max(0, -r)]
    return np.pad(x, ((max(0, t), max(0, b)), (max(0, l), max(0, r)), (0, 0)), mode="reflect")
for method in sys.argv[2:]:
    wm = load_watermark(method, models_dir=HERE / "kit" / "models", device="cuda")
    def marg(x, sgn):
        l = wm.logits(torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None].float() / 255)[0].cpu().numpy() * sgn; return float(l.mean()), float((l > 0).mean())
    ids = [i for i, m in enumerate(trm) if m["method"] == method and i % 5 != 0]; fulls = [i for i in ids if trm[i]["full_watermark"] and MT[i].mean() > 0.85][:N]
    P = [-16, -12, -8, -4, 0, 4, 8, 10, 12, 14, 16, 20, 24, 32]; M = []; B = []
    for i in fulls:
        x = np.ascontiguousarray(RT[i]); sgn = np.array([1.0 if c == "1" else -1.0 for c in trm[i]["message"]]); r = [marg(reframe(x, p, p, p, p), sgn) for p in P]; M.append([a for a, b in r]); B.append([b for a, b in r])
    M, B = np.array(M), np.array(B); ps = np.array(P)[M.argmax(1)]; i0 = P.index(0)
    print(f"\n## {method} : {len(fulls)} pleines. Marge du décodeur moyenne par p :", " ".join(f"{p}:{m:.2f}" for p, m in zip(P, M.mean(0))))
    print(f"   bits justes moyens par p :", " ".join(f"{p}:{b:.3f}" for p, b in zip(P, B.mean(0))))
    print(f"   p* par image : médiane {np.median(ps):.0f}, quartiles {np.percentile(ps, 25):.0f}-{np.percentile(ps, 75):.0f}, écart-type {ps.std():.1f}, part de p* < 0 : {(ps < 0).mean():.2f}, répartition {dict(zip(*np.unique(ps, return_counts=True)))}")
    print(f"   marge à p=0 : {M[:, i0].mean():.2f} -> à p* : {M.max(1).mean():.2f} ; bits {B[:, i0].mean():.3f} -> {B[np.arange(len(B)), M.argmax(1)].mean():.3f}")
    # centrage et forme : marge totale 2q par axe, part en haut (ou à gauche) f
    q = int(np.median(ps)) if np.median(ps) > 0 else 12; res = {}
    for i in fulls[:24]:
        x = np.ascontiguousarray(RT[i]); sgn = np.array([1.0 if c == "1" else -1.0 for c in trm[i]["message"]])
        for f in (0, 0.25, 0.5, 0.75, 1.0):
            t = int(round(2 * q * f)); res.setdefault(("vertical", f), []).append(marg(reframe(x, t, 2 * q - t, q, q), sgn)[0]); res.setdefault(("horizontal", f), []).append(marg(reframe(x, q, q, t, 2 * q - t), sgn)[0])
        for py, px in ((q, 0), (0, q), (q, q), (2 * q, q), (q, 2 * q)): res.setdefault(("forme", (py, px)), []).append(marg(reframe(x, py, py, px, px), sgn)[0])
    for k in ("vertical", "horizontal", "forme"): print(f"   {k} (q={q}) :", " ".join(f"{f}:{np.mean(v):.2f}" for (kk, f), v in res.items() if kk == k))
    st = json.load(open(ROOT / f"stats_{method}.json")); rects = [r for r in st if not r["full"] and r["rect"] > 0.98 and r["i"] % 5 != 0]; PR = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16]; MR = []
    for r in rects:
        x = np.ascontiguousarray(RT[r["i"]][r["y0"]:r["y0"] + r["h"], r["x0"]:r["x0"] + r["w"]]); sgn = np.array([1.0 if c == "1" else -1.0 for c in trm[r["i"]]["message"]])
        MR.append([marg(reframe(x, min(p, r["h"] - 1), min(p, r["h"] - 1), min(p, r["w"] - 1), min(p, r["w"] - 1)), sgn)[0] for p in PR])
    MR = np.array(MR); pr = np.array(PR)[MR.argmax(1)]; side = np.array([min(r["h"], r["w"]) for r in rects]); ok = MR.max(1) > 0.3 * np.median(M.max(1))
    print(f"   rectangles ({len(rects)}, dont {int(ok.sum())} lisibles) : marge moyenne par p :", " ".join(f"{p}:{m:.2f}" for p, m in zip(PR, MR[ok].mean(0))))
    if ok.sum() > 5: print(f"   p* des rectangles lisibles : médiane {np.median(pr[ok]):.0f}, quartiles {np.percentile(pr[ok], 25):.0f}-{np.percentile(pr[ok], 75):.0f} ; pente de p* contre le petit côté : {np.polyfit(side[ok], pr[ok], 1)[0]:.4f} px/px (0,023 attendu si zoom) ; corrélation {np.corrcoef(side[ok], pr[ok])[0, 1]:.2f}")
    del wm; torch.cuda.empty_cache()
