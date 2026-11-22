"""TrustMark : pourquoi la validation croisée monte et pas la réserve ? Bruit ou vrai écart ?
Compare un run candidat à mixr (base) sur : réserve en ensemble (5 modèles moyennés, comme au test), réserve en modèle seul (comme le hors-pli),
hors-pli, et l'ensemble homogène des 170 images (modèle seul partout). Usage : python tmcv.py <methode> <dossier base> <dossier candidat>"""
import sys, glob
from pathlib import Path
import numpy as np
from verif import HERE, RNG, miou_vec, boot, boot_diff, load_truth

me, dbase, dcand = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
meta, g = load_truth(check_png=False); idx = {m["image_name"]: i for i, m in enumerate(meta)}; bym = {m["image_name"]: m for m in meta}

def load(d):
    """rend : réserve ensemble {nom: masque}, réserve par modèle {nom: [scores]}, hors-pli {nom: masque}"""
    folds = sorted(f for f in glob.glob(str(d / "fold*")) if (Path(f) / "res.npz").exists() and (Path(f) / "oof.npz").exists())
    acc = {}; single = {}; oof = {}
    for f in folds:
        z = np.load(Path(f) / "res.npz")
        for n, l in zip(z["names"], z["logits"].astype(np.float32)):
            n = str(n); acc.setdefault(n, []).append(l); single.setdefault(n, []).append(float(miou_vec((l > 0)[None], g[n][None])[0]))
        z = np.load(Path(f) / "oof.npz")
        for n, l in zip(z["names"], z["logits"].astype(np.float32)): oof[str(n)] = float(miou_vec((l > 0)[None], g[str(n)][None])[0])
    ens = {n: float(miou_vec((np.mean(v, 0) > 0)[None], g[n][None])[0]) for n, v in acc.items()}
    return len(folds), ens, {n: float(np.mean(v)) for n, v in single.items()}, oof

nb, eb, sb, ob = load(dbase); nc, ec, sc, oc = load(dcand)
print(f"[{me}] base {dbase} ({nb} plis) | candidat {dcand} ({nc} plis)")
rn = sorted(set(eb) & set(ec)); on = sorted(set(ob) & set(oc)); assert all(idx[n] % 5 == 0 for n in rn) and not any(idx[n] % 5 == 0 for n in on)
A = lambda d, names: np.array([d[n] for n in names])
def line(tag, a, b):
    d, lo, hi, pr = boot_diff(a, b); la, ha = boot(a)
    print(f"  {tag:46s} n={len(a):3d} candidat {a.mean():.4f} [{la:.3f} ; {ha:.3f}] base {b.mean():.4f} écart {d:+.4f} IC95 [{lo:+.3f} ; {hi:+.3f}] P(mieux) {pr:.2f}")
    return a - b
g_ens = line("réserve, ensemble des plis (comme au test)", A(ec, rn), A(eb, rn))
g_sin = line("réserve, modèle seul (moyenne des plis)", A(sc, rn), A(sb, rn))
g_oof = line("hors-pli, modèle seul", A(oc, on), A(ob, on))
allc = np.concatenate([A(sc, rn), A(oc, on)]); allb = np.concatenate([A(sb, rn), A(ob, on)])
line("170 images, modèle seul partout", allc, allb)
# le gain diffère-t-il entre réserve et hors-pli ? (deux échantillons indépendants)
d = np.array([g_oof[RNG.integers(0, len(g_oof), len(g_oof))].mean() - g_sin[RNG.integers(0, len(g_sin), len(g_sin))].mean() for _ in range(10000)])
print(f"  gain hors-pli moins gain réserve (modèle seul) : {g_oof.mean() - g_sin.mean():+.4f} IC95 [{np.percentile(d, 2.5):+.3f} ; {np.percentile(d, 97.5):+.3f}] -> {'vrai écart' if np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0 else 'compatible avec le bruit'}")
d = np.array([g_ens[RNG.integers(0, len(g_ens), len(g_ens))].mean() - g_sin[RNG.integers(0, len(g_sin), len(g_sin))].mean() for _ in range(1)])
dd = g_ens - g_sin; m = dd[RNG.integers(0, len(dd), (10000, len(dd)))].mean(1)
print(f"  effet de l'ensemble sur le gain (réserve : gain en ensemble moins gain en modèle seul) : {dd.mean():+.4f} IC95 [{np.percentile(m, 2.5):+.3f} ; {np.percentile(m, 97.5):+.3f}]")
# la réserve est-elle plus dure ? niveau de la BASE, réserve contre hors-pli, modèle seul, et composition
b_res, b_oof = A(sb, rn), A(ob, on)
d = np.array([b_res[RNG.integers(0, len(b_res), len(b_res))].mean() - b_oof[RNG.integers(0, len(b_oof), len(b_oof))].mean() for _ in range(10000)])
print(f"  niveau de la base, réserve moins hors-pli (modèle seul) : {b_res.mean() - b_oof.mean():+.4f} IC95 [{np.percentile(d, 2.5):+.3f} ; {np.percentile(d, 97.5):+.3f}]")
def strat(n):
    G = g[n]; fr = G.mean(); a = min(fr, 1 - fr)
    if bym[n]["full_watermark"]: k = "pleine"
    else:
        ys, xs = np.where(G); k = "rectangle" if G.sum() >= 0.98 * (ys.max() + 1 - ys.min()) * (xs.max() + 1 - xs.min()) else "objet"
    return k, ("<3%" if a < 0.03 else "3-8%" if a < 0.08 else ">8%")
print("  composition (part des images) et niveau de la base par strate : réserve | hors-pli")
S = {}
for grp, names, sc_ in (("res", rn, sb), ("oof", on, ob)):
    for n in names: S.setdefault(strat(n), {"res": [], "oof": []})[grp].append(sc_[n])
for k in sorted(S):
    r, o = S[k]["res"], S[k]["oof"]
    print(f"    {k[0]:9s} {k[1]:5s} : {len(r) / len(rn):.2f} ({np.mean(r) if r else float('nan'):.3f}, n={len(r)}) | {len(o) / len(on):.2f} ({np.mean(o) if o else float('nan'):.3f}, n={len(o)})")
small_r = np.mean([strat(n)[1] == "<3%" for n in rn]); small_o = np.mean([strat(n)[1] == "<3%" for n in on])
print(f"  part des zones minoritaires < 3 % : réserve {small_r:.2f}, hors-pli {small_o:.2f}")
# réserve repondérée à la composition des 170 images (post-stratification sur type x aire)
w_all = {k: (len(v["res"]) + len(v["oof"])) / (len(rn) + len(on)) for k, v in S.items()}
for lab, d_ in (("base", eb), ("candidat", ec)):
    num = 0; den = 0
    for k, v in S.items():
        rr = [d_[n] for n in rn if strat(n) == k]
        if rr: num += w_all[k] * np.mean(rr); den += w_all[k]
    print(f"  réserve (ensemble) repondérée à la composition des 170 images, {lab} : {num / den:.4f} (brut {np.mean([d_[n] for n in rn]):.4f})")
