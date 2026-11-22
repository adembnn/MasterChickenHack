"""Candidat N de t1-assemblage contre une ou plusieurs références (masques honnêtes candidat_<R>_honnete_*.npz) :
écart apparié réserve / hors réserve / 510, par filigrane et population, moyenne des trois (bootstrap stratifié), fichier de test."""
import sys, json, hashlib
import numpy as np, torch
from verif import HERE, METHODS, RNG, miou_vec, boot, boot_diff, load_truth
N = sys.argv[1]; REFS = sys.argv[2].split(",")
A = HERE / "t1-assemblage"; meta, g = load_truth(check_png=False); idx = {m["image_name"]: i for i, m in enumerate(meta)}; bym = {m["image_name"]: m for m in meta}
def lm(c, me):
    z = np.load(A / f"candidat_{c}_honnete_{me}.npz"); return {str(a): m.astype(bool) for a, m in zip(z["names"], z["masks"])}
CONDS = (("réserve", lambda i: i % 5 == 0), ("hors réserve", lambda i: i % 5 != 0), ("510", lambda i: True))
for R in REFS:
    print(f"\n================ candidat {N} contre candidat {R} ================")
    D = {l: {} for l, _ in CONDS}; S = {l: {} for l, _ in CONDS}
    for me in METHODS:
        cN = lm(N, me); cR = lm(R, me); assert set(cN) == set(cR) == {m["image_name"] for m in meta if m["method"] == me}
        for lab, cond in CONDS:
            nn = [a for a in cN if cond(idx[a])]; G = np.stack([g[a] for a in nn]); sN = miou_vec(np.stack([cN[a] for a in nn]), G); sR = miou_vec(np.stack([cR[a] for a in nn]), G)
            full = np.array([bym[a]["full_watermark"] for a in nn]); d, lo, hi, pr = boot_diff(sN, sR); D[lab][me] = sN - sR; S[lab][me] = (sN.mean(), sR.mean()); la, ha = boot(sN)
            print(f"[{me}] {lab:12s} n={len(nn):3d} cand {N} {sN.mean():.4f} IC95 [{la:.3f} ; {ha:.3f}] (pleines {sN[full].mean():.4f}, partielles {sN[~full].mean():.4f}) | cand {R} {sR.mean():.4f} (pleines {sR[full].mean():.4f}, partielles {sR[~full].mean():.4f}) | écart {d:+.4f} IC95 [{lo:+.3f} ; {hi:+.3f}] P(mieux) {pr:.2f} ; mieux/pires {(sN > sR + 1e-9).sum()}/{(sN < sR - 1e-9).sum()}")
            if lab != "510":
                for l2, k in (("pleines", full), ("partielles", ~full)):
                    if (np.abs(sN[k] - sR[k]) > 1e-9).any(): d2, lo2, hi2, _ = boot_diff(sN[k], sR[k]); print(f"        {l2:10s} n={k.sum():3d} écart {d2:+.4f} IC95 [{lo2:+.3f} ; {hi2:+.3f}] ; mieux/pires {(sN[k] > sR[k] + 1e-9).sum()}/{(sN[k] < sR[k] - 1e-9).sum()}")
    for lab in D:
        bs = np.array([np.mean([D[lab][m][RNG.integers(0, len(D[lab][m]), len(D[lab][m]))].mean() for m in METHODS]) for _ in range(20000)])
        print(f"MOYENNE DES TROIS, {lab:12s} : cand {N} {np.mean([S[lab][m][0] for m in METHODS]):.4f} | cand {R} {np.mean([S[lab][m][1] for m in METHODS]):.4f} | écart apparié {np.mean([D[lab][m].mean() for m in METHODS]):+.4f} IC95 [{np.percentile(bs, 2.5):+.4f} ; {np.percentile(bs, 97.5):+.4f}] ; P(écart >= 0,02) = {np.mean(bs >= 0.02):.3f} ; P(écart > 0) = {np.mean(bs > 0):.3f}")
    # fichier de test
    tm = json.load(open(HERE / "release" / "test" / "metadata.json")); tn = [m["image_name"] for m in tm]
    def lf(c):
        p = A / f"submission_candidat_{c}.pt"; s = torch.load(p, weights_only=True); return p, s
    p, sub = lf(N); _, sr = lf(R); M = sub["masks"]
    print(f"[fichier {N}] sha256 {hashlib.sha256(open(p, 'rb').read()).hexdigest()} ; clés {list(sub.keys())} ; forme {tuple(M.shape)} ; type {M.dtype} ; valeurs {sorted(torch.unique(M).tolist())} ; noms distincts {len(set(sub['image_names']))} ; égaux au test : {set(sub['image_names']) == set(tn)} ; même ordre que cand {R} : {list(sub['image_names']) == list(sr['image_names'])}")
    sN = {a: m.numpy().astype(bool) for a, m in zip(sub["image_names"], sub["masks"])}; sR = {a: m.numpy().astype(bool) for a, m in zip(sr["image_names"], sr["masks"])}
    for me in METHODS:
        te = [m["image_name"] for m in tm if m["method"] == me]; ch = np.array([(sN[a] != sR[a]).mean() for a in te]); cN = lm(N, me); cR = lm(R, me); tr = np.array([(cN[a] != cR[a]).mean() for a in cN])
        print(f"[fichier] {me:10s} part filigranée test {np.mean([sN[a].mean() for a in te]):.3f} (cand {R} : {np.mean([sR[a].mean() for a in te]):.3f}) ; train honnête {np.mean([cN[a].mean() for a in cN]):.3f} ; cases changées test {ch.mean():.4f} | train {tr.mean():.4f} ; images changées test {np.mean(ch > 0):.2f} (> 5 % des cases : {np.mean(ch > 0.05):.2f}) | train {np.mean(tr > 0):.2f} ({np.mean(tr > 0.05):.2f}) ; tout-0 test {np.mean([not sN[a].any() for a in te]):.3f}, tout-1 {np.mean([sN[a].all() for a in te]):.3f}")
