"""Vérification d'un candidat de t1-assemblage : masques honnêtes du train (recette réglée hors réserve), fichier de contrôle, fichier de test.
Usage : python cand5.py <n>"""
import sys, json, glob
from pathlib import Path
import numpy as np, torch
from verif import HERE, METHODS, RNG, miou_vec, boot, boot_diff, load_truth, stats
n = sys.argv[1]; A = HERE / "t1-assemblage"; meta, g = load_truth(check_png=False); idx = {m["image_name"]: i for i, m in enumerate(meta)}; bym = {m["image_name"]: m for m in meta}
tm = json.load(open(HERE / "release" / "test" / "metadata.json"))
npz = lambda p: {str(a): l for a, l in zip(np.load(p)["names"], np.load(p)["logits"].astype(np.float32) > 0)}
def reel(d, kind):
    acc = {}; out = {}
    for f in sorted(glob.glob(str(d / "fold*" / "res.npz"))):
        if kind == "test":
            z = np.load(f.replace("res.npz", "test.npz"))
            for a, l in zip(z["names"], z["logits"].astype(np.float32)): acc.setdefault(str(a), []).append(l)
            continue
        z = np.load(f)
        for a, l in zip(z["names"], z["logits"].astype(np.float32)): acc.setdefault(str(a), []).append(l)
        z = np.load(f.replace("res.npz", "oof.npz")); out.update({str(a): l > 0 for a, l in zip(z["names"], z["logits"].astype(np.float32))})
    out.update({a: np.mean(v, 0) > 0 for a, v in acc.items()}); return out
J = A / "julich/t1-reel/runs/mixr/trustmark"
v6tr = {"stegastamp": npz(HERE / "t1-reel/pred_train_stegastamp.npz"), "trustmark": reel(J, "train"), "pixelseal": npz(HERE / "t1-pixelseal/pred_train_pixelseal.npz")}
v6te = {"stegastamp": npz(HERE / "t1-reel/pred_test_stegastamp.npz"), "trustmark": reel(J, "test"), "pixelseal": npz(HERE / "t1-pixelseal/pred_test_pixelseal.npz")}
ctl = torch.load(A / f"controle_train_{n}.pt", weights_only=True); ctl = {a: m.numpy().astype(bool) for a, m in zip(ctl["image_names"], ctl["masks"])}
sub = torch.load(A / f"submission_candidat_{n}.pt", weights_only=True); names_te = list(sub["image_names"]); M = sub["masks"]
print(f"[fichier test] {len(names_te)} noms, {len(set(names_te))} distincts, forme {tuple(M.shape)}, dtype {M.dtype}, valeurs {sorted(torch.unique(M).tolist())}, noms = ceux du test : {set(names_te) == {m['image_name'] for m in tm}}")
subm = {a: m.numpy().astype(bool) for a, m in zip(names_te, M)}
D = {}; Dc = {}; rows = {}
for me in METHODS:
    z = np.load(A / f"candidat_{n}_honnete_{me}.npz"); hon = {str(a): m.astype(bool) for a, m in zip(z["names"], z["masks"])}; resflag = {str(a): bool(r) for a, r in zip(z["names"], z["reserve"])}
    assert all(resflag[a] == (idx[a] % 5 == 0) for a in hon), "drapeau de réserve du fichier honnête faux"
    rn = [a for a in hon if idx[a] % 5 == 0]; G = np.stack([g[a] for a in rn]); full = np.array([bym[a]["full_watermark"] for a in rn])
    sh = miou_vec(np.stack([hon[a] for a in rn]), G); s6 = miou_vec(np.stack([v6tr[me][a] for a in rn]), G); sc = miou_vec(np.stack([ctl[a] for a in rn]), G)
    d, lo, hi, pr = boot_diff(sh, s6); la, ha = boot(sh); D[me] = sh - s6; Dc[me] = sc - s6; rows[me] = (sh.mean(), s6.mean())
    print(f"[{me}] réserve n={len(rn)} : candidat honnête {sh.mean():.4f} IC95 [{la:.3f} ; {ha:.3f}] (pleines {sh[full].mean():.4f} n={full.sum()}, partielles {sh[~full].mean():.4f} n={(~full).sum()}) | v6 {s6.mean():.4f} (pleines {s6[full].mean():.4f}, partielles {s6[~full].mean():.4f}) | écart apparié {d:+.4f} IC95 [{lo:+.3f} ; {hi:+.3f}] P(mieux) {pr:.2f}")
    for lab, k in (("pleines", full), ("partielles", ~full)):
        d2, lo2, hi2, pr2 = boot_diff(sh[k], s6[k]); print(f"      {lab:10s} écart {d2:+.4f} IC95 [{lo2:+.3f} ; {hi2:+.3f}] ; images mieux / égales / pires : {(sh[k] > s6[k] + 1e-9).sum()} / {(np.abs(sh[k] - s6[k]) <= 1e-9).sum()} / {(sh[k] < s6[k] - 1e-9).sum()}")
    print(f"      fichier de contrôle (seuil re-réglé sur les 170, OPTIMISTE) sur la réserve : {sc.mean():.4f} ; masques honnêtes = contrôle sur {np.mean([np.array_equal(hon[a], ctl[a]) for a in rn]):.2f} des images de réserve")
    hn = [a for a in hon if idx[a] % 5 != 0]; Gh = np.stack([g[a] for a in hn]); print(f"      hors réserve (réglé dessus, ne compte pas) : candidat {miou_vec(np.stack([hon[a] for a in hn]), Gh).mean():.4f} | v6 {miou_vec(np.stack([v6tr[me][a] for a in hn]), Gh).mean():.4f}")
    # cohérence train / test et différences avec v6 reconstruite
    te = [m["image_name"] for m in tm if m["method"] == me]; Pte = np.stack([subm[a] for a in te]); P6 = np.stack([v6te[me][a] for a in te]); Ptr = np.stack([hon[a] for a in hon]); P6tr = np.stack([v6tr[me][a] for a in hon])
    ch = (Pte != P6).reshape(len(te), -1).mean(1); chtr = (Ptr != P6tr).reshape(len(Ptr), -1).mean(1)
    print(f"      test : part filigranée {Pte.mean():.3f} (v6 {P6.mean():.3f}) ; images changées contre v6 : test {np.mean(ch > 0):.2f} (plus de 5 % des cases : {np.mean(ch > 0.05):.2f}, cases changées {ch.mean():.4f}) | train {np.mean(chtr > 0):.2f} ({np.mean(chtr > 0.05):.2f}, {chtr.mean():.4f})")
    print("      stats train honnête :", json.dumps(stats(Ptr))); print("      stats test          :", json.dumps(stats(Pte)))
bs = [np.mean([D[me][RNG.integers(0, len(D[me]), len(D[me]))].mean() for me in METHODS]) for _ in range(20000)]
print(f"\nMOYENNE DES TROIS, réserve : candidat honnête {np.mean([rows[m][0] for m in METHODS]):.4f} | v6 {np.mean([rows[m][1] for m in METHODS]):.4f} | écart apparié {np.mean([D[m].mean() for m in METHODS]):+.4f} IC95 [{np.percentile(bs, 2.5):+.4f} ; {np.percentile(bs, 97.5):+.4f}] ; P(écart >= 0,02) = {np.mean(np.array(bs) >= 0.02):.2f}")
