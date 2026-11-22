"""Candidat 5 de t1-assemblage : je reconstruis les masques HONNÊTES du train (recette réglée hors réserve) à partir du fichier de contrôle
(recette finale) : seules deux populations ont un seuil final différent du seuil honnête (TrustMark pleines, StegaStamp pleines, mode « seuil » simple) ;
pour elles je recalcule masque = (logit centré-réduit > seuil honnête) et je vérifie ma reconstruction en retrouvant le contrôle avec le seuil final."""
import json, glob
from pathlib import Path
import numpy as np, torch
from verif import HERE, METHODS, RNG, miou_vec, boot, boot_diff, load_truth, stats
A = HERE / "t1-assemblage"; meta, g = load_truth(check_png=False); idx = {m["image_name"]: i for i, m in enumerate(meta)}; bym = {m["image_name"]: m for m in meta}
T = json.load(open(A / "assemble_5.json")); tm = json.load(open(HERE / "release" / "test" / "metadata.json"))
lg = lambda p: {str(a): l for a, l in zip(np.load(p)["names"], np.load(p)["logits"].astype(np.float32))}
def reel(d, kind):
    acc = {}; out = {}
    for f in sorted(glob.glob(str(d / "fold*" / "res.npz"))):
        z = np.load(f.replace("res.npz", "test.npz") if kind == "test" else f)
        for a, l in zip(z["names"], z["logits"].astype(np.float32)): acc.setdefault(str(a), []).append(l)
        if kind != "test":
            z = np.load(f.replace("res.npz", "oof.npz")); out.update({str(a): l for a, l in zip(z["names"], z["logits"].astype(np.float32))})
    out.update({a: np.mean(v, 0) for a, v in acc.items()}); return out
J = A / "julich/t1-reel/runs/mixr/trustmark"
v6tr = {"stegastamp": lg(HERE / "t1-reel/pred_train_stegastamp.npz"), "trustmark": reel(J, "train"), "pixelseal": lg(HERE / "t1-pixelseal/pred_train_pixelseal.npz")}
v6te = {"stegastamp": lg(HERE / "t1-reel/pred_test_stegastamp.npz"), "trustmark": reel(J, "test"), "pixelseal": lg(HERE / "t1-pixelseal/pred_test_pixelseal.npz")}
ctl = torch.load(A / "controle_train_5.pt", weights_only=True); ctl = {a: m.numpy().astype(bool) for a, m in zip(ctl["image_names"], ctl["masks"])}
sub = torch.load(A / "submission_candidat_5.pt", weights_only=True); subm = {a: m.numpy().astype(bool) for a, m in zip(sub["image_names"], sub["masks"])}
f = np.load(A / "feat_train.npz"); acc = {str(a): float(v) for a, v in zip(f["names"], f["acc"])}; ft = np.load(A / "feat_test.npz"); acct = {str(a): float(v) for a, v in zip(ft["names"], ft["acc"])}
SRC = {"trustmark": HERE / "t1-reel/pred_train_trustmark.npz", "stegastamp": A / "pred_train_rang-reel-pred_stegastamp.npz"}
D = {}; rows = {}; ALLH = {}
for me in METHODS:
    names = [m["image_name"] for m in meta if m["method"] == me]; th = T["pop"][me]["th"] if isinstance(T["pop"][me], dict) and "th" in T["pop"][me] else None
    pf = np.array([acc[a] > th for a in names]); hon = {a: ctl[a] for a in names}; c = T["choix"][me]["pleines"]
    if c["p"] != c["p_final"]:
        assert c["p"]["mode"] == "seuil" and c["p"]["sigma"] == 0 and c["p"]["area"] == 0 and not c["p"]["clamp"] and list(c["w"].values()) == [1], "recette non simple"
        L = lg(SRC[me]); hr = np.stack([L[a] for a in names if idx[a] % 5 != 0]); mu, sd = float(hr.mean()), float(hr.std()) + 1e-6
        same = np.mean([np.array_equal((L[a] - mu) / sd > c["p_final"]["t"], ctl[a]) for a, p in zip(names, pf) if p])
        for a, p in zip(names, pf):
            if p: hon[a] = (L[a] - mu) / sd > c["p"]["t"]
        print(f"[{me}] pleines : seuil honnête {c['p']['t']:.3f}, seuil final {c['p_final']['t']:.3f} ; ma reconstruction retrouve le fichier de contrôle sur {same:.3f} des images classées pleines")
    else: print(f"[{me}] recette finale = recette réglée hors réserve pour les deux populations : contrôle = honnête")
    ALLH.update(hon)
    rn = [a for a in names if idx[a] % 5 == 0]; G = np.stack([g[a] for a in rn]); full = np.array([bym[a]["full_watermark"] for a in rn])
    sh = miou_vec(np.stack([hon[a] for a in rn]), G); s6 = miou_vec(np.stack([v6tr[me][a] > 0 for a in rn]), G); sc = miou_vec(np.stack([ctl[a] for a in rn]), G)
    d, lo, hi, pr = boot_diff(sh, s6); la, ha = boot(sh); D[me] = sh - s6; rows[me] = (sh.mean(), s6.mean(), sc.mean())
    print(f"   réserve n={len(rn)} : candidat honnête {sh.mean():.4f} IC95 [{la:.3f} ; {ha:.3f}] (pleines {sh[full].mean():.4f} n={full.sum()}, partielles {sh[~full].mean():.4f} n={(~full).sum()}) | annoncé {T['score'][me]['honnete']['reserve']} | v6 {s6.mean():.4f} (pleines {s6[full].mean():.4f}, partielles {s6[~full].mean():.4f}) | écart apparié {d:+.4f} IC95 [{lo:+.3f} ; {hi:+.3f}] P(mieux) {pr:.2f}")
    for lab, k in (("pleines", full), ("partielles", ~full)):
        d2, lo2, hi2, pr2 = boot_diff(sh[k], s6[k]); print(f"      {lab:10s} écart {d2:+.4f} IC95 [{lo2:+.3f} ; {hi2:+.3f}] ; images mieux / égales / pires : {(sh[k] > s6[k] + 1e-9).sum()} / {(np.abs(sh[k] - s6[k]) <= 1e-9).sum()} / {(sh[k] < s6[k] - 1e-9).sum()}")
    print(f"      fichier de contrôle (seuil final, re-réglé sur les 170) sur la réserve : {sc.mean():.4f}")
    hn = [a for a in names if idx[a] % 5 != 0]; Gh = np.stack([g[a] for a in hn]); sah = miou_vec(np.stack([hon[a] for a in hn]), Gh); s6h = miou_vec(np.stack([v6tr[me][a] > 0 for a in hn]), Gh)
    print(f"      hors réserve (recettes réglées dessus : optimiste pour le candidat) : candidat {sah.mean():.4f} | v6 {s6h.mean():.4f}")
    te = [m["image_name"] for m in tm if m["method"] == me]; Pte = np.stack([subm[a] for a in te]); P6 = np.stack([v6te[me][a] > 0 for a in te]); Ptr = np.stack([hon[a] for a in names]); P6tr = np.stack([v6tr[me][a] > 0 for a in names])
    ch = (Pte != P6).reshape(len(te), -1).mean(1); chtr = (Ptr != P6tr).reshape(len(Ptr), -1).mean(1); pft = np.array([acct[a] > th for a in te])
    print(f"      test : part filigranée {Pte.mean():.3f} (v6 reconstruite {P6.mean():.3f}) ; classées pleines : train {pf.mean():.3f}, test {pft.mean():.3f} ; images changées contre v6 : test {np.mean(ch > 0):.2f} (> 5 % des cases : {np.mean(ch > 0.05):.2f} ; cases changées {ch.mean():.4f}) | train {np.mean(chtr > 0):.2f} ({np.mean(chtr > 0.05):.2f} ; {chtr.mean():.4f})")
    print("      stats train honnête :", json.dumps(stats(Ptr))); print("      stats test          :", json.dumps(stats(Pte)))
    # usage de la boîte du décodeur : part des images classées partielles dont la confiance dépasse z0, train contre test
    z0 = T["choix"][me]["partielles"]["p"].get("box")
    if z0 is not None:
        def zs(split):
            out = {}
            for fn in T["boites"][me]["fichiers"]:
                if f"_{split}" not in fn: continue
                for r in json.load(open(HERE / fn)) if isinstance(json.load(open(HERE / fn)), list) else json.load(open(HERE / fn)).get("boxes", []):
                    if r.get("box") is None: continue
                    out[r["name"]] = float(r["z"]) if "z" in r and r["z"] is not None else (float(r["bits"]) - 0.5) * 2 * (256 ** 0.5)
            return out
        ztr, zte = zs("train"), zs("test"); ptr = [a for a, p in zip(names, pf) if not p]; pte = [a for a, p in zip(te, pft) if not p]
        utr = [a for a in ptr if ztr.get(a, -9) >= z0]; ute = [a for a in pte if zte.get(a, -9) >= z0]
        isrect = lambda a: (lambda M: M.sum() >= 0.98 * (np.where(M)[0].max() + 1 - np.where(M)[0].min()) * (np.where(M)[1].max() + 1 - np.where(M)[1].min()))(g[a])
        print(f"      boîte du décodeur (z0 = {z0}) : rendue pour {len(utr)}/{len(ptr)} = {len(utr) / len(ptr):.2f} des partielles du train (dont réserve {sum(idx[a] % 5 == 0 for a in utr)}), {len(ute)}/{len(pte)} = {len(ute) / len(pte):.2f} au test ; au train, {np.mean([isrect(a) for a in utr]):.2f} des boîtes rendues sont de vrais rectangles, mIoU sur ces images {miou_vec(np.stack([hon[a] for a in utr]), np.stack([g[a] for a in utr])).mean():.3f}")
bs = np.array([np.mean([D[me][RNG.integers(0, len(D[me]), len(D[me]))].mean() for me in METHODS]) for _ in range(20000)])
print(f"\nMOYENNE DES TROIS, réserve : candidat honnête {np.mean([rows[m][0] for m in METHODS]):.4f} (annoncé {T['score']['ensemble']['honnete']['reserve']} sur 102 images groupées) | v6 {np.mean([rows[m][1] for m in METHODS]):.4f} | écart apparié {np.mean([D[m].mean() for m in METHODS]):+.4f} IC95 [{np.percentile(bs, 2.5):+.4f} ; {np.percentile(bs, 97.5):+.4f}] ; P(écart >= 0,02) = {np.mean(bs >= 0.02):.3f} ; contrôle (seuil final) {np.mean([rows[m][2] for m in METHODS]):.4f}")
np.savez_compressed(Path(__file__).with_name("cand5_honnete.npz"), names=np.array(list(ALLH)), masks=np.stack([ALLH[a] for a in ALLH]))
