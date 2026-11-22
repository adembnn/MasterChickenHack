"""Vérificateur indépendant T1 (session t1-verificateur). Aucun code des fabricantes n'est importé.
Score officiel réécrit d'après kit/wmloc/utils.py : par image, moyenne sur c dans {0,1} de |P_c & G_c| / |P_c | G_c| (1 si union vide).
Usage : python verif.py base            -> recalcule la base 1227
"""
import sys, json, glob
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent.parent
if not (HERE / "release").exists(): HERE = HERE.parent          # adem/t1-verificateur sur Jülich
REL = HERE / "release"
METHODS = ("pixelseal", "trustmark", "stegastamp")
RNG = np.random.default_rng(20261121)


def miou_vec(P, G):
    """P, G : [N,128,128] booléens. Renvoie le score officiel par image."""
    P = P.reshape(len(P), -1).astype(bool); G = G.reshape(len(G), -1).astype(bool)
    out = np.zeros(len(P))
    for c in (True, False):
        p = P == c; g = G == c
        inter = (p & g).sum(1).astype(float); union = (p | g).sum(1).astype(float)
        out += np.where(union == 0, 1.0, inter / np.maximum(union, 1))
    return out / 2


def miou_naif(p, g):
    s = 0.0
    for c in (0, 1):
        a = {i for i, v in enumerate(p.ravel()) if v == c}; b = {i for i, v in enumerate(g.ravel()) if v == c}
        s += 1.0 if not (a | b) else len(a & b) / len(a | b)
    return s / 2


def boot(x, n=10000):
    """Intervalle à 95 % par rééchantillonnage des images (tirage avec remise)."""
    x = np.asarray(x, float)
    m = x[RNG.integers(0, len(x), (n, len(x)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_diff(a, b, n=10000):
    """Écart apparié a-b sur les mêmes images : moyenne, intervalle 95 %, part des tirages où a > b."""
    d = np.asarray(a, float) - np.asarray(b, float)
    m = d[RNG.integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), float((m > 0).mean())


def load_truth(check_png=True):
    meta = json.load(open(REL / "train" / "metadata.json"))
    z = np.load(REL / "train" / "masks_128.npz"); g = {str(n): m.astype(bool) for n, m in zip(z["names"], z["masks"])}
    if check_png:                                              # recalcul depuis les PNG 512 : règle >= 50 % par case 4x4
        from PIL import Image
        bad = 0
        for m in meta:
            a = np.asarray(Image.open(REL / "train" / "masks" / f"{m['image_name']}.png").convert("L")) > 127
            d = a.reshape(128, 4, 128, 4).mean((1, 3)) >= 0.5
            bad += int((d != g[m["image_name"]]).any())
        print(f"[vérité] masks_128.npz contre PNG 512 rééchantillonnés par moi : {bad} image(s) différente(s) sur {len(meta)}")
    return meta, g


def ncomp(m):
    """Nombre de composantes connexes (4-voisins) de la classe vraie."""
    try:
        from scipy import ndimage
        return int(ndimage.label(m)[1])
    except Exception:
        m = m.copy(); n = 0; H, W = m.shape
        for y, x in zip(*np.where(m)):
            if not m[y, x]: continue
            n += 1; st = [(y, x)]; m[y, x] = False
            while st:
                a, b = st.pop()
                for u, v in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)):
                    if 0 <= u < H and 0 <= v < W and m[u, v]: m[u, v] = False; st.append((u, v))
        return n


def stats(M):
    """Statistiques sans vérité d'un lot de masques [N,128,128]."""
    fr = M.reshape(len(M), -1).mean(1)
    nc = np.array([ncomp(m) for m in M]); nh = np.array([ncomp(~m) for m in M])
    q = lambda v: [round(float(x), 3) for x in np.percentile(v, [10, 25, 50, 75, 90])]
    return dict(n=len(M), part_cellules=round(float(M.mean()), 3), part_images_majoritaires=round(float((fr > 0.5).mean()), 3),
                tout0=round(float((fr == 0).mean()), 3), tout1=round(float((fr == 1).mean()), 3),
                aire_maj_q=q(fr[fr > 0.5]) if (fr > 0.5).any() else None, aire_min_q=q(fr[fr <= 0.5]) if (fr <= 0.5).any() else None,
                comp_filigrane_moy=round(float(nc.mean()), 2), comp_propre_moy=round(float(nh.mean()), 2))


def report(tag, names, P, meta, g, sel=None):
    """Score sur un ensemble d'images nommées. Renvoie les scores par image."""
    byname = {m["image_name"]: (i, m) for i, m in enumerate(meta)}
    G = np.stack([g[n] for n in names]); s = miou_vec(P, G)
    full = np.array([byname[n][1]["full_watermark"] for n in names])
    lo, hi = boot(s)
    line = f"{tag:42s} n={len(s):3d}  mIoU {s.mean():.4f}  IC95 [{lo:.3f} ; {hi:.3f}]  pleines {s[full].mean():.4f} (n={full.sum()})  partielles {s[~full].mean():.4f} (n={(~full).sum()})"
    print(line)
    return s, full


def avg_folds(files, key="logits"):
    acc = {}
    for f in files:
        z = np.load(f)
        for n, l in zip(z["names"], z[key].astype(np.float32)): acc.setdefault(str(n), []).append(l)
    return {n: np.mean(v, 0) for n, v in acc.items()}, {n: len(v) for n, v in acc.items()}


def cmd_base():
    meta, g = load_truth()
    # auto-test de mon score contre une version naïve par ensembles
    a = RNG.random((6, 128, 128)) > 0.5; b = RNG.random((6, 128, 128)) > 0.3; a[0] = False; b[0] = False; a[1] = True
    assert np.allclose(miou_vec(a, b), [miou_naif(x, y) for x, y in zip(a, b)]), "score vectorisé != score naïf"
    print("[auto-test] score vectorisé = score naïf sur 6 cas (dont tout-0 / tout-0 et tout-1)")
    idx = {m["image_name"]: i for i, m in enumerate(meta)}
    res_all = {n for n, i in idx.items() if i % 5 == 0}
    for me in METHODS:
        k = [i for i, m in enumerate(meta) if m["method"] == me]
        print(f"[effectifs] {me}: {len(k)} images, réserve globale (index % 5 == 0) = {sum(i % 5 == 0 for i in k)}, pleines dans la réserve = {sum(meta[i]['full_watermark'] for i in k if i % 5 == 0)}")
    out = {}; per = {}; test_pred = {}; train_pred = {}
    for me in ("stegastamp", "trustmark"):
        d = HERE / "adem" / "t1-reel" / "runs" / "mixr" / me
        fr = sorted(glob.glob(str(d / "fold*" / "res.npz"))); fo = sorted(glob.glob(str(d / "fold*" / "oof.npz"))); ft = sorted(glob.glob(str(d / "fold*" / "test.npz")))
        print(f"\n== {me} : {len(fr)} plis (res), {len(fo)} (oof), {len(ft)} (test) ; args : {(d / 'fold0' / 'args.json').read_text()[:400]}")
        res, cnt = avg_folds(fr); names = sorted(res, key=lambda n: idx[n])
        own = {m["image_name"] for m in meta if m["method"] == me}
        # contrôles d'intégrité des plis
        ok_res = set(res) == (own & res_all) and set(cnt.values()) == {len(fr)}
        oofn = [set(map(str, np.load(f)["names"])) for f in fo]
        disj = sum(len(a_ & b_) for i_, a_ in enumerate(oofn) for b_ in oofn[i_ + 1:]) == 0
        cover = set().union(*oofn) == own - res_all
        print(f"[plis] réserve = exactement les index % 5 == 0 du filigrane, présente dans chaque pli : {ok_res} ; plis hors-pli disjoints : {disj} ; hors-pli ∪ réserve = 170 : {cover and len(own) == 170}")
        P = np.stack([res[n] for n in names]) > 0
        s, full = report(f"{me} réserve, 5 plis moyennés, seuil 0", names, P, meta, g); per[me] = (names, s, full)
        for f in fr:
            z = np.load(f); nn = [str(n) for n in z["names"]]
            report(f"   pli {Path(f).parent.name} seul (réserve)", nn, z["logits"].astype(np.float32) > 0, meta, g)
        oo = {}
        for f in fo:
            z = np.load(f); oo.update({str(n): l for n, l in zip(z["names"], z["logits"].astype(np.float32))})
        on = sorted(oo, key=lambda n: idx[n]); so, _ = report(f"{me} hors-pli (CV, 1 modèle par image)", on, np.stack([oo[n] for n in on]) > 0, meta, g)
        tst, _ = avg_folds(ft); test_pred[me] = tst
        train_pred[me] = {**{n: res[n] > 0 for n in names}, **{n: oo[n] > 0 for n in on}}
        out[me] = dict(reserve=float(s.mean()), ic=boot(s), cv=float(so.mean()), ic_cv=boot(so), n=len(s))
    # PixelSeal run_D_res512
    me = "pixelseal"; d = HERE / "adem" / "t1-pixelseal" / "run_D_res512"
    z = np.load(d / "pred_train.npz"); nn = [str(n) for n in z["names"]]; role = z["role"]; lg = z["logits"].astype(np.float32)
    print(f"\n== pixelseal run_D_res512 ; rôles : {dict(zip(*np.unique(role, return_counts=True)))}")
    ok = all((idx[n] % 5 == 0) == (r == 0) for n, r in zip(nn, role)); print(f"[plis] rôle 0 = exactement index % 5 == 0 : {ok}")
    r0 = [i for i, r in enumerate(role) if r == 0]
    s, full = report("pixelseal réserve (rôle 0), seuil 0", [nn[i] for i in r0], lg[r0] > 0, meta, g); per[me] = ([nn[i] for i in r0], s, full)
    for r, lab in ((1, "pli tenu à l'écart (rôle 1)"), (2, "images apprises (rôle 2)")):
        k = [i for i, x in enumerate(role) if x == r]
        if k: report(f"pixelseal {lab}", [nn[i] for i in k], lg[k] > 0, meta, g)
    out[me] = dict(reserve=float(s.mean()), ic=boot(s), n=len(s))
    zt = np.load(d / "pred_test.npz"); test_pred[me] = {str(n): l for n, l in zip(zt["names"], zt["logits"].astype(np.float32))}
    k01 = [i for i, x in enumerate(role) if x != 2]; train_pred[me] = {nn[i]: lg[i] > 0 for i in k01}
    # ensemble : moyenne des trois filigranes à poids égal (le test en a 340 chacun)
    means = []
    for _ in range(10000):
        means.append(np.mean([per[m][1][RNG.integers(0, len(per[m][1]), len(per[m][1]))].mean() for m in METHODS]))
    tot = np.mean([per[m][1].mean() for m in METHODS]); pool = np.concatenate([per[m][1] for m in METHODS])
    print(f"\n== ENSEMBLE réserve : moyenne des 3 filigranes {tot:.4f}  IC95 [{np.percentile(means, 2.5):.3f} ; {np.percentile(means, 97.5):.3f}] ; moyenne des 102 images {pool.mean():.4f} ; écart-type par image {pool.std():.3f}")
    print(f"   bruit attendu du score public (306 images, 30 % du test) : écart-type {pool.std() / np.sqrt(306):.4f} ; de la note finale (714 images) : {pool.std() / np.sqrt(714):.4f}")
    # fichier soumis
    try:
        import torch
        sub = torch.load(HERE / "adem" / "submission_v4.pt", weights_only=True); sm = {n: m.numpy().astype(bool) for n, m in zip(sub["image_names"], sub["masks"])}
        tm = json.load(open(REL / "test" / "metadata.json")); diff = 0; tot_n = 0
        for m in tm:
            n = m["image_name"]; tot_n += 1; diff += int((sm[n] != (test_pred[m["method"]][n] > 0)).any())
        print(f"\n[soumission_v4.pt] {len(sm)} masques, dtype {sub['masks'].dtype}, forme {tuple(sub['masks'].shape)} ; images dont le masque diffère des prédictions test recalculées : {diff} / {tot_n}")
    except Exception as e:
        print("soumission non relue :", repr(e)); tm = json.load(open(REL / "test" / "metadata.json"))
    # cohérence train / test sans vérité
    print("\n== Cohérence train (prédictions non apprises) / test, par filigrane")
    bym = {m["image_name"]: m for m in meta}
    for me in METHODS:
        tn = sorted(train_pred[me]); Ptr = np.stack([train_pred[me][n] for n in tn]); Gtr = np.stack([g[n] for n in tn])
        Pte = np.stack([test_pred[me][m["image_name"]] > 0 for m in tm if m["method"] == me])
        print(me, "\n   vérité train :", json.dumps(stats(Gtr)), "\n   préd. train  :", json.dumps(stats(Ptr)), "\n   préd. test   :", json.dumps(stats(Pte)))
    json.dump(out, open(Path(__file__).with_name("base.json"), "w"), indent=1)
    np.savez_compressed(Path(__file__).with_name("base_scores.npz"), **{f"{m}_names": np.array(per[m][0]) for m in METHODS}, **{f"{m}_s": per[m][1] for m in METHODS}, **{f"{m}_full": per[m][2] for m in METHODS})


def paired(tag, names, s):
    """Écart apparié contre la base 1227 sur les images de réserve communes."""
    b = np.load(Path(__file__).with_name("base_scores.npz"))
    for me in METHODS:
        bn = {str(n): v for n, v in zip(b[f"{me}_names"], b[f"{me}_s"])}; k = [i for i, n in enumerate(names) if n in bn]
        if not k: continue
        a_ = np.array([s[i] for i in k]); b_ = np.array([bn[names[i]] for i in k]); d, lo, hi, pr = boot_diff(a_, b_)
        sig = "SIGNIFICATIF" if lo > 0 or hi < 0 else "non significatif"
        print(f"   contre base 1227 [{me}] n={len(k)} : {a_.mean():.4f} contre {b_.mean():.4f}, écart {d:+.4f} IC95 [{lo:+.3f} ; {hi:+.3f}], P(mieux) {pr:.2f} -> {sig}")


def cmd_reel(tag):
    """Run à 5 plis de t1-reel : réserve (plis moyennés) et hors-pli."""
    meta, g = load_truth(check_png=False); idx = {m["image_name"]: i for i, m in enumerate(meta)}
    for me in METHODS:
        d = HERE / "adem" / "t1-reel" / "runs" / tag / me
        fr = sorted(f for f in glob.glob(str(d / "fold*" / "res.npz")) if (Path(f).parent / "done").exists())
        if not fr: continue
        res, cnt = avg_folds(fr); names = sorted(res, key=lambda n: idx[n])
        assert all(idx[n] % 5 == 0 for n in names), "image hors réserve dans res.npz"
        s, _ = report(f"{tag}/{me} réserve, {len(fr)} plis moyennés", names, np.stack([res[n] for n in names]) > 0, meta, g); paired(tag, names, s)
        oo = {}; seen = 0
        for f in fr:
            z = np.load(Path(f).with_name("oof.npz")); seen += len(z["names"]); oo.update({str(n): l for n, l in zip(z["names"], z["logits"].astype(np.float32))})
        assert seen == len(oo) and not any(idx[n] % 5 == 0 for n in oo), "plis hors-pli non disjoints ou réserve dedans"
        on = sorted(oo); so, _ = report(f"{tag}/{me} hors-pli ({len(fr)} plis)", on, np.stack([oo[n] for n in on]) > 0, meta, g)
        if True:                                              # écart apparié hors-pli contre mixr (même découpage, images communes)
            bo = {}
            for f in sorted(glob.glob(str(HERE / "adem" / "t1-reel" / "runs" / "mixr" / me / "fold*" / "oof.npz"))):
                z = np.load(f); bo.update({str(n): l.astype(np.float32) > 0 for n, l in zip(z["names"], z["logits"])})
            if set(on) <= set(bo):
                sb = miou_vec(np.stack([bo[n] for n in on]), np.stack([g[n] for n in on])); d_, lo, hi, pr = boot_diff(so, sb)
                print(f"   hors-pli contre mixr [{me}] n={len(on)} : écart {d_:+.4f} IC95 [{lo:+.3f} ; {hi:+.3f}], P(mieux) {pr:.2f} -> {'SIGNIFICATIF' if lo > 0 or hi < 0 else 'non significatif'}")


def cmd_ps(run):
    """Run de t1-pixelseal (pred_train.npz avec rôles : 0 réserve, 1 pli tenu à l'écart, 2 appris)."""
    meta, g = load_truth(check_png=False); idx = {m["image_name"]: i for i, m in enumerate(meta)}
    z = np.load(HERE / "adem" / "t1-pixelseal" / run / "pred_train.npz"); nn = [str(n) for n in z["names"]]; role = z["role"]; lg = z["logits"].astype(np.float32)
    assert all((idx[n] % 5 == 0) == (r == 0) for n, r in zip(nn, role)), "rôle 0 != réserve"
    for r, lab in ((0, "réserve"), (1, "pli tenu à l'écart"), (2, "APPRIS (ne compte pas)")):
        k = [i for i, x in enumerate(role) if x == r]
        if not k: continue
        s, _ = report(f"{run} {lab}", [nn[i] for i in k], lg[k] > 0, meta, g)
        if r == 0: paired(run, [nn[i] for i in k], s)
    zb = np.load(HERE / "adem" / "t1-pixelseal" / "run_D_res512" / "pred_train.npz")
    if (zb["role"] == role).all() and run != "run_D_res512":
        k = np.where(role != 2)[0]; G = np.stack([g[nn[i]] for i in k])
        a_ = miou_vec(lg[k] > 0, G); b_ = miou_vec(zb["logits"].astype(np.float32)[k] > 0, G); d_, lo, hi, pr = boot_diff(a_, b_)
        print(f"   réserve + pli (n={len(k)}) contre run_D : {a_.mean():.4f} contre {b_.mean():.4f}, écart {d_:+.4f} IC95 [{lo:+.3f} ; {hi:+.3f}], P(mieux) {pr:.2f} -> {'SIGNIFICATIF' if lo > 0 or hi < 0 else 'non significatif'}")
    else: print("   (rôles différents de run_D : pas de comparaison appariée sur le pli)")


def cmd_pt(path):
    """Fichier .pt de masques du TRAIN (contrôle d'un assembleur) : score sur la réserve seulement."""
    import torch
    meta, g = load_truth(check_png=False); idx = {m["image_name"]: i for i, m in enumerate(meta)}; bym = {m["image_name"]: m for m in meta}
    d = torch.load(path, weights_only=True); pm = {n: m.numpy().astype(bool) for n, m in zip(d["image_names"], d["masks"])}
    tot = []
    for me in METHODS:
        for lab, cond in (("réserve", lambda i: i % 5 == 0), ("HORS réserve (réglé dessus)", lambda i: i % 5 != 0)):
            nn = [n for n in pm if bym[n]["method"] == me and cond(idx[n])]
            s, _ = report(f"{Path(path).name} {me} {lab}", nn, np.stack([pm[n] for n in nn]), meta, g)
            if lab == "réserve": paired(path, nn, s); tot.append(s.mean())
    print(f"   moyenne des 3 filigranes sur la réserve : {np.mean(tot):.4f}")


if __name__ == "__main__":
    {"base": cmd_base, "reel": cmd_reel, "ps": cmd_ps, "pt": cmd_pt}[sys.argv[1]](*sys.argv[2:])
