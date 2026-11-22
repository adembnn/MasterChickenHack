"""Fusion finale TrustMark : pleines = régression logistique par case (logit du réseau + cartes de re-tamponnage + cartes de l'effaceur), apprise sur les
pleines HORS réserve du train ; partielles = fichier d'entrée inchangé (rectangles déjà posés). Local, numpy seul.
Train : prédictions hors-pli (validation croisée à 4 plis hors réserve ; réserve prédite par le modèle appris sur tout le hors réserve).
Usage : python pfuse3.py <pred_train.npz> <pred_test.npz> <nom_sortie>   (les cartes sont lues dans out/)"""
import sys, json, glob, numpy as np
from pathlib import Path
import os
O = Path("/tmp/eac-t1-partage/t1-residus"); OUTD = Path(__file__).resolve().parent; R = Path(os.path.expanduser("~/eac/t1/release/train")); meta = json.loads((R / "metadata.json").read_text()); m128 = np.load(R / "masks_128.npz"); G = dict(zip(m128["names"].tolist(), m128["masks"]))
ptr, pte, name = sys.argv[1:4]
def miou(p, g):
    s = 0.0
    for c in (0, 1):
        u = ((p == c) | (g == c)).sum(); s += 1.0 if u == 0 else ((p == c) & (g == c)).sum() / u
    return s / 2
def load_maps(split):
    pz = np.load(O / f"maps_trustmark_{split}_pz.npz", allow_pickle=True); rm = np.load(O / f"maps_trustmark_{split}_rm.npz", allow_pickle=True); names = pz["names"].tolist(); assert rm["names"].tolist() == names
    rat = {}
    for f in sorted(glob.glob(str(O / f"maps_trustmark_{split}_pleine_s*of*.npz"))):
        m = np.load(f, allow_pickle=True); rat.update(dict(zip(m["names"].tolist(), m["ratio"].astype(np.float32))))
    lr = np.log(np.maximum(np.stack([rat[n] for n in names]), 1e-3)); lr = lr - np.median(lr, axis=(2, 3), keepdims=True)
    RMl = [np.clip(rm[k].astype(np.float32), -20, 20) for k in ("g", "zg", "e")]
    import os
    if os.environ.get("PINV") and (O / f"maps_trustmark_{split}_pinv.npz").exists():                 # variantes : message inversé (A), cosinus au résidu moyen des messages au hasard (F), E
        pv = np.load(O / f"maps_trustmark_{split}_pinv.npz", allow_pickle=True); assert pv["names"].tolist() == names
        for k in ("A", "F", "E"):                                                                    # 3 lissages -> on complète à 5 canaux en répétant le dernier
            v = np.clip(pv[k].astype(np.float32), -20, 20); RMl.append(np.concatenate([v, v[:, -1:], v[:, -1:]], 1))
    return names, pz["z"].astype(np.float32), lr, RMl, pz["r3"].astype(np.float32)
def feats(L, Z, lr, RM, r3, i):
    return np.concatenate([np.clip(L[i], -12, 12)[None], Z[i], Z[i] - np.median(Z[i], axis=(1, 2), keepdims=True), lr[i], np.full((1, 128, 128), r3[i], np.float32)] + [q[i] for q in RM] + [RM[1][i] - np.median(RM[1][i], axis=(1, 2), keepdims=True)]).reshape(-1, 128 * 128).T
def fit(X, y, it=300, l2=1e-3):
    mu = X.mean(0); sd = X.std(0) + 1e-6; Xs = (X - mu) / sd; w = np.zeros(X.shape[1], np.float32); b = 0.0; wt = np.where(y > 0, 0.5 / max(y.mean(), 1e-6), 0.5 / max(1 - y.mean(), 1e-6)).astype(np.float32)
    for _ in range(it):
        p = 1 / (1 + np.exp(-(Xs @ w + b))); gr = (p - y) * wt; w -= 1.0 * (Xs.T @ gr / len(y) + l2 * w); b -= 1.0 * gr.mean()
    return mu, sd, w, b
def apply(mdl, F_): mu, sd, w, b = mdl; return (((F_ - mu) / sd) @ w + b).reshape(128, 128)
ztr = np.load(ptr, allow_pickle=True); NTR = ztr["names"].tolist(); LTR = ztr["logits"].astype(np.float32); pos = {n: i for i, n in enumerate(NTR)}
names, Z, lr, RM, r3 = load_maps("train"); L = np.stack([LTR[pos[n]] for n in names]); g = np.stack([G[n] > 0 for n in names]); gi = {e["image_name"]: i for i, e in enumerate(meta)}; gidx = np.array([gi[n] for n in names]); res = gidx % 5 == 0; ok = r3 < 0.6
FT = [feats(L, Z, lr, RM, r3, i) for i in range(len(names))]
def train_on(idx): return fit(np.concatenate([FT[i] for i in idx]), np.concatenate([g[i].ravel() for i in idx]).astype(np.float32))
def thr_on(S, idx):
    ts = np.linspace(-3, 3, 25); return ts[int(np.argmax([np.mean([miou(S[i] > t, g[i]) for i in idx]) for t in ts]))]
S = {}; T = {}; fold = gidx % 5
for f in (1, 2, 3, 4):                                                # hors-pli sur le hors réserve
    a_ = np.where(~res & ok & (fold != f))[0]; b_ = np.where(~res & ok & (fold == f))[0]; m_ = train_on(a_); Sa = {i: apply(m_, FT[i]) for i in a_}; t_ = thr_on(Sa, a_)
    for i in b_: S[i] = apply(m_, FT[i]) - t_
allhr = np.where(~res & ok)[0]; MDL = train_on(allhr); Sall = {i: apply(MDL, FT[i]) for i in allhr}; THR = thr_on(Sall, allhr)
for i in np.where(res & ok)[0]: S[i] = apply(MDL, FT[i]) - THR
OUTTR = LTR.copy(); base = np.array([miou(L[i] > 0, g[i]) for i in range(len(names))]); new = base.copy()
for i, s in S.items(): OUTTR[pos[names[i]]] = np.clip(s, -12, 12); new[i] = miou(s > 0, g[i])
d = new - base
print("PFUSE3 base=%s | pleines TrustMark n=%d (toile fiable %d)" % (Path(ptr).name, len(names), ok.sum()))
print("  pleines hors réserve (hors-pli) : %.4f -> %.4f (%+.4f ± %.4f ; améliorées %d, dégradées %d)" % (base[~res].mean(), new[~res].mean(), d[~res].mean(), d[~res].std() / np.sqrt((~res).sum()), (d[~res] > 1e-6).sum(), (d[~res] < -1e-6).sum()))
print("  pleines RÉSERVE                 : %.4f -> %.4f (%+.4f ± %.4f ; n=%d ; améliorées %d, dégradées %d)" % (base[res].mean(), new[res].mean(), d[res].mean(), d[res].std() / np.sqrt(res.sum()), res.sum(), (d[res] > 1e-6).sum(), (d[res] < -1e-6).sum()))
info = {e["image_name"]: i for i, e in enumerate(meta) if e["method"] == "trustmark"}; allres = np.array([info[n] % 5 == 0 for n in NTR])
s0 = np.array([miou(LTR[i] > 0, G[n] > 0) for i, n in enumerate(NTR)]); s1 = np.array([miou(OUTTR[i] > 0, G[n] > 0) for i, n in enumerate(NTR)])
print("  TrustMark entier (170 images, fichier d'entrée -> sortie) : hors réserve %.4f -> %.4f | RÉSERVE %.4f -> %.4f (n=%d)" % (s0[~allres].mean(), s1[~allres].mean(), s0[allres].mean(), s1[allres].mean(), allres.sum()))
np.savez_compressed(OUTD / f"pred_train_{name}_trustmark.npz", names=np.array(NTR), logits=OUTTR.astype(np.float16))
if (O / "maps_trustmark_test_pz.npz").exists():
    zte = np.load(pte, allow_pickle=True); NTE = zte["names"].tolist(); LTE = zte["logits"].astype(np.float32); pt = {n: i for i, n in enumerate(NTE)}
    tn, Zt, lrt, RMt, r3t = load_maps("test"); Lt = np.stack([LTE[pt[n]] for n in tn]); OUTTE = LTE.copy(); k = 0
    for i in range(len(tn)):
        if r3t[i] < 0.6: OUTTE[pt[tn[i]]] = np.clip(apply(MDL, feats(Lt, Zt, lrt, RMt, r3t, i)) - THR, -12, 12); k += 1
    np.savez_compressed(OUTD / f"pred_test_{name}_trustmark.npz", names=np.array(NTE), logits=OUTTE.astype(np.float16))
    ch = np.array([((LTE[pt[n]] > 0) != (OUTTE[pt[n]] > 0)).mean() for n in tn])
    print("  test : %d images classées pleines, %d avec toile fiable et masque refait ; part propre prédite sur ces images : avant %.3f, après %.3f (train, vérité : %.3f) ; cases changées en moyenne %.3f" % (len(tn), k,
          np.mean([(LTE[pt[n]] <= 0).mean() for n, r in zip(tn, r3t) if r < 0.6]), np.mean([(OUTTE[pt[n]] <= 0).mean() for n, r in zip(tn, r3t) if r < 0.6]), np.mean([1 - g[i].mean() for i in range(len(names)) if ok[i]]), ch.mean()))
else: print("  (cartes du test absentes : fichier test non écrit)")
