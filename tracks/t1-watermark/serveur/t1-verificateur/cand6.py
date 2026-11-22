"""Candidat 6 : (a) écart apparié contre le candidat 5 (masques honnêtes des deux) ; (b) fuite de la correction de décalage des boîtes TrustMark
(4 nombres = médiane du décalage relatif par bord, apprise sur les rectangles du train, réserve comprise) : on la ré-apprend SANS la réserve."""
import json
from pathlib import Path
import numpy as np
from verif import HERE, METHODS, RNG, miou_vec, boot, boot_diff, load_truth
A = HERE / "t1-assemblage"; meta, g = load_truth(check_png=False); idx = {m["image_name"]: i for i, m in enumerate(meta)}; bym = {m["image_name"]: m for m in meta}
z5 = np.load(Path(__file__).with_name("cand5_honnete.npz")); c5 = {str(a): m for a, m in zip(z5["names"], z5["masks"])}
D = {}
for me in METHODS:
    z = np.load(A / f"candidat_6_honnete_{me}.npz"); c6 = {str(a): m.astype(bool) for a, m in zip(z["names"], z["masks"])}
    for lab, cond in (("réserve", lambda i: i % 5 == 0), ("hors réserve", lambda i: i % 5 != 0)):
        nn = [a for a in c6 if cond(idx[a])]; G = np.stack([g[a] for a in nn]); s6 = miou_vec(np.stack([c6[a] for a in nn]), G); s5 = miou_vec(np.stack([c5[a] for a in nn]), G)
        d, lo, hi, pr = boot_diff(s6, s5); ch = np.mean([not np.array_equal(c6[a], c5[a]) for a in nn])
        print(f"[{me}] {lab:12s} n={len(nn):3d} candidat 6 {s6.mean():.4f} | candidat 5 {s5.mean():.4f} | écart apparié {d:+.4f} IC95 [{lo:+.3f} ; {hi:+.3f}] | images changées {ch:.2f} ; mieux / pires : {(s6 > s5 + 1e-9).sum()} / {(s6 < s5 - 1e-9).sum()}")
        if lab == "réserve": D[me] = s6 - s5
bs = np.array([np.mean([D[m][RNG.integers(0, len(D[m]), len(D[m]))].mean() for m in METHODS]) for _ in range(20000)])
print(f"MOYENNE DES TROIS, réserve, candidat 6 contre candidat 5 : {np.mean([D[m].mean() for m in METHODS]):+.4f} IC95 [{np.percentile(bs, 2.5):+.4f} ; {np.percentile(bs, 97.5):+.4f}]")

# ---- correction de décalage des boîtes TrustMark
B = {r["name"]: r for r in json.load(open(A / "boxtm" / "boxes_trustmark_train_affine.json"))}
raw = {}
for f in ("boxes_trustmark_train_s0of2.json", "boxes_trustmark_train_s1of2.json"):
    for r in json.load(open(A / "boxtm" / f)): raw[r["name"]] = r
R = [n for n, r in raw.items() if r.get("box") is not None and not r["full"] and r["rect"] and r["z"] >= 6 and "box_marge" in B.get(n, {})]
BM = np.array([B[n]["box_marge"] for n in R], float); GT = np.array([raw[n]["gt_box"] for n in R], float); S = np.stack([BM[:, 2] - BM[:, 0], BM[:, 3] - BM[:, 1]] * 2, 1)
res = np.array([idx[n] % 5 == 0 for n in R]); CALL = np.median((BM - GT) / S, 0); CNR = np.median(((BM - GT) / S)[~res], 0)
print(f"\n[correction] {len(R)} rectangles servent à l'apprendre, dont {res.sum()} de la réserve | correction apprise sur tous : {CALL.round(4)} | sans la réserve : {CNR.round(4)}")
used = [n for n, r in B.items() if "box_marge" in r]; rec = np.array([[round(t) for t in (np.array(B[n]["box_marge"], float) - CALL * np.array([B[n]["box_marge"][2] - B[n]["box_marge"][0], B[n]["box_marge"][3] - B[n]["box_marge"][1]] * 2))] for n in used])
print(f"[correction] la boîte écrite = boîte affinée − correction(tous) × taille : vrai pour {np.mean([list(a) == list(B[n]['box']) for a, n in zip(rec, used)]):.2f} des boîtes (0 = pas de correction appliquée, 1 = appliquée)")
def bm(b):
    m = np.zeros((512, 512), np.float32); y0, x0, y1, x1 = [int(v) for v in b]; m[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = 1
    return m.reshape(128, 4, 128, 4).mean((1, 3)) >= 0.5
def corr(n, C):
    b = np.array(B[n]["box_marge"], float); sz = np.array([b[2] - b[0], b[3] - b[1]] * 2); return [int(round(t)) for t in (b - C * sz)]
f = np.load(A / "feat_train.npz"); acc = {str(a): float(v) for a, v in zip(f["names"], f["acc"])}
for lab, cond in (("réserve", lambda i: i % 5 == 0), ("hors réserve", lambda i: i % 5 != 0)):
    nn = [n for n in used if cond(idx[n]) and acc[n] <= 0.76 and B[n]["z"] >= 5]; G = np.stack([g[n] for n in nn])
    v = {"boîte de bs.py (candidat 5)": [bm(B[n]["box0"]) for n in nn], "boîte affinée, sans correction": [bm(B[n]["box_marge"]) for n in nn],
         "affinée + correction apprise sur tous (candidat 6)": [bm(corr(n, CALL)) for n in nn], "affinée + correction apprise SANS la réserve": [bm(corr(n, CNR)) for n in nn], "boîte écrite dans le fichier": [bm(B[n]["box"]) for n in nn]}
    sc = {k: miou_vec(np.stack(m), G) for k, m in v.items()}
    print(f"[boîtes TrustMark rendues, {lab}, n={len(nn)}] " + " | ".join(f"{k} : {s.mean():.4f}" for k, s in sc.items()))
    d, lo, hi, pr = boot_diff(sc["affinée + correction apprise sur tous (candidat 6)"], sc["affinée + correction apprise SANS la réserve"]); print(f"     effet de la fuite (tous − sans réserve) sur ces images : {d:+.4f} IC95 [{lo:+.4f} ; {hi:+.4f}] ; rapporté à la réserve TrustMark (39 images) : {d * len(nn) / 39:+.4f}" if lab == "réserve" else f"     (tous − sans réserve) : {d:+.4f}")
