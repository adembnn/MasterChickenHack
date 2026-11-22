"""Candidat N (défaut 11) de t1-assemblage contre le candidat 5 (dernier envoi, masques honnêtes reconstruits) et contre le candidat 6 :
écart apparié réserve / hors réserve / 510, par filigrane et par population ; fichier de test ; fuites signalées (rétrécissement du cadre TrustMark
mesuré réserve comprise, seuil R3 StegaStamp fixé à la main)."""
import sys, json, hashlib
from pathlib import Path
import numpy as np, torch
from verif import HERE, METHODS, RNG, miou_vec, boot, boot_diff, load_truth
N = sys.argv[1] if len(sys.argv) > 1 else "11"; AJ = sys.argv[2] if len(sys.argv) > 2 else "10"
A = HERE / "t1-assemblage"; meta, g = load_truth(check_png=False); idx = {m["image_name"]: i for i, m in enumerate(meta)}; bym = {m["image_name"]: m for m in meta}
z5 = np.load(Path(__file__).with_name("cand5_honnete.npz")); c5 = {str(a): m.astype(bool) for a, m in zip(z5["names"], z5["masks"])}
D = {"réserve": {}, "hors réserve": {}, "510": {}}; S = {}; D6 = {}
for me in METHODS:
    z = np.load(A / f"candidat_{N}_honnete_{me}.npz"); cN = {str(a): m.astype(bool) for a, m in zip(z["names"], z["masks"])}
    z = np.load(A / f"candidat_6_honnete_{me}.npz"); c6 = {str(a): m.astype(bool) for a, m in zip(z["names"], z["masks"])}
    assert set(cN) == set(c6) == {m["image_name"] for m in meta if m["method"] == me}
    for lab, cond in (("réserve", lambda i: i % 5 == 0), ("hors réserve", lambda i: i % 5 != 0), ("510", lambda i: True)):
        nn = [a for a in cN if cond(idx[a])]; G = np.stack([g[a] for a in nn]); sN = miou_vec(np.stack([cN[a] for a in nn]), G); s5 = miou_vec(np.stack([c5[a] for a in nn]), G); s6 = miou_vec(np.stack([c6[a] for a in nn]), G)
        full = np.array([bym[a]["full_watermark"] for a in nn]); d, lo, hi, pr = boot_diff(sN, s5); d6, lo6, hi6, _ = boot_diff(sN, s6); D[lab][me] = sN - s5; S.setdefault(lab, {})[me] = (sN.mean(), s5.mean(), s6.mean())
        if lab == "réserve": D6[me] = sN - s6; la, ha = boot(sN)
        print(f"[{me}] {lab:12s} n={len(nn):3d} candidat {N} {sN.mean():.4f}" + (f" IC95 [{la:.3f} ; {ha:.3f}]" if lab == "réserve" else "") + f" (pleines {sN[full].mean():.4f}, partielles {sN[~full].mean():.4f}) | candidat 5 {s5.mean():.4f} | écart apparié {d:+.4f} IC95 [{lo:+.3f} ; {hi:+.3f}] P(mieux) {pr:.2f} ; mieux / pires {(sN > s5 + 1e-9).sum()} / {(sN < s5 - 1e-9).sum()} | contre candidat 6 {d6:+.4f} [{lo6:+.3f} ; {hi6:+.3f}]")
        if lab != "510":
            for l2, k in (("pleines", full), ("partielles", ~full)):
                if (np.abs(sN[k] - s5[k]) > 1e-9).any(): d2, lo2, hi2, _ = boot_diff(sN[k], s5[k]); print(f"        {l2:10s} n={k.sum():3d} écart contre candidat 5 {d2:+.4f} IC95 [{lo2:+.3f} ; {hi2:+.3f}] ; mieux / pires {(sN[k] > s5[k] + 1e-9).sum()} / {(sN[k] < s5[k] - 1e-9).sum()}")
print()
for lab in D:
    bs = np.array([np.mean([D[lab][m][RNG.integers(0, len(D[lab][m]), len(D[lab][m]))].mean() for m in METHODS]) for _ in range(20000)])
    print(f"MOYENNE DES TROIS, {lab:12s} : candidat {N} {np.mean([S[lab][m][0] for m in METHODS]):.4f} | candidat 5 {np.mean([S[lab][m][1] for m in METHODS]):.4f} | candidat 6 {np.mean([S[lab][m][2] for m in METHODS]):.4f} | écart apparié contre candidat 5 {np.mean([D[lab][m].mean() for m in METHODS]):+.4f} IC95 [{np.percentile(bs, 2.5):+.4f} ; {np.percentile(bs, 97.5):+.4f}] ; P(écart >= 0,02) = {np.mean(bs >= 0.02):.3f}")
bs = np.array([np.mean([D6[m][RNG.integers(0, len(D6[m]), len(D6[m]))].mean() for m in METHODS]) for _ in range(20000)]); print(f"MOYENNE DES TROIS, réserve, contre candidat 6 : {np.mean([D6[m].mean() for m in METHODS]):+.4f} IC95 [{np.percentile(bs, 2.5):+.4f} ; {np.percentile(bs, 97.5):+.4f}]")
# variante : StegaStamp du candidat 6 (le changement StegaStamp est négatif sur la réserve)
v = {m: D["réserve"][m] for m in METHODS}; z = np.load(A / "candidat_6_honnete_stegastamp.npz"); nn = [str(a) for a in z["names"] if idx[str(a)] % 5 == 0]; G = np.stack([g[a] for a in nn]); c6 = {str(a): m.astype(bool) for a, m in zip(z["names"], z["masks"])}
v["stegastamp"] = miou_vec(np.stack([c6[a] for a in nn]), G) - miou_vec(np.stack([c5[a] for a in nn]), G); bs = np.array([np.mean([v[m][RNG.integers(0, len(v[m]), len(v[m]))].mean() for m in METHODS]) for _ in range(20000)])
print(f"VARIANTE (TrustMark du {N}, PixelSeal et StegaStamp du 6), réserve, contre candidat 5 : {np.mean([v[m].mean() for m in METHODS]):+.4f} IC95 [{np.percentile(bs, 2.5):+.4f} ; {np.percentile(bs, 97.5):+.4f}] ; P(>= 0,02) = {np.mean(bs >= 0.02):.3f}")
# ---- fichier de test
p = A / f"submission_candidat_{N}.pt"; print(f"\n[fichier] sha256 {hashlib.sha256(open(p, 'rb').read()).hexdigest()}")
sub = torch.load(p, weights_only=True); s5f = torch.load(A / "submission_candidat_5.pt", weights_only=True); tm = json.load(open(HERE / "release" / "test" / "metadata.json")); tn = [m["image_name"] for m in tm]
M = sub["masks"]; print(f"[fichier] clés {list(sub.keys())} ; forme {tuple(M.shape)} ; type {M.dtype} ; valeurs {sorted(torch.unique(M).tolist())} ; noms distincts {len(set(sub['image_names']))} ; égaux à ceux du test : {set(sub['image_names']) == set(tn)} ; même ordre que le candidat 5 : {list(sub['image_names']) == list(s5f['image_names'])}")
sN = {a: m.numpy().astype(bool) for a, m in zip(sub["image_names"], sub["masks"])}; s5 = {a: m.numpy().astype(bool) for a, m in zip(s5f["image_names"], s5f["masks"])}
for me in METHODS:
    te = [m["image_name"] for m in tm if m["method"] == me]; ch = np.array([(sN[a] != s5[a]).mean() for a in te]); z = np.load(A / f"candidat_{N}_honnete_{me}.npz"); tr = np.array([(m.astype(bool) != c5[str(a)]).mean() for a, m in zip(z["names"], z["masks"])])
    print(f"[fichier] {me:10s} part filigranée test {np.mean([sN[a].mean() for a in te]):.3f} (candidat 5 : {np.mean([s5[a].mean() for a in te]):.3f}) ; images changées contre le candidat 5 : test {np.mean(ch > 0):.2f} (> 5 % des cases : {np.mean(ch > 0.05):.2f}) | train {np.mean(tr > 0):.2f} ({np.mean(tr > 0.05):.2f}) ; tout-0 au test {np.mean([not sN[a].any() for a in te]):.3f}, tout-1 {np.mean([sN[a].all() for a in te]):.3f}")
# ---- fuite 2 : rétrécissement du cadre TrustMark (+4, +4, -3, -3) mesuré sur des rectangles du train, réserve comprise
T = json.load(open(A / f"assemble_{AJ}.json")); f = np.load(A / "feat_train.npz"); acc = {str(a): float(v) for a, v in zip(f["names"], f["acc"])}
def bm(b):
    m = np.zeros((512, 512), np.float32); y0, x0, y1, x1 = [int(round(v)) for v in b]; m[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = 1
    return m.reshape(128, 4, 128, 4).mean((1, 3)) >= 0.5
H = {r["name"]: r for r in json.load(open(HERE / "t1-residus" / "boxes_trustmark_train_g13k_r3c.json")) if r.get("box") is not None}
B = {r["name"]: r for r in json.load(open(A / "boxtm2" / "boxes_trustmark_train_cadre.json"))}; th = T["pop"]["trustmark"]["th"]; z0 = T["choix"]["trustmark"]["partielles"]["p"]["box"]
cad = [n for n, r in B.items() if r.get("src") == "cadre"]; rect = [n for n in cad if H[n].get("rect") and H[n].get("gt_box") is not None]; rr = np.array([idx[n] % 5 == 0 for n in rect])
off = np.array([np.array(H[n]["gt_box"], float) - np.array(H[n]["frame"], float) for n in rect]); print(f"\n[cadre TrustMark] {len(cad)} cadres calés au train (réserve {sum(idx[n] % 5 == 0 for n in cad)}), dont {len(rect)} vrais rectangles (réserve {rr.sum()}) ; décalage vérité − cadre, médiane sur tous {np.median(off, 0).round(2)}, sans la réserve {np.median(off[~rr], 0).round(2)}, réserve seule {np.median(off[rr], 0).round(2) if rr.any() else None} ; appliqué : [4, 4, -3, -3]")
onr = np.round(np.median(off[~rr], 0))
for lab, cond in (("réserve", lambda i: i % 5 == 0), ("hors réserve", lambda i: i % 5 != 0)):
    nn = [n for n in cad if cond(idx[n]) and acc[n] <= th and B[n]["z"] >= z0]
    if not nn: continue
    G = np.stack([g[n] for n in nn]); fr = lambda n: np.array(H[n]["frame"], float)
    v = {"cadre brut": [bm(fr(n)) for n in nn], "cadre + [4, 4, -3, -3] (candidat)": [bm(fr(n) + [4, 4, -3, -3]) for n in nn], f"cadre + décalage appris sans la réserve {onr.astype(int).tolist()}": [bm(fr(n) + onr) for n in nn], "boîte écrite": [bm(B[n]["box"]) for n in nn]}
    sc = {k: miou_vec(np.stack(m), G) for k, m in v.items()}; print(f"[cadre TrustMark, {lab}, n={len(nn)} boîtes rendues] " + " | ".join(f"{k} : {s.mean():.4f}" for k, s in sc.items()))
    if lab == "réserve":
        k = list(sc); d, lo, hi, _ = boot_diff(sc[k[1]], sc[k[2]]); print(f"     effet de la fuite (appliqué − appris sans la réserve) : {d:+.4f} IC95 [{lo:+.4f} ; {hi:+.4f}] sur ces images, soit {d * len(nn) / 39:+.4f} sur la réserve TrustMark")
# ---- fuite 1 : seuil R3 StegaStamp
BS = json.load(open(A / "boxnet2" / "boxes_stegastamp_train_cadre.json")); print(f"\n[cadre StegaStamp] clés d'une entrée : {sorted(set().union(*[set(r) for r in BS]))}")
mv = [r for r in BS if r.get("box") is not None and r.get("box0") is not None and list(r["box"]) != list(r["box0"])]; th = T["pop"]["stegastamp"]["th"]; z0 = T["choix"]["stegastamp"]["partielles"]["p"]["box"]
print(f"[cadre StegaStamp] {len(mv)} boîtes recalées sur {len(BS)} au train, dont {sum(idx[r['name']] % 5 == 0 for r in mv)} de la réserve ; valeurs de r3 des recalées : {sorted(round(float(r['r3']), 2) for r in mv if 'r3' in r)}")
for lab, cond in (("réserve", lambda i: i % 5 == 0), ("hors réserve", lambda i: i % 5 != 0)):
    rs = [r for r in mv if cond(idx[r["name"]]) and acc.get(r["name"], 1) <= th and r["z"] >= z0]
    if not rs: print(f"   {lab} : aucune boîte recalée rendue"); continue
    G = np.stack([g[r["name"]] for r in rs]); a = miou_vec(np.stack([bm(r["box"]) for r in rs]), G); b = miou_vec(np.stack([bm(r["box0"]) for r in rs]), G); print(f"   {lab} : {len(rs)} boîtes recalées rendues ; recalée {a.mean():.4f} | non recalée {b.mean():.4f} | écart {a.mean() - b.mean():+.4f}, soit {(a - b).sum() / (33 if lab == 'réserve' else 137):+.4f} sur StegaStamp {lab}")
