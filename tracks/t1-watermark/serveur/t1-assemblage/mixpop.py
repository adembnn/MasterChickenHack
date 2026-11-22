"""Candidat mixte par population pour un filigrane : images predites pleines prises dans le candidat A, predites partielles dans le candidat B.
La population est predite comme dans cv5.py (justesse du decodeur sur l'image entiere > seuil regle sur les 4 plis d'apprentissage) : aucune verite utilisee.
Ecrit candidat_<n>_cv_<filigrane>.npz (les deux autres filigranes sont copies de B) et affiche l'ecart par population predite.
Usage : python mixpop.py 40 pixelseal pleines=38 partielles=39"""
import sys, shutil, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent)); from t1lib import *
n, m = sys.argv[1], sys.argv[2]; kv = dict(a.split("=") for a in sys.argv[3:]); A, B = kv["pleines"], kv["partielles"]; rng = np.random.default_rng(0)
ztr = np.load(ME / "feat_train.npz"); ACC = dict(zip(ztr["names"].tolist(), ztr["acc"])); rows = {r["image_name"]: r for r in load_train(m)}
a = np.load(ME / f"candidat_{A}_cv_{m}.npz"); b = np.load(ME / f"candidat_{B}_cv_{m}.npz"); names = a["names"].tolist(); assert names == b["names"].tolist()
G = np.stack([rows[k]["gt"] for k in names]); fold = a["fold"]; full_true = np.array([rows[k]["full"] for k in names]); acc = np.array([ACC[k] for k in names]); res = np.array([rows[k]["reserve"] for k in names]); pf = np.zeros(len(names), bool)
for f in range(5):
    fit = fold != f; cands = np.unique(acc); th = max(cands, key=lambda t: (np.mean((acc[fit] > t) == full_true[fit]), -abs(t - 0.75))); pf[fold == f] = acc[fold == f] > th
M = np.where(pf[:, None, None], a["masks"], b["masks"]); np.savez_compressed(ME / f"candidat_{n}_cv_{m}.npz", names=np.array(names), masks=M, fold=fold)
for o in METHODS:
    if o != m: shutil.copy(ME / f"candidat_{B}_cv_{o}.npz", ME / f"candidat_{n}_cv_{o}.npz")
sa, sb, sm = miou(a["masks"], G), miou(b["masks"], G), miou(M, G); ic = lambda d: "IC95 [%+.4f ; %+.4f]" % tuple(np.quantile([rng.choice(d, len(d)).mean() for _ in range(4000)], [.025, .975]))
print(f"{m} : population predite juste sur {int((pf == full_true).sum())} / {len(names)} images ; {int(pf.sum())} predites pleines, {int((~pf).sum())} partielles")
for nom, k in (("predites pleines", pf), ("predites partielles", ~pf)):
    d = (sb - sa)[k]; print(f"   {nom} (n={int(k.sum())}) : candidat {A} {sa[k].mean():.4f} | candidat {B} {sb[k].mean():.4f} | ecart {B} - {A} {d.mean():+.4f} {ic(d)} ({int((d > 1e-6).sum())} mieux, {int((d < -1e-6).sum())} pires) | reserve seule (n={int((k & res).sum())}) {(sb - sa)[k & res].mean():+.4f}")
d = sm - sa; print(f"   mixte {n} : {sm.mean():.4f} | contre {A} : {d.mean():+.4f} {ic(d)} ; sur la moyenne des trois filigranes {d.mean() / 3:+.4f} | reserve seule {d[res].mean():+.4f}")
