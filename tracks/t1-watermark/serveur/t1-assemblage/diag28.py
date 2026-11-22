import numpy as np, sys
sys.path.insert(0, "."); from t1lib import *
z = np.load("../t1-reel/f_s1/pred_train_stegastamp.npz"); raw = dict(zip([str(k) for k in z["names"]], z["logits"].astype(np.float32) > 0))
c = np.load("candidat_28_honnete_stegastamp.npz"); rows = {r["image_name"]: r for r in load_train("stegastamp")}; f = np.load("feat_train.npz"); acc = dict(zip(f["names"].tolist(), f["acc"]))
D = []
for n, mk in zip(c["names"].tolist(), c["masks"]):
    r = rows[n]; a = float(miou(raw[n][None], r["gt"][None])[0]); b = float(miou(mk[None], r["gt"][None])[0]); D.append((b - a, n, r["full"], r["reserve"], a, b, float(acc[n]), float(r["gt"].mean()), float(raw[n].mean()), float(mk.mean())))
D.sort(); res = np.array([d[3] for d in D]); dd = np.array([d[0] for d in D]); full = np.array([d[2] for d in D]); pp = np.array([d[6] > 0.68 for d in D])
print("assemblage - carte brute : reserve %+.4f, hors reserve %+.4f | par population PREDITE : pleines %+.4f (n=%d), partielles %+.4f (n=%d)" % (dd[res].mean(), dd[~res].mean(), dd[pp].mean(), pp.sum(), dd[~pp].mean(), (~pp).sum()))
print("images mal routees (vraie population != predite) :", int((full != pp).sum()), "| ecart moyen sur elles %+.3f" % dd[full != pp].mean())
print("les 8 plus degradees (ecart, nom, vraie pleine, reserve, brut, assemble, bits image, part vraie, part brute, part assemblee) :")
for d in D[:8]: print("   %+.3f %s pleine=%s res=%s brut %.3f -> %.3f | bits %.2f | part filigranee vraie %.2f, brute %.2f, assemblee %.2f" % d)
