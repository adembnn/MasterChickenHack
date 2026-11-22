"""Les cartes « rapport de re-tamponnage » de t1-residus (pleines TrustMark : ~0,4 dans le filigrane, plus haut dans un trou) aident-elles ma carte ?
1. valeur seule : AUC par image de -rapport contre le vrai masque (5 lissages) ;
2. fusion simple : logit fusionné = mon logit + alpha * (tau - rapport_k) ; alpha, tau, k réglés sur 4 plis d'images hors réserve et appliqués au 5e (emboîté), puis à la réserve.
Chiffres : mIoU officiel et IoU du trou sur les pleines pour ma carte seule, la leur seule, la fusion."""
import sys, json, glob, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent; HERE = ROOT.parent if (ROOT.parent / "kit").exists() else ROOT.parent.parent; CACHE = ROOT / "cache"
trm = json.loads((HERE / "release/train/metadata.json").read_text()); Y = np.load(CACHE / "Y128_train.npy"); name2i = {m["image_name"]: i for i, m in enumerate(trm)}
P = np.load(sys.argv[1]); mine = dict(zip(P["names"].tolist(), P["logits"].astype(np.float32)))
names, ratio, r3 = [], [], []
for f in sorted(glob.glob(sys.argv[2])):
    z = np.load(f); names += z["names"].tolist(); ratio.append(z["ratio"].astype(np.float32)); r3.append(z["r3"])
ratio = np.concatenate(ratio); r3 = np.concatenate(r3); print("cartes t1-residus :", ratio.shape, "| toiles fiables (r3 < 0,6) :", int((r3 < 0.6).sum()), "/", len(r3))
keep = [j for j, n in enumerate(names) if n in mine and n in name2i]; names = [names[j] for j in keep]; ratio = ratio[keep]; r3 = r3[keep]
idx = np.array([name2i[n] for n in names]); G = Y[idx].astype(bool); L = np.stack([mine[n] for n in names]); res = idx % 5 == 0
def iou0(p, g): u = ((~p) | (~g)).sum(); return 1.0 if u == 0 else ((~p) & (~g)).sum() / u
def miou(p, g): u1 = (p | g).sum(); return 0.5 * (iou0(p, g) + (1.0 if u1 == 0 else (p & g).sum() / u1))
def auc(s, l):
    s = s.ravel(); l = l.ravel(); o = np.argsort(s, kind="stable"); r = np.empty(len(o)); r[o] = np.arange(1, len(o) + 1); n1 = l.sum(); n0 = len(l) - n1; return (r[l].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)
both = np.array([g.any() and (~g).any() for g in G])
print("AUC par image (filigrané > propre), pleines hors réserve : ma carte", round(float(np.mean([auc(l, g) for l, g, b, r in zip(L, G, both, res) if b and not r])), 3),
      "| -rapport par lissage :", [round(float(np.mean([auc(-q[k], g) for q, g, b, r in zip(ratio, G, both, res) if b and not r])), 3) for k in range(ratio.shape[1])])
def score(Z, sel): return float(np.mean([miou(z > 0, g) for z, g in zip(Z[sel], G[sel])])), float(np.mean([iou0(z > 0, g) for z, g in zip(Z[sel], G[sel])]))
def fused(al, tau, k): return L + al * (tau - ratio[:, k])
grid = [(al, tau, k) for al in (0, 2, 5, 10, 20, 40) for tau in (0.45, 0.55, 0.65, 0.8, 1.0) for k in range(ratio.shape[1])]
tr = np.where(~res)[0]; fold = np.arange(len(tr)) % 5; nested = np.zeros((len(tr), 2))
for f in range(5):
    a_ = tr[fold != f]; b_ = tr[fold == f]; best = max(grid, key=lambda g_: score(fused(*g_), a_)[0]); Z = fused(*best)
    for j in b_: nested[np.where(tr == j)[0][0]] = (miou(Z[j] > 0, G[j]), iou0(Z[j] > 0, G[j]))
best = max(grid, key=lambda g_: score(fused(*g_), tr)[0]); print("meilleur réglage sur tout le hors-réserve (alpha, tau, lissage) :", best)
for nm, sel in (("HORS RÉSERVE", ~res), ("RÉSERVE", res)):
    a = score(L, sel); c = score(fused(*best), sel); only = max(((score(tau - ratio[:, k], sel), tau, k) for tau in (0.45, 0.55, 0.65, 0.8, 1.0) for k in range(ratio.shape[1])), key=lambda t: t[0][0])
    print(f"{nm} ({int(sel.sum())} pleines) : ma carte mIoU {a[0]:.3f}, trou {a[1]:.3f} | rapport seul (meilleur seuil, optimiste) mIoU {only[0][0]:.3f}, trou {only[0][1]:.3f} | fusion mIoU {c[0]:.3f}, trou {c[1]:.3f}" + (f" | fusion emboîtée mIoU {nested[:, 0].mean():.3f}, trou {nested[:, 1].mean():.3f}" if nm.startswith("HORS") else ""))
