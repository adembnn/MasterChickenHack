"""A partir des etats (public, nature) compatibles avec les 10 scores : quel envoi apprend le plus, et quel fichier a la meilleure esperance ? Aucune requete."""
import sys, glob, itertools, collections, numpy as np
exec(open("pubmask.py", encoding="utf-8").read().split("x = rng.integers(0, 3")[0].replace("rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)", "rng = np.random.default_rng(0)").replace("C = int(sys.argv[2]) if len(sys.argv) > 2 else 1500; STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 4000", ""))
X = np.unique(np.concatenate([np.load(f) for f in glob.glob("pubmask_*.npy")]), axis=0)
print(f"{len(X)} etats distincts compatibles avec les 10 scores ; guichets publics : {(X > 0).sum(1).mean():.1f} en moyenne")
print("P(public)            :", " ".join(f"{e}={(X[:, ix[e]] > 0).mean():.2f}" for e in E))
print("P(copie | public)    :", " ".join(f"{e}={((X[:, ix[e]] == 1).sum() / max(1, (X[:, ix[e]] > 0).sum())):.2f}" for e in E))
base = FP8 + ["26"]
def lab_pred(iso, groups=G, relabel=None):
    lab = np.full(33, -1)
    for e in iso: lab[ix[e]] = 100 + ix[e]
    for k, g in enumerate(groups):
        for e in g: lab[ix[e]] = 200 + k
    return (lab[A] == lab[B]).astype(np.float32)
def dist(pr):
    return np.round(scores(X, pr[None, :])[:, 0] * 20) / 20
s0 = scores(X, lab_pred(base)[None, :])[:, 0]
# erreurs publiques restantes dans le fichier retenu (base b26)
pub = X > 0; var = X == 2
isol = np.zeros(33, bool)
for e in base + ["17", "18"]: isol[ix[e]] = True
miss = (pub & var & ~isol[None, :]); wrong = (pub & ~var & isol[None, :])
print("variantes publiques non isolees par etat : moyenne %.2f ; repartition" % miss.sum(1).mean(), dict(sorted(collections.Counter(miss.sum(1).tolist()).items())))
for fi, nm in enumerate(("Qwen-3B", "Qwen-1.5B", "Llama")):
    print(f"   {nm} : P(au moins une variante publique non isolee) = {miss[:, fi * 11:(fi + 1) * 11].any(1).mean():.2f}")
print("copies publiques isolees a tort : moyenne %.2f" % wrong.sum(1).mean(), " ; P(ep-17 = ep-18 groupe a tort ou raison non lisible)")
print("guichets les plus souvent « variante publique non isolee » :", " ".join(f"{E[i]}={miss[:, i].mean():.2f}" for i in np.argsort(-miss.mean(0))[:14]))
tests = collections.OrderedDict()
for e in ["25", "10", "13", "06", "29", "08", "33", "02", "14", "23", "20", "30", "07", "16", "27", "31", "04", "21", "32"]: tests["+ ep-" + e] = lab_pred(base + [e])
tests["+ ep-10 + ep-13"] = lab_pred(base + ["10", "13"])
tests["ep-33 avec 17 (18 a part)"] = lab_pred(base + ["18"], groups=(["17", "33"],))
tests["ep-33 avec 17 et 18"] = lab_pred(base, groups=(["17", "18", "33"],))
tests["+ ep-09 + ep-11 seuls"] = lab_pred(base + ["09", "11"])
print("envoi (base b26 = 0,80) : P(hausse) | P(inchange) | P(baisse) | scores")
rows = []
for nm, pr in tests.items():
    s = dist(pr); c = collections.Counter(np.round(s, 2)); rows.append(((s > .825).mean(), nm, ((s <= .825) & (s >= .775)).mean(), (s < .775).mean(), c))
for up, nm, eq, dn, c in sorted(rows, reverse=True):
    print(f"  {nm:28s} {up:.2f} | {eq:.2f} | {dn:.2f} | " + " ".join(f"{k:.2f}:{n / len(X):.2f}" for k, n in sorted(c.items())))
