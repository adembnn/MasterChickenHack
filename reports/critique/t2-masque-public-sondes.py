"""A partir des etats (public, nature) compatibles avec les 10 scores : quel envoi apprend le plus, et quel fichier a la meilleure esperance ? Aucune requete."""
import sys, glob, itertools, collections, numpy as np
exec(open("pubmask.py", encoding="utf-8").read().split("x = rng.integers(0, 3")[0].replace("rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)", "rng = np.random.default_rng(0)").replace("C = int(sys.argv[2]) if len(sys.argv) > 2 else 1500; STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 4000", ""))
X = np.unique(np.concatenate([np.load(f) for f in glob.glob("pubmask_*.npy")]), axis=0)
print(f"{len(X)} etats distincts compatibles avec les 10 scores ; guichets publics : {(X > 0).sum(1).mean():.1f} en moyenne")
print("P(public)            :", " ".join(f"{e}={(X[:, ix[e]] > 0).mean():.2f}" for e in E))
print("P(copie | public)    :", " ".join(f"{e}={((X[:, ix[e]] == 1).sum() / max(1, (X[:, ix[e]] > 0).sum())):.2f}" for e in E))
base = FP8
def dist(iso, groups=G):
    s = np.round(scores(X, pred(iso, groups)[None, :])[:, 0] * 20) / 20
    c = collections.Counter(np.round(s, 2)); H = -sum(n / len(s) * np.log2(n / len(s)) for n in c.values())
    return c, H, s
cands = {}
untested = ["01","02","14","23","25","33","06","07","08","20","26","30","10","13","16","27","29","31","04","21","32"]
for e in untested: cands[e] = base + [e]
for a, b in [("10","13"),("01","25"),("26","06"),("13","27"),("10","27"),("29","31"),("01","26"),("25","26"),("10","26"),("13","26"),("01","10"),("01","13"),("25","10"),("25","13")]:
    cands[a + "+" + b] = base + [a, b]
for t in [("10","13","26"),("01","25","26"),("01","10","26"),("01","25","10","13","26"),("10","13","27","29","31","16"),("06","07","08","20","26","30"),("01","02","14","23","25","33")]:
    cands["+".join(t)] = base + list(t)
rows = []
for name, iso in cands.items():
    c, H, s = dist(iso); rows.append((H, name, c, (s > 0.625).mean(), (s < 0.575).mean()))
rows.sort(reverse=True)
print("envoi (1967 + guichets isoles) : information (bits) | P(hausse) | P(baisse) | scores possibles")
for H, name, c, up, dn in rows[:22]:
    print(f"  {name:28s} {H:.2f} | {up:.2f} | {dn:.2f} | " + " ".join(f"{k:.2f}:{n / len(X):.2f}" for k, n in sorted(c.items())))
