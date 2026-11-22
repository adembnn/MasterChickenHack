"""A partir des etats (public, nature) compatibles avec les 10 scores : quel envoi apprend le plus, et quel fichier a la meilleure esperance ? Aucune requete."""
import sys, glob, itertools, collections, numpy as np
exec(open("pubdup.py", encoding="utf-8").read().split("x = rng.integers(0, 6")[0].replace("rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)", "rng = np.random.default_rng(0)").replace("C = int(sys.argv[2]) if len(sys.argv) > 2 else 1500; STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 4000", ""))
X = np.unique(np.concatenate([np.load(f) for f in glob.glob("pubdup_*.npy")]), axis=0)
print(f"{len(X)} etats distincts compatibles avec les 10 scores ; guichets publics : {(X > 0).sum(1).mean():.1f} en moyenne")
print("P(public)            :", " ".join(f"{e}={(X[:, ix[e]] > 0).mean():.2f}" for e in E))
print("P(copie | public)    :", " ".join(f"{e}={((X[:, ix[e]] == 1).sum() / max(1, (X[:, ix[e]] > 0).sum())):.2f}" for e in E))
i17, i18 = ix["17"], ix["18"]
both = (X[:, i17] > 0) & (X[:, i18] > 0)
print(f"ep-17 et ep-18 tous deux publics : {both.mean():.2f} des etats ; parmi eux, MEME modele : {(X[both, i17] == X[both, i18]).mean():.2f}, modeles differents : {(X[both, i17] != X[both, i18]).mean():.2f}")
print("P(ep-17 copie publique) =", f"{(X[:, i17] == 1).mean():.2f}", "; P(ep-18 copie publique) =", f"{(X[:, i18] == 1).mean():.2f}")
dup = collections.Counter()
for fi, f in enumerate(FAM):
    for a_, b_ in itertools.combinations(f, 2):
        m_ = (X[:, ix[a_]] > 1) & (X[:, ix[a_]] == X[:, ix[b_]])
        dup[(a_, b_)] = m_.mean()
print("paires « meme variante » les plus frequentes :", " ".join(f"{a_}={b_}:{v:.2f}" for (a_, b_), v in dup.most_common(8)))
def dist(iso, groups=G, back=()):
    lab = np.full(33, -1)
    for e in iso: lab[ix[e]] = 100 + ix[e]
    for k, g in enumerate(groups):
        for e in g: lab[ix[e]] = 200 + k
    for e in back: lab[ix[e]] = -1
    pr = (lab[A] == lab[B]).astype(np.float32)
    s = np.round(scores(X, pr[None, :])[:, 0] * 20) / 20
    return s
tests = {"1967 + ep-01": dict(iso=FP8 + ["01"]), "1967 + ep-04": dict(iso=FP8 + ["04"]), "1967 avec 17 | 18 separes": dict(iso=FP8 + ["17", "18"], groups=()),
         "1967, ep-17 remis en base, ep-18 seul": dict(iso=FP8 + ["18"], groups=()), "1967, ep-18 remis en base, ep-17 seul": dict(iso=FP8 + ["17"], groups=()),
         "1967 + ep-26": dict(iso=FP8 + ["26"]), "1967 + ep-25": dict(iso=FP8 + ["25"]), "1967 + {10, 13}": dict(iso=FP8 + ["10", "13"]), "1967 + ep-21": dict(iso=FP8 + ["21"]), "1967 + ep-32": dict(iso=FP8 + ["32"]),
         "1967 + {33, 30, 31}": dict(iso=FP8 + ["33", "30", "31"]), "1967 + {02, 08, 29}": dict(iso=FP8 + ["02", "08", "29"]), "1967 + {23, 20, 16}": dict(iso=FP8 + ["23", "20", "16"]),
         "1967 + ep-05 et ep-15 regroupes": dict(iso=["22"], groups=(["17", "18"], ["05", "15"]))}
print("envoi : P(hausse) | P(inchange) | P(baisse) | gain moyen si hausse | esperance du max(score, 0,60) - 0,60")
for name, kw in tests.items():
    s = dist(**kw); up = s > 0.625; dn = s < 0.575
    print(f"  {name:40s} {up.mean():.2f} | {(~up & ~dn).mean():.2f} | {dn.mean():.2f} | {(s[up] - 0.6).mean() if up.any() else 0:+.3f} | {np.maximum(s - 0.6, 0).mean():+.4f}")
