"""t2-oracle : hypothese « classes entieres » de la session relais (aucune requete). Le public note 6 classes de configuration sur 11,
les memes dans les trois familles (18 guichets) ; chaque classe a la meme verite dans les trois familles (copie ou variante).
Classes : 4 alignees (T=1, prompt, persona, tronque) ; 3 « simples » {17,04,25}/{06,32,26}/{13,19,10} et 4 « intermediaires »
{12,24,01,18}/{22,28,03,21}/{15,05,11,09}, alignement entre familles inconnu : toutes les permutations (3!^2 x 4!^2 = 20 736).
Pour chaque alignement, chaque choix de 6 classes publiques (462) et chaque verite des 6 classes (64) : les 12 scores."""
import sys, itertools, collections, time, numpy as np
sys.argv = ["x", "a4", "S", "0"]
exec(open("E:/EAC/reports/critique/t2-oracle-rejet.py").read().split("if __name__")[0])
ALIGN = [("33","30","31"), ("02","08","29"), ("23","20","16"), ("14","07","27")]
SIMP = [["17","04","25"], ["06","32","26"], ["13","19","10"]]
INTER = [["12","24","01","18"], ["22","28","03","21"], ["15","05","11","09"]]
SUBS6 = list(itertools.combinations(range(11), 6)); TR = np.array(list(itertools.product([0, 1], repeat=6)), bool)
hits = []; t0 = time.time(); nconf = 0
for ps1 in itertools.permutations(range(3)):
  for ps2 in itertools.permutations(range(3)):
    simp = [(SIMP[0][i], SIMP[1][ps1[i]], SIMP[2][ps2[i]]) for i in range(3)]
    for pi1 in itertools.permutations(range(4)):
      for pi2 in itertools.permutations(range(4)):
        inter = [(INTER[0][i], INTER[1][pi1[i]], INTER[2][pi2[i]]) for i in range(4)]
        CLS = ALIGN + simp + inter; cidx = np.array([[ix[e] for e in c] for c in CLS])
        P = np.zeros((len(SUBS6), 33), bool)
        for q, sub in enumerate(SUBS6): P[q, cidx[list(sub)].ravel()] = True
        m = np.repeat(P[:, A] & P[:, B], len(TR), 0)
        lab = np.zeros((len(SUBS6), len(TR), 33), int)
        for q, sub in enumerate(SUBS6):
            for j, c in enumerate(sub):
                for e in cidx[c]: lab[q, TR[:, j], e] = 100 + e
        lab = lab.reshape(-1, 33)
        s = scores(lab, m); ok = (np.abs(s - OBS[None, :]) <= 0.025 + 1e-6).all(1); nconf += len(lab)
        for t in np.nonzero(ok)[0]: hits.append((tuple(map(tuple, CLS)), SUBS6[t // len(TR)], tuple(TR[t % len(TR)])))
print(f"configurations essayees : {nconf:.3g} ; compatibles avec les 12 scores : {len(hits)} ({time.time() - t0:.0f} s)")
c = collections.Counter()
for CLS, sub, tr in hits: c[" ; ".join(f"{'/'.join(CLS[k])}={'VAR' if v else 'copie'}" for k, v in zip(sub, tr))] += 1
for d, n in c.most_common(40): print(n, "|", d)
