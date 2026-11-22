"""Inference conjointe (guichets publics, nature des guichets publics) a partir des 10 scores publics. Aucune requete.
Modele : le public = paires entre guichets d'un sous-ensemble S. Etat d'un guichet : 0 = non public, 1 = public et copie, 2 = public et variante
(chaque variante est son propre modele). Score = 0,1 + 0,9 x MCC des paires internes aux familles dont les deux guichets sont publics, arrondi a 0,05.
Echantillonnage par recuit sur de nombreuses chaines ; on garde les etats qui redonnent les 10 scores."""
import sys, itertools, numpy as np, collections
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
C = int(sys.argv[2]) if len(sys.argv) > 2 else 1500; STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
FAM = [["01","02","04","12","14","17","18","23","24","25","33"], ["03","06","07","08","20","21","22","26","28","30","32"], ["05","09","10","11","13","15","16","19","27","29","31"]]
E = [e for f in FAM for e in f]; ix = {e: i for i, e in enumerate(E)}
pairs = [(ix[a], ix[b]) for f in FAM for a, b in itertools.combinations(f, 2)]; A = np.array([p[0] for p in pairs]); B = np.array([p[1] for p in pairs])
def pred(iso, groups=()):
    lab = np.full(33, -1)
    for e in iso: lab[ix[e]] = 100 + ix[e]
    for k, g in enumerate(groups):
        for e in g: lab[ix[e]] = 200 + k
    return (lab[A] == lab[B]).astype(np.float32)
FP8 = ["05", "15", "22"]; G = (["17", "18"],)
SUBS = [("1803", pred(["14","23","07","28","16"]), .10), ("1882", pred(FP8), .40), ("1967", pred(FP8, G), .60), ("2052", pred(FP8 + ["17","18"]), .60),
        ("2176", pred(FP8 + ["11"], G), .60), ("2383", pred(FP8 + ["11","19"], G), .60), ("2491", pred(FP8 + ["28","03"], G), .60),
        ("2595", pred(FP8 + ["09","11"], G), .60), ("2707", pred(FP8 + ["12","24"], G), .60),
        ("2795", pred(FP8 + ["17","18","04","12","03","21","32","09","11"]), .30), ("2899", pred(FP8 + ["01"], G), .60), ("2998", pred(FP8 + ["26"], G), .80)]
PF = np.stack([s[1] for s in SUBS]); OBS = np.array([s[2] for s in SUBS], dtype=np.float32)
def scores(x, PFm=PF):
    pub = x > 0; var = x == 2
    m = pub[:, A] & pub[:, B]; t = ~var[:, A] & ~var[:, B]
    Mt = (m & t).astype(np.float32); Mn = (m & ~t).astype(np.float32)
    tp = Mt @ PFm.T; fp = Mn @ PFm.T; fn = Mt.sum(1, keepdims=True) - tp; tn = Mn.sum(1, keepdims=True) - fp
    d = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.1 + 0.9 * np.maximum(0, np.where(d > 0, (tp * tn - fp * fn) / np.maximum(d, 1e-9), 0))
def energy(x):
    s = scores(x); dev = np.maximum(0, np.abs(s - OBS[None, :]) - 0.025)
    dev[:, 0] = np.maximum(0, s[:, 0] - 0.125)                       # 1803 : 0,10 = MCC <= 0
    npub = (x > 0).sum(1)
    return (dev / 0.01) ** 2 @ np.ones(len(SUBS)) + ((npub - 18) / 4.0) ** 2 * 0.5, dev.max(1)
x = rng.integers(0, 3, (C, 33)).astype(np.int8)
en, _ = energy(x)
for step in range(STEPS):
    T = max(1.0, 8.0 * (1 - step / (0.7 * STEPS)))
    y = x.copy(); j = rng.integers(0, 33, C); y[np.arange(C), j] = (y[np.arange(C), j] + rng.integers(1, 3, C)) % 3
    e2, _ = energy(y); acc = rng.random(C) < np.exp(np.minimum(0, (en - e2) / T))
    x[acc] = y[acc]; en[acc] = e2[acc]
_, dv = energy(x)
ok = dv < 1e-6
X = np.unique(x[ok], axis=0)
print(f"{ok.sum()} chaines sur {C} redonnent les 10 scores ; {len(X)} etats distincts")
if len(X) == 0: sys.exit()
np.save(f"pubmask_{sys.argv[1] if len(sys.argv) > 1 else 0}.npy", X)
print("guichets publics par etat : moyenne %.1f (min %d, max %d)" % ((X > 0).sum(1).mean(), (X > 0).sum(1).min(), (X > 0).sum(1).max()))
print("P(public) :", " ".join(f"{e}={(X[:, ix[e]] > 0).mean():.2f}" for e in E))
print("P(public et copie) :", " ".join(f"{e}={(X[:, ix[e]] == 1).mean():.2f}" for e in E))
print("P(public et variante) :", " ".join(f"{e}={(X[:, ix[e]] == 2).mean():.2f}" for e in E))
