"""Hypothese « gabarit » : verite fixee (liste de variantes, chacune son modele), on cherche les ensembles S de guichets publics qui redonnent les 10 scores.
Usage : pubS.py <nom> <variantes separees par des virgules> [graine]. Sortie : part des chaines qui reussissent, P(public), issue des envois candidats."""
import sys, itertools, collections, numpy as np
name, VAR = sys.argv[1], set(sys.argv[2].split(",")); rng = np.random.default_rng(int(sys.argv[3]) if len(sys.argv) > 3 else 0)
C, STEPS = 3000, 2500
FAM = [["01","02","04","12","14","17","18","23","24","25","33"], ["03","06","07","08","20","21","22","26","28","30","32"], ["05","09","10","11","13","15","16","19","27","29","31"]]
E = [e for f in FAM for e in f]; ix = {e: i for i, e in enumerate(E)}
pairs = [(ix[a], ix[b]) for f in FAM for a, b in itertools.combinations(f, 2)]; A = np.array([p[0] for p in pairs]); B = np.array([p[1] for p in pairs])
isvar = np.array([e in VAR for e in E]); TRUE = (~isvar[A] & ~isvar[B])
def pred(iso, groups=()):
    lab = np.full(33, -1)
    for e in iso: lab[ix[e]] = 100 + ix[e]
    for k, g in enumerate(groups):
        for e in g: lab[ix[e]] = 200 + k
    return (lab[A] == lab[B]).astype(np.float32)
FP8 = ["05", "15", "22"]; G = (["17", "18"],)
SUBS = [("1803", pred(["14","23","07","28","16"]), .10), ("1882", pred(FP8), .40), ("1967", pred(FP8, G), .60), ("2052", pred(FP8 + ["17","18"]), .60),
        ("2176", pred(FP8 + ["11"], G), .60), ("2383", pred(FP8 + ["11","19"], G), .60), ("2491", pred(FP8 + ["28","03"], G), .60),
        ("2595", pred(FP8 + ["09","11"], G), .60), ("2707", pred(FP8 + ["12","24"], G), .60), ("2795", pred(FP8 + ["17","18","04","12","03","21","32","09","11"]), .30)]
PF = np.stack([s[1] for s in SUBS]); OBS = np.array([s[2] for s in SUBS], dtype=np.float32)
def scores(x, PFm=PF):
    m = x[:, A] & x[:, B]
    Mt = (m & TRUE[None, :]).astype(np.float32); Mn = (m & ~TRUE[None, :]).astype(np.float32)
    tp = Mt @ PFm.T; fp = Mn @ PFm.T; fn = Mt.sum(1, keepdims=True) - tp; tn = Mn.sum(1, keepdims=True) - fp
    d = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.1 + 0.9 * np.maximum(0, np.where(d > 0, (tp * tn - fp * fn) / np.maximum(d, 1e-9), 0))
def energy(x):
    s = scores(x); dev = np.maximum(0, np.abs(s - OBS[None, :]) - 0.025); dev[:, 0] = np.maximum(0, s[:, 0] - 0.125)
    return ((dev / 0.01) ** 2).sum(1) + ((x.sum(1) - 18) / 4.0) ** 2 * 0.5, dev.max(1)
x = rng.random((C, 33)) < 0.55; en, _ = energy(x)
for step in range(STEPS):
    T = max(1.0, 8.0 * (1 - step / (0.7 * STEPS)))
    y = x.copy(); j = rng.integers(0, 33, C); y[np.arange(C), j] ^= True
    e2, _ = energy(y); acc = rng.random(C) < np.exp(np.minimum(0, (en - e2) / T)); x[acc] = y[acc]; en[acc] = e2[acc]
_, dv = energy(x); ok = dv < 1e-6; X = np.unique(x[ok], axis=0)
print(f"== {name} : variantes {sorted(VAR)}")
print(f"   {ok.mean():.3f} des chaines redonnent les 10 scores ; {len(X)} ensembles publics distincts ; taille moyenne {X.sum(1).mean() if len(X) else 0:.1f}")
if len(X) == 0:
    s0 = scores(x[np.argsort(en)[:1]])[0]; print("   meilleur essai :", " ".join(f"{n}={v:.2f}" for (n, _, _), v in zip(SUBS, s0))); sys.exit()
print("   P(public) :", " ".join(f"{e}={X[:, ix[e]].mean():.2f}" for e in E))
tests = {"ep-01": FP8 + ["01"], "ep-26": FP8 + ["26"], "ep-21": FP8 + ["21"], "ep-24": FP8 + ["24"], "{10,13}": FP8 + ["10", "13"], "ep-25": FP8 + ["25"],
         "plain-like {17 retire, 04}": None, "T=1 {33,30,31}": FP8 + ["33","30","31"], "prompt {02,08,29}": FP8 + ["02","08","29"], "persona {23,20,16}": FP8 + ["23","20","16"],
         "tronque {14,07,27}": FP8 + ["14","07","27"], "filigrane ? {25,06,19}": FP8 + ["25","06","19"], "3e simple {26,10}": FP8 + ["26","10"], "LoRA faible {01,21,09}": FP8 + ["01","21","09"],
         "toutes les variantes de l'hypothese": None}
for nm, iso in tests.items():
    if nm.startswith("plain"): continue
    if iso is None: pr = pred(sorted(VAR))
    else: pr = pred(iso, G)
    s = np.round(scores(X, pr[None, :])[:, 0] * 20) / 20
    c = collections.Counter(np.round(s, 2))
    print(f"   {nm:36s} hausse {(s > .625).mean():.2f} | inchange {((s <= .625) & (s >= .575)).mean():.2f} | baisse {(s < .575).mean():.2f} | " + " ".join(f"{k:.2f}:{n / len(s):.2f}" for k, n in sorted(c.items())))
