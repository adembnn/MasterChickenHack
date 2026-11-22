"""Simulation statistique de bout en bout du MCC_modele attendu sur les 33 endpoints T2.

Entree : les taux de divergence de generation gloutonne mesures sur le jumeau local
OLMo-2-1B (twin/results_exp1_a100.json, n=300 prompts). Aucune hypothese inventee :
p_type = 1 - identical_frac.

Pour chaque structure plausible de 33 endpoints :
  1. chaque endpoint recoit un type (copie du meme modele = bruit de fond ; ou variante
     alteree : int8, fp8, nf4, LoRA) et donc un vrai taux de divergence p vs la base ;
  2. on observe k ~ Binomial(N=100, p) divergences sur les N reponses du budget ;
  3. regle de decision : "meme modele que la base" si k/N <= tau, sinon isole ;
  4. MCC sur les 528 paires (famille supposee juste, paires inter-familles = TN).
On balaie tau pour trouver le seuil optimal, et on compare a la regle naive.

Python stdlib uniquement. Usage : python3 mcc_simulation.py [--trials 3000] [--seed 0]
"""
import argparse
import bisect
import json
import math
import random
from pathlib import Path

TWIN_PATH = Path(__file__).parent / "twin" / "results_exp1_a100.json"
N_HACK = 100
N_TWIN = 300
TAU_GRID = [i / 100 for i in range(0, 101)]

SAME_TYPES = ["noise_bs1", "noise_bs8", "noise_bs32", "noise_dtype_float16"]
ALTERED_TYPES = ["int8", "fp8_e4m3", "fp8_e5m2", "nf4", "lora_0.3", "lora_0.7", "lora_1.0", "lora_1.5"]
SUBTLE_TYPES = ["int8", "int8", "int8", "fp8_e4m3", "fp8_e4m3", "lora_0.3"]


# Copie des identical_frac de results_exp1_a100.json (n=300), utilisee si le JSON
# n'est pas dans le depot (twin/ vit sur Julich, hors git).
IDENTICAL_FRAC_FALLBACK = {
    "noise_bs1": 0.70, "noise_bs8": 0.66, "noise_bs32": 0.6833333333333333,
    "noise_dtype_float16": 0.6666666666666666,
    "fp8_e4m3": 0.22333333333333333, "fp8_e5m2": 0.06666666666666667,
    "int8": 0.5033333333333333, "nf4": 0.016666666666666666,
    "lora_0.3": 0.18, "lora_0.7": 0.016666666666666666, "lora_1.0": 0.0, "lora_1.5": 0.0,
}


def load_rates():
    if TWIN_PATH.exists():
        data = json.loads(TWIN_PATH.read_text())
        frac = {k: v["identical_frac"] for k, v in data["results"].items()}
    else:
        frac = IDENTICAL_FRAC_FALLBACK
    return {k: 1.0 - v for k, v in frac.items()}


RATES = load_rates()


# ---------------------------------------------------------------- binomial sampling
class Binomial:
    """Echantillonneur exact Binomial(n, p) par inversion de la CDF precalculee."""

    def __init__(self, n, p):
        self.n = n
        pmf = []
        for k in range(n + 1):
            if p <= 0:
                pmf.append(1.0 if k == 0 else 0.0)
            elif p >= 1:
                pmf.append(1.0 if k == n else 0.0)
            else:
                pmf.append(math.comb(n, k) * p ** k * (1 - p) ** (n - k))
        self.cdf = []
        acc = 0.0
        for v in pmf:
            acc += v
            self.cdf.append(acc)
        self.cdf[-1] = 1.0

    def sample(self, rng):
        return bisect.bisect_left(self.cdf, rng.random())

    def tail_prob(self, k):
        """P(K > k)."""
        return 1.0 - self.cdf[k] if 0 <= k < self.n else (1.0 if k < 0 else 0.0)


_BIN_CACHE = {}


def binom(n, p):
    key = (n, round(p, 6))
    if key not in _BIN_CACHE:
        _BIN_CACHE[key] = Binomial(n, p)
    return _BIN_CACHE[key]


# ---------------------------------------------------------------- structures
# Une structure = liste de familles ; une famille = liste de (is_same_model, type).
def fam(n_same, altered, rng):
    eps = [(True, SAME_TYPES[i % len(SAME_TYPES)]) for i in range(n_same)]
    eps += [(False, t) for t in altered]
    rng.shuffle(eps)
    return eps


def s1_three_by_eleven(rng):
    alt = ["int8", "fp8_e4m3", "fp8_e5m2", "nf4", "lora_0.3", "lora_0.7"]
    return [fam(5, alt, rng) for _ in range(3)]


def s2_six_medium(rng):
    sizes = [6, 6, 6, 5, 5, 5]
    fams, i = [], 0
    for s in sizes:
        alt = [ALTERED_TYPES[(i + j) % len(ALTERED_TYPES)] for j in range(s - 3)]
        i += s - 3
        fams.append(fam(3, alt, rng))
    return fams


def s3_dominant_plus_small(rng):
    big = fam(9, ["int8", "fp8_e4m3", "fp8_e5m2", "nf4", "lora_0.3", "lora_0.7"], rng)
    small = [fam(1, [ALTERED_TYPES[(2 * i) % 8], ALTERED_TYPES[(2 * i + 1) % 8]], rng) for i in range(6)]
    return [big] + small


def s4_eleven_triplets(rng):
    return [fam(2, [ALTERED_TYPES[i % len(ALTERED_TYPES)]], rng) for i in range(11)]


def s5_subtle_heavy(rng):
    return [fam(5, SUBTLE_TYPES, rng) for _ in range(3)]


def s6_random(rng):
    k = rng.randint(3, 7)
    cuts = sorted(rng.sample(range(1, 33), k - 1))
    sizes = [b - a for a, b in zip([0] + cuts, cuts + [33])]
    fams = []
    for s in sizes:
        n_same = max(1, round(s * rng.uniform(0.3, 0.6)))
        n_same = min(n_same, s)
        alt = [rng.choice(ALTERED_TYPES) for _ in range(s - n_same)]
        fams.append(fam(n_same, alt, rng))
    return fams


STRUCTURES = {
    "S1 3 familles x 11 (5 memes + 6 variantes)": s1_three_by_eleven,
    "S2 6 familles de 5-6 (3 memes + 2-3 variantes)": s2_six_medium,
    "S3 1 dominante (9 memes + 6 var.) + 6 triplets": s3_dominant_plus_small,
    "S4 11 triplets (2 memes + 1 variante)": s4_eleven_triplets,
    "S5 3 x 11, variantes subtiles (int8/fp8/LoRA 0.3)": s5_subtle_heavy,
    "S6 aleatoire (3-7 familles, 30-60 % memes)": s6_random,
}


# ---------------------------------------------------------------- MCC
def c2(n):
    return n * (n - 1) // 2


def mcc(tp, fp, fn, tn):
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if d == 0 else (tp * tn - fp * fn) / d


def mcc_from_decisions(families, decisions_same, n_total=33):
    """families : [[(is_same, type)]], decisions_same : [[bool]] alignes."""
    tp = fp = fn = intra = 0
    for f, dec in zip(families, decisions_same):
        a = sum(1 for (s, _), d in zip(f, dec) if s and d)
        b = sum(1 for (s, _), d in zip(f, dec) if s and not d)
        c = sum(1 for (s, _), d in zip(f, dec) if (not s) and d)
        tp += c2(a)
        fn += c2(a + b) - c2(a)
        fp += c2(a + c) - c2(a)
        intra += c2(len(f))
    tn = c2(n_total) - tp - fp - fn
    return mcc(tp, fp, fn, tn), (tp, fp, fn, tn)


def positives(families):
    return sum(c2(sum(1 for s, _ in f if s)) for f in families)


# ---------------------------------------------------------------- simulation
def simulate(builder, trials, n_resp, rng, noise_shift=0.0):
    """Renvoie {tau: [mcc par essai]} et la liste des structures tirees."""
    curves = {t: [] for t in TAU_GRID}
    for _ in range(trials):
        fams = builder(rng)
        phats = []
        for f in fams:
            row = []
            for is_same, typ in f:
                p = RATES[typ] + (noise_shift if is_same else 0.0)
                p = min(max(p, 0.0), 1.0)
                row.append(binom(n_resp, p).sample(rng) / n_resp)
            phats.append(row)
        for t in TAU_GRID:
            dec = [[ph <= t + 1e-12 for ph in row] for row in phats]
            curves[t].append(mcc_from_decisions(fams, dec)[0])
    return curves


def pct(xs, q):
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[i]


def summarize(curve):
    return sum(curve) / len(curve), pct(curve, 0.10), pct(curve, 0.50), pct(curve, 0.90)


def best_tau(curves):
    return max(TAU_GRID, key=lambda t: sum(curves[t]) / len(curves[t]))


# ---------------------------------------------------------------- rules
NOISE_MEAN = sum(RATES[t] for t in SAME_TYPES) / len(SAME_TYPES)
NOISE_MAX = max(RATES[t] for t in SAME_TYPES)


def rule_apriori(n):
    """Seuil a priori sans oracle : pire taux de bruit + 2 ecarts-types binomiaux."""
    se = math.sqrt(NOISE_MAX * (1 - NOISE_MAX) / n)
    return round(NOISE_MAX + 2 * se, 2)


def fmt(x):
    return f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("## 1. Taux de divergence mesures (jumeau OLMo-2-1B, n=300, p = 1 - identical_frac)\n")
    print("| type | p mesure | ecart-type de p_hat a N=100 | a N=300 |")
    print("|---|---|---|---|")
    for t in SAME_TYPES + ALTERED_TYPES:
        p = RATES[t]
        print(f"| {t} | {p:.3f} | {math.sqrt(p*(1-p)/N_HACK):.3f} | {math.sqrt(p*(1-p)/N_TWIN):.3f} |")
    print(f"\nBruit de fond : moyenne {NOISE_MEAN:.3f}, max {NOISE_MAX:.3f}.\n")

    tau_apriori = rule_apriori(N_HACK)
    print(f"Seuil a priori (max bruit + 2 SE a N=100) : tau = {tau_apriori}\n")

    print("## 2. Puissance de detection par type, a N=100, selon le seuil tau\n")
    taus_show = [0.32, 0.36, 0.40, tau_apriori, 0.45, 0.50]
    taus_show = sorted(set(taus_show))
    print("| type | p | " + " | ".join(f"P(isole) tau={t}" for t in taus_show) + " |")
    print("|---|---|" + "---|" * len(taus_show))
    for t in SAME_TYPES + ALTERED_TYPES:
        p = RATES[t]
        b = binom(N_HACK, p)
        cells = [fmt(b.tail_prob(int(math.floor(tau * N_HACK + 1e-9)))) for tau in taus_show]
        label = "FP si isole" if t in SAME_TYPES else "puissance"
        print(f"| {t} ({label}) | {p:.3f} | " + " | ".join(cells) + " |")
    print()

    print("## 3. MCC par structure, balayage du seuil (N=100)\n")
    results = {}
    agg = {t: [] for t in TAU_GRID}
    for name, builder in STRUCTURES.items():
        rng = random.Random(args.seed)
        curves = simulate(builder, args.trials, N_HACK, rng)
        for t in TAU_GRID:
            agg[t].extend(curves[t])
        tstar = best_tau(curves)
        results[name] = (curves, tstar)
        fams = builder(random.Random(args.seed))
        pos = positives(fams)
        print(f"### {name}")
        print(f"Paires positives (meme modele) : {pos} / 528 ; familles : {len(fams)}\n")
        print("| tau | MCC moyen | p10 | mediane | p90 |")
        print("|---|---|---|---|---|")
        for t in [0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.55, 0.60, 0.70]:
            m, p10, med, p90 = summarize(curves[t])
            star = " *" if abs(t - tstar) < 1e-9 else ""
            print(f"| {t:.2f}{star} | {fmt(m)} | {fmt(p10)} | {fmt(med)} | {fmt(p90)} |")
        if tstar not in [0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.55, 0.60, 0.70]:
            m, p10, med, p90 = summarize(curves[tstar])
            print(f"| {tstar:.2f} * | {fmt(m)} | {fmt(p10)} | {fmt(med)} | {fmt(p90)} |")
        print(f"\nSeuil optimal tau* = {tstar:.2f}\n")

    tau_global = best_tau(agg)
    print(f"## 4. Seuil optimal global (toutes structures confondues) : tau* = {tau_global:.2f}\n")

    print("## 5. Notre methode vs methode naive (MCC moyen [p10 ; p90], N=100)\n")
    rules = [
        ("naive A : seuil = moyenne du bruit (" + f"{NOISE_MEAN:.2f}" + "), sans correction", round(NOISE_MEAN, 2)),
        ("naive B : seuil = 0.50 (majorite)", 0.50),
        (f"notre methode, regle a priori (max bruit + 2 SE = {tau_apriori})", tau_apriori),
        (f"notre methode, seuil optimal global ({tau_global:.2f})", tau_global),
    ]
    print("| structure | " + " | ".join(r[0] for r in rules) + " | oracle par structure |")
    print("|---|" + "---|" * (len(rules) + 1))
    gains = []
    for name, (curves, tstar) in results.items():
        cells = []
        for _, t in rules:
            m, p10, _, p90 = summarize(curves[t])
            cells.append(f"{fmt(m)} [{fmt(p10)} ; {fmt(p90)}]")
        m, p10, _, p90 = summarize(curves[tstar])
        cells.append(f"{fmt(m)} (tau={tstar:.2f})")
        gains.append(summarize(curves[tau_apriori])[0] - summarize(curves[0.50])[0])
        print(f"| {name} | " + " | ".join(cells) + " |")
    print(f"\nGain moyen de la regle a priori sur la naive B (0.50) : {sum(gains)/len(gains):+.3f} MCC\n")

    print("## 6. Cout du budget : N=100 (hackathon) vs N=300 (jumeau), au seuil optimal de chaque N\n")
    print("| structure | MCC N=100 (tau*) | MCC N=300 (tau*) | perte due au budget |")
    print("|---|---|---|---|")
    for name, builder in STRUCTURES.items():
        c100, t100 = results[name]
        c300 = simulate(builder, max(500, args.trials // 3), N_TWIN, random.Random(args.seed + 1))
        t300 = best_tau(c300)
        m100 = summarize(c100[t100])[0]
        m300 = summarize(c300[t300])[0]
        print(f"| {name} | {fmt(m100)} (tau={t100:.2f}) | {fmt(m300)} (tau={t300:.2f}) | {m100 - m300:+.3f} |")
    print()

    print("## 7. Sensibilite : si le bruit de fond reel des endpoints est plus fort que sur le jumeau\n")
    print("(decalage ajoute au p des copies du meme modele ; structure S5 subtile et S1 ; seuil a priori recalcule sur le bruit decale et seuil oracle)\n")
    print("| decalage bruit | bruit max | MCC S1 a priori | MCC S1 oracle (tau*) | MCC S5 a priori | MCC S5 oracle (tau*) |")
    print("|---|---|---|---|---|---|")
    for shift in [0.0, 0.04, 0.08, 0.12, 0.16]:
        nm = NOISE_MAX + shift
        ta = round(nm + 2 * math.sqrt(nm * (1 - nm) / N_HACK), 2)
        cells = []
        for key in ["S1 3 familles x 11 (5 memes + 6 variantes)", "S5 3 x 11, variantes subtiles (int8/fp8/LoRA 0.3)"]:
            c = simulate(STRUCTURES[key], max(500, args.trials // 3), N_HACK, random.Random(args.seed + 2), noise_shift=shift)
            ts = best_tau(c)
            cells.append(f"{fmt(summarize(c[ta])[0])}")
            cells.append(f"{fmt(summarize(c[ts])[0])} ({ts:.2f})")
        print(f"| +{shift:.2f} | {nm:.2f} | " + " | ".join(cells) + " |")
    print()

    print("## 8. Anatomie d'une erreur (S1, un tirage, tau a priori)\n")
    rng = random.Random(args.seed + 7)
    fams = s1_three_by_eleven(rng)
    dec = []
    print("| famille | type | vrai | p | k/100 observe | decision |")
    print("|---|---|---|---|---|---|")
    for fi, f in enumerate(fams):
        row = []
        for is_same, typ in f:
            ph = binom(N_HACK, RATES[typ]).sample(rng) / N_HACK
            d = ph <= tau_apriori + 1e-12
            row.append(d)
            ok = "ok" if d == is_same else "ERREUR"
            print(f"| {fi} | {typ} | {'meme' if is_same else 'variante'} | {RATES[typ]:.3f} | {ph:.2f} | {'meme' if d else 'isole'} ({ok}) |")
        dec.append(row)
    m, (tp, fp, fn, tn) = mcc_from_decisions(fams, dec)
    print(f"\nMCC de ce tirage : {m:.3f} (TP={tp}, FP={fp}, FN={fn}, TN={tn})")


if __name__ == "__main__":
    main()
