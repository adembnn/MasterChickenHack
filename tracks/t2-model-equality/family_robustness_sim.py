"""T2 : le seuil tau fixe (0.42-0.43, calibre uniquement sur le jumeau OLMo-2-1B)
generalise-t-il a une famille de modele dont le bruit de fond differe ?

Aucune mesure reelle de bruit/quantification n'existe aujourd'hui pour une famille
non-OLMo (fingerprint_bank.py et family_detection.py ne mesurent que l'identification
de famille par le TEXTE, pas le taux de divergence glouton sous quantification/LoRA).

Ce script ne invente donc pas de "vraies" valeurs pour Qwen/Mistral/Phi : il fait une
analyse de sensibilite (deja amorcee dans mcc_simulation.py §7, mais seulement pour un
decalage uniforme du bruit). Ici on modelise explicitement une HETEROGENEITE ENTRE
FAMILLES du bruit de fond (chaque famille tire son propre p_bruit), a partir du seul
fait mesure dont on dispose : le bruit de fond d'OLMo-2-1B varie DEJA de 0.30 a 0.34
rien qu'en changeant la taille de lot de serving (results_exp1_a100.json). On etend
cet intervalle connu a une plage plus large et plausible entre familles (tailles de
modele, tokenizer, implementation serveur differents), et on compare 3 strategies :

  A. tau fixe global = 0.43 (methode v3 actuelle, un seul chiffre pour toutes les familles)
  B. tau oracle par famille (borne superieure theorique : on suppose qu'on connait le
     vrai bruit de chaque famille sans aucune mesure -> juste pour cadrer l'ecart max)
  C. tau ADAPTATIF mesure par auto-consistance : pour chaque famille (candidat "meme
     modele" repere par l'etape F), on depense un petit nombre de reponses pour rejouer
     EXACTEMENT le meme prompt 2x sur le MEME endpoint repete (ou sur deux endpoints deja
     suspectes du meme cluster) et on estime le bruit reel de CETTE famille a partir de
     cette redondance, au lieu d'utiliser 0.43 partout.

Cout de C en reponses du budget (100/endpoint) : estime a la fin.

Python stdlib uniquement. Usage : python3 family_robustness_sim.py
"""
import bisect
import math
import random

N_HACK = 100
TRIALS = 4000

# Taux mesures sur le jumeau OLMo-2-1B (results_exp1_a100.json, n=300) -- seule donnee reelle.
OLMO_SAME = [0.30, 0.34, 0.3167, 0.3333]  # bs1, bs8, bs32, dtype fp16 (p = 1 - identical_frac)
OLMO_ALTERED = {
    "int8": 0.497, "fp8_e4m3": 0.777, "fp8_e5m2": 0.933, "nf4": 0.983,
    "lora_0.3": 0.820, "lora_0.7": 0.983, "lora_1.0": 1.0, "lora_1.5": 1.0,
}
OLMO_NOISE_MEAN = sum(OLMO_SAME) / len(OLMO_SAME)
OLMO_NOISE_MAX = max(OLMO_SAME)
TAU_FIXED = 0.43  # methode v3, PR #12/#14


class Binomial:
    def __init__(self, n, p):
        self.n = n
        pmf = [math.comb(n, k) * p**k * (1 - p) ** (n - k) if 0 < p < 1
               else (1.0 if (k == 0 and p <= 0) or (k == n and p >= 1) else 0.0)
               for k in range(n + 1)]
        acc, cdf = 0.0, []
        for v in pmf:
            acc += v
            cdf.append(acc)
        cdf[-1] = 1.0
        self.cdf = cdf

    def sample(self, rng):
        return bisect.bisect_left(self.cdf, rng.random())


_CACHE = {}


def binom(n, p):
    key = (n, round(p, 6))
    if key not in _CACHE:
        _CACHE[key] = Binomial(n, p)
    return _CACHE[key]


def c2(n):
    return n * (n - 1) // 2


def mcc(tp, fp, fn, tn):
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return 0.0 if d == 0 else (tp * tn - fp * fn) / d


def mcc_from_decisions(families, decisions):
    tp = fp = fn = 0
    n_total = sum(len(f) for f in families)
    for f, dec in zip(families, decisions):
        a = sum(1 for (s, _), d in zip(f, dec) if s and d)
        b = sum(1 for (s, _), d in zip(f, dec) if s and not d)
        c = sum(1 for (s, _), d in zip(f, dec) if (not s) and d)
        tp += c2(a)
        fn += c2(a + b) - c2(a)
        fp += c2(a + c) - c2(a)
    tn = c2(n_total) - tp - fp - fn
    return mcc(tp, fp, fn, tn)


def make_families(rng, n_fam, fam_size, altered_per_fam, noise_spread):
    """n_fam familles de fam_size endpoints, chacune avec SON propre bruit de fond
    p_bruit_famille tire autour de OLMO_NOISE_MEAN +/- noise_spread (uniforme),
    et altered_per_fam variantes alterees (types OLMo, appliques en relatif au
    bruit de la famille : meme ratio divergence_alteree / divergence_bruit qu'OLMo)."""
    families = []
    true_noise = []
    for _ in range(n_fam):
        p_bruit = rng.uniform(max(0.02, OLMO_NOISE_MEAN - noise_spread),
                               min(0.60, OLMO_NOISE_MEAN + noise_spread))
        true_noise.append(p_bruit)
        n_same = fam_size - altered_per_fam
        eps = [(True, p_bruit) for _ in range(n_same)]
        alt_types = rng.sample(list(OLMO_ALTERED), altered_per_fam) if altered_per_fam <= len(OLMO_ALTERED) \
            else [rng.choice(list(OLMO_ALTERED)) for _ in range(altered_per_fam)]
        for t in alt_types:
            # meme facteur multiplicatif qu'OLMo entre bruit et variante,
            # applique au bruit REEL (pas OLMo) de cette famille
            ratio = OLMO_ALTERED[t] / OLMO_NOISE_MEAN
            p_alt = min(0.999, p_bruit * ratio)
            eps.append((False, p_alt))
        rng.shuffle(eps)
        families.append(eps)
    return families, true_noise


def decide_fixed(families, tau, rng):
    return [[binom(N_HACK, p).sample(rng) / N_HACK <= tau for (_, p) in f] for f in families]


def decide_oracle_per_family(families, true_noise, rng):
    decs = []
    for f, p_bruit in zip(families, true_noise):
        se = math.sqrt(p_bruit * (1 - p_bruit) / N_HACK)
        tau_f = p_bruit + 2 * se
        decs.append([binom(N_HACK, p).sample(rng) / N_HACK <= tau_f for (_, p) in f])
    return decs


def decide_adaptive(families, true_noise, rng, n_repeat_budget):
    """Mesure le bruit de CHAQUE famille en rejouant n_repeat_budget prompts
    identiques sur le meme endpoint (ou deux endpoints deja suspectes 'meme modele'),
    PRIS SUR LE BUDGET DE 100 (donc N_HACK - n_repeat_budget reponses restent pour la
    comparaison glouton-vs-base). tau_famille = p_hat_bruit_mesure + 2*SE(n_repeat)."""
    decs = []
    n_decide = N_HACK - n_repeat_budget
    for f, p_bruit in zip(families, true_noise):
        k_noise = binom(n_repeat_budget, p_bruit).sample(rng)
        p_hat_bruit = k_noise / n_repeat_budget
        se = math.sqrt(max(p_hat_bruit, 1 / n_repeat_budget) * (1 - p_hat_bruit) / n_repeat_budget)
        tau_f = min(0.60, p_hat_bruit + 2 * se)
        row = []
        for (is_same, p) in f:
            k = binom(n_decide, p).sample(rng)
            row.append(k / n_decide <= tau_f)
        decs.append(row)
    return decs


def run(noise_spread, n_fam=6, fam_size_range=(4, 7), altered_frac=0.45, n_repeat_budget=20, trials=TRIALS, seed=0):
    rng = random.Random(seed)
    results = {"fixe": [], "oracle_famille": [], "adaptatif": []}
    for _ in range(trials):
        fam_size = rng.randint(*fam_size_range)
        altered = max(1, round(fam_size * altered_frac))
        families, true_noise = make_families(rng, n_fam, fam_size, altered, noise_spread)
        results["fixe"].append(mcc_from_decisions(families, decide_fixed(families, TAU_FIXED, rng)))
        results["oracle_famille"].append(mcc_from_decisions(families, decide_oracle_per_family(families, true_noise, rng)))
        results["adaptatif"].append(mcc_from_decisions(families, decide_adaptive(families, true_noise, rng, n_repeat_budget)))
    return {k: sum(v) / len(v) for k, v in results.items()}


def pct(xs, q):
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[i]


def run_full(noise_spread, n_fam=6, fam_size_range=(4, 7), altered_frac=0.45, n_repeat_budget=20, trials=TRIALS, seed=0):
    rng = random.Random(seed)
    curves = {"fixe": [], "oracle_famille": [], "adaptatif": []}
    for _ in range(trials):
        fam_size = rng.randint(*fam_size_range)
        altered = max(1, round(fam_size * altered_frac))
        families, true_noise = make_families(rng, n_fam, fam_size, altered, noise_spread)
        curves["fixe"].append(mcc_from_decisions(families, decide_fixed(families, TAU_FIXED, rng)))
        curves["oracle_famille"].append(mcc_from_decisions(families, decide_oracle_per_family(families, true_noise, rng)))
        curves["adaptatif"].append(mcc_from_decisions(families, decide_adaptive(families, true_noise, rng, n_repeat_budget)))
    out = {}
    for k, v in curves.items():
        out[k] = (sum(v) / len(v), pct(v, 0.10), pct(v, 0.50), pct(v, 0.90))
    return out


def main():
    print("## Bruit de fond OLMo-2-1B mesure (variation deja due a la seule taille de lot)")
    print(f"same-model p (bs1/bs8/bs32/fp16) = {OLMO_SAME} -> moyenne {OLMO_NOISE_MEAN:.3f}, max {OLMO_NOISE_MAX:.3f}\n")

    print("## MCC moyen (4000 tirages) selon l'ecart-type suppose du bruit ENTRE familles (demi-largeur)")
    print("(0.00 = toutes les familles ont exactement le bruit d'OLMo ; >0 = heterogeneite inter-famille,")
    print(" hypothese de travail puisqu'aucune mesure reelle n'existe pour les autres familles)\n")
    print("| demi-largeur bruit inter-famille | tau fixe 0.43 (methode v3) | tau oracle/famille (borne sup.) | tau adaptatif (20 reponses d'auto-consistance / famille) |")
    print("|---|---|---|---|")
    for spread in [0.00, 0.03, 0.06, 0.10, 0.15, 0.20]:
        r = run(spread)
        print(f"| ±{spread:.2f} | {r['fixe']:.3f} | {r['oracle_famille']:.3f} | {r['adaptatif']:.3f} |")
    print()

    print("## Detail (moyenne [p10 ; p90]) a demi-largeur = 0.15 (scenario plausible : hetero. de taille/implementation)\n")
    detail = run_full(0.15)
    for k, (m, p10, med, p90) in detail.items():
        print(f"- {k} : {m:.3f} [{p10:.3f} ; {p90:.3f}] (mediane {med:.3f})")
    print()

    print("## Cout en reponses du budget de l'auto-consistance adaptative")
    for nrep in [10, 20, 30]:
        r = run(0.15, n_repeat_budget=nrep)
        print(f"n_repeat={nrep} (reste {N_HACK-nrep} reponses pour la decision) -> MCC adaptatif = {r['adaptatif']:.3f} (fixe={r['fixe']:.3f})")


if __name__ == "__main__":
    main()
