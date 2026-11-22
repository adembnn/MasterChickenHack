#!/usr/bin/env python
"""Agrège calib_results.json (plusieurs corpus x grille de s) en courbe de calibration s -> divergence.

Pour chaque s : moyenne et écart-type entre corpus de identical_frac et de mean_first_div, plus
l'intervalle binomial (Wilson) sur identical_frac dû au nombre fini de prompts.
Ajuste identical_frac(s) = f0 * exp(-(s/s0)^k) et mean_first_div(s) = d0 * exp(-s/tau), puis
fournit un estimateur inverse : divergence observée -> s estimé, avec la bande d'incertitude
obtenue en propageant la dispersion entre corpus.
Usage : python analyze_calibration.py calib_results.json [--noise_frac 0.68]
"""
import argparse, json, math, statistics as st


def wilson(p, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def fit_stretched_exp(xs, ys, f0):
    """y = f0*exp(-(x/s0)^k) ; ajustement par moindres carrés sur une grille (pas de scipy)."""
    best = None
    for k10 in range(5, 40):
        k = k10 / 10
        for s0_100 in range(2, 200):
            s0 = s0_100 / 100
            err = sum((f0 * math.exp(-((x / s0) ** k)) - y) ** 2 for x, y in zip(xs, ys))
            if best is None or err < best[0]:
                best = (err, s0, k)
    return best[1], best[2], best[0]


def fit_exp(xs, ys):
    """y = d0*exp(-x/tau), ajusté sur log(y) pour y > 0."""
    pts = [(x, math.log(y)) for x, y in zip(xs, ys) if y and y > 0]
    if len(pts) < 2:
        return None
    mx = st.mean(p[0] for p in pts)
    my = st.mean(p[1] for p in pts)
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    slope = sxy / sxx
    d0 = math.exp(my - slope * mx)
    return d0, -1 / slope if slope < 0 else float("inf")


def invert_stretched(frac, f0, s0, k):
    if frac >= f0:
        return 0.0
    if frac <= 0:
        return float("inf")
    return s0 * (-math.log(frac / f0)) ** (1 / k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--noise_frac", type=float, default=0.68,
                    help="identical_frac du plancher de bruit (base vs base, autre batch/dtype), mesuré dans exp1")
    a = ap.parse_args()
    d = json.load(open(a.path))
    s_values = [float(s) for s in d["s_values"]]
    seeds = [str(s) for s in d["corpus_seeds"]]
    res = d["results"]
    n = next(iter(next(iter(res.values())).values()))["n"]

    rows = []
    for s in s_values:
        fr = [res[sd][str(s)]["identical_frac"] for sd in seeds if str(s) in res[sd]]
        md = [res[sd][str(s)]["mean_first_div"] for sd in seeds if str(s) in res[sd]]
        md = [x if x is not None else float("nan") for x in md]
        mdd = [res[sd][str(s)]["median_first_div"] for sd in seeds if str(s) in res[sd]]
        fm = st.mean(fr)
        fs = st.stdev(fr) if len(fr) > 1 else 0.0
        mm = st.mean(x for x in md if not math.isnan(x)) if any(not math.isnan(x) for x in md) else float("nan")
        ms = st.stdev([x for x in md if not math.isnan(x)]) if sum(not math.isnan(x) for x in md) > 1 else 0.0
        lo, hi = wilson(fm, n * len(fr))
        rows.append(dict(s=s, frac=fm, frac_sd=fs, frac_lo=lo, frac_hi=hi, frac_per=fr,
                         mfd=mm, mfd_sd=ms, mfd_per=md, med_per=mdd))

    print(f"# Courbe de calibration (n={n} prompts/corpus, {len(seeds)} corpus : {', '.join(seeds)})\n")
    print("| s | identical_frac moy. | sd corpus | IC95 Wilson (pooled) | par corpus | mean_first_div moy. | sd corpus | par corpus | médiane par corpus |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        per = " / ".join(f"{x:.3f}" for x in r["frac_per"])
        mper = " / ".join("—" if math.isnan(x) else f"{x:.1f}" for x in r["mfd_per"])
        medper = " / ".join("—" if x is None else str(x) for x in r["med_per"])
        mfd = "—" if math.isnan(r["mfd"]) else f"{r['mfd']:.1f}"
        print(f"| {r['s']:.2f} | {r['frac']:.3f} | {r['frac_sd']:.3f} | [{r['frac_lo']:.3f}, {r['frac_hi']:.3f}] | {per} | {mfd} | {r['mfd_sd']:.1f} | {mper} | {medper} |")

    xs = [r["s"] for r in rows]
    ys = [r["frac"] for r in rows]
    free = min(((fit_stretched_exp(xs, ys, f0 / 100), f0 / 100) for f0 in range(50, 96, 2)), key=lambda t: t[0][2])
    print(f"\n## Ajustement avec plancher f0 libre : f0 = {free[1]:.2f}, s0 = {free[0][0]:.3f}, k = {free[0][1]:.2f}, erreur = {free[0][2]:.4f}")
    s0, k, err = fit_stretched_exp(xs, ys, a.noise_frac)
    print(f"\n## Ajustement identical_frac(s) = {a.noise_frac:.2f} * exp(-(s/s0)^k)")
    print(f"s0 = {s0:.3f}, k = {k:.2f}, erreur quadratique = {err:.4f}")
    print("\n| s | mesuré | ajusté |\n|---|---|---|")
    for x, y in zip(xs, ys):
        print(f"| {x:.2f} | {y:.3f} | {a.noise_frac * math.exp(-((x / s0) ** k)):.3f} |")

    mx = [r["s"] for r in rows if not math.isnan(r["mfd"]) and r["mfd"] > 0]
    my = [r["mfd"] for r in rows if not math.isnan(r["mfd"]) and r["mfd"] > 0]
    fe = fit_exp(mx, my)
    if fe:
        d0, tau = fe
        print(f"\n## Ajustement mean_first_div(s) = d0 * exp(-s/tau)")
        print(f"d0 = {d0:.1f} tokens, tau = {tau:.3f}")

    print("\n## Estimateur inverse : identical_frac observée -> s estimé")
    print("(bande = min/max des s obtenus en ré-ajustant corpus par corpus)")
    per_corpus_fits = []
    for sd in seeds:
        ys_c = [res[sd][str(s)]["identical_frac"] for s in s_values]
        per_corpus_fits.append(fit_stretched_exp(s_values, ys_c, a.noise_frac))
    print("\n| identical_frac observée | s estimé (fit global) | bande inter-corpus |\n|---|---|---|")
    for frac in [0.60, 0.50, 0.40, 0.30, 0.20, 0.15, 0.10, 0.05, 0.02, 0.01]:
        s_hat = invert_stretched(frac, a.noise_frac, s0, k)
        band = [invert_stretched(frac, a.noise_frac, f[0], f[1]) for f in per_corpus_fits]
        print(f"| {frac:.2f} | {s_hat:.3f} | [{min(band):.3f}, {max(band):.3f}] |")

    print("\n## Pouvoir séparateur entre les 3 forces attendues (0.3 / 0.7 / 1.5)")
    for s in [0.3, 0.7, 1.5]:
        r = next(x for x in rows if abs(x["s"] - s) < 1e-9)
        mfd = "—" if math.isnan(r["mfd"]) else f"{r['mfd']:.1f} ± {r['mfd_sd']:.1f}"
        print(f"- s={s}: identical_frac = {r['frac']:.3f} ± {r['frac_sd']:.3f} (IC95 pooled [{r['frac_lo']:.3f}, {r['frac_hi']:.3f}]), mean_first_div = {mfd}")


if __name__ == "__main__":
    main()
