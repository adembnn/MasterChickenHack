#!/usr/bin/env python
"""Partitionnement (famille, modèle) à partir d'une matrice de distance/similarité bruitée
entre points de terminaison (endpoints), pour maximiser le MCC attendu sur les 528 paires
avec la pondération 0.1*MCC_famille + 0.9*MCC_modèle (reports/sources/Model_Equality_Testing.txt).

Usage en production (vraies données) :
    from clustering import partition_from_distance_matrix
    result = partition_from_distance_matrix(dist, endpoint_ids, algo="correlation")
    # result.family[ep], result.model[ep] -> colonnes du CSV de soumission

`dist` est une matrice de distance symétrique à diagonale nulle entre endpoints : n'importe
quelle statistique de dissemblance par paire convient (MMD, 1 - P(même) calibrée, divergence
de génération gloutonne comme dans tracks/t2-model-equality/twin/exp1_divergence.py, etc.),
pourvu que "plus grand" veuille dire "plus différent".

Usage en simulation (ce fichier exécuté seul) : génère des matrices de distance synthétiques
bruitées pour plusieurs scénarios de structure (nombre de familles, un gros cluster "base" +
quelques singletons variants) et compare trois algorithmes :
    - seuil optimisé sur un dendrogramme à lien moyen ("hierarchical")
    - clustering de corrélation par recherche locale ("correlation")
    - MAP bayésien avec prior sur la taille des clusters ("bayes_size_prior")

    python clustering.py                        # toutes les simulations, résumé JSON sur stdout
    python clustering.py --scenario moyen_3fam   # un seul scénario
    python clustering.py --generator divergence  # bruit binomial calibré sur le jumeau local
    python clustering.py --out results.json      # écrit aussi le détail dans un fichier
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import statistics
import sys
import zlib
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 1. Score : MCC et score composite 0.1*famille + 0.9*modèle
# ---------------------------------------------------------------------------

def mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    """Matthews correlation coefficient. 0.0 si le dénominateur est nul (convention du sujet)."""
    num = tp * tn - fp * fn
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if den <= 0:
        return 0.0
    return num / math.sqrt(den)


def _pairs(n: int):
    return itertools.combinations(range(n), 2)


def mcc_from_labels(true_labels: list, pred_labels: list, restrict_to: set | None = None) -> float:
    """MCC de la question "même groupe ?" sur toutes les paires i<j, ou seulement celles de
    `restrict_to` (ensemble de paires (i, j), i<j) si fourni — c'est ainsi que MCC_modèle se
    restreint aux paires de vraie même famille (reports/sources/Model_Equality_Testing.txt)."""
    tp = tn = fp = fn = 0
    n = len(true_labels)
    for i, j in _pairs(n):
        if restrict_to is not None and (i, j) not in restrict_to:
            continue
        true_same = true_labels[i] == true_labels[j]
        pred_same = pred_labels[i] == pred_labels[j]
        if true_same and pred_same:
            tp += 1
        elif (not true_same) and (not pred_same):
            tn += 1
        elif pred_same and not true_same:
            fp += 1
        else:
            fn += 1
    return mcc(tp, tn, fp, fn)


def composite_score(true_family, true_model, pred_family, pred_model) -> dict:
    """Score officiel : 0.1*MCC_famille (toutes les paires) + 0.9*MCC_modèle (paires de vraie
    même famille uniquement). Retourne le détail pour diagnostic."""
    n = len(true_family)
    same_family_pairs = {(i, j) for i, j in _pairs(n) if true_family[i] == true_family[j]}
    mcc_family = mcc_from_labels(true_family, pred_family)
    mcc_model = mcc_from_labels(true_model, pred_model, restrict_to=same_family_pairs)
    total = 0.1 * mcc_family + 0.9 * mcc_model
    return {"mcc_family": mcc_family, "mcc_model": mcc_model, "score": total}


# ---------------------------------------------------------------------------
# 2. Simulateur de structure vraie et de matrice de distance bruitée
# ---------------------------------------------------------------------------

@dataclass
class Structure:
    n: int
    family: list          # family[i] = id de famille (int)
    model: list            # model[i] = id de modèle vrai (int), unique entre familles
    variant_kind: list     # variant_kind[i] = type de variante ("base", "nf4", "lora_0.3", ...)


# Magnitudes relatives de signal par type de variante, calibrées sur les ordres de grandeur
# mesurés/estimés dans reports/titouan/t2-statistics.md (section 2.5 et 2.7) : int8/fp8/LoRA
# faible sont à peine au-dessus du bruit de nuisance, nf4/LoRA forte et "autre famille" sont
# nettement séparés. Valeurs en unités arbitraires de distance (le bruit de mesure est à l'échelle 1).
VARIANT_SIGNAL = {
    "base": 0.0,
    "int8": 0.15,
    "fp8": 0.2,
    "lora_0.3": 0.3,
    "nf4": 1.4,
    "lora_0.7": 1.8,
    "lora_1.5": 4.0,
    "other_quant": 1.0,
}
FAMILY_SIGNAL = 12.0  # écart entre familles : très supérieur à tout signal intra-famille (2.5)


def make_structure(n_families: int, family_sizes: list[int], base_fraction: float, rnd: random.Random) -> Structure:
    """Construit une structure "vraie" : par famille, une fraction `base_fraction` des endpoints
    partage le même modèle "base" (gros cluster, ex. plusieurs déguisements du même modèle), le
    reste est réparti en quelques singletons de variantes différentes (fp8, nf4, LoRA à 2-3
    échelles...), conformément au scénario de reports/titouan/t2-statistics.md section 6.1."""
    assert len(family_sizes) == n_families
    family, model, kind = [], [], []
    next_model_id = 0
    variant_pool = ["int8", "fp8", "nf4", "lora_0.3", "lora_0.7", "lora_1.5", "other_quant"]
    for fam_id, size in enumerate(family_sizes):
        n_base = max(1, round(size * base_fraction))
        n_variants = size - n_base
        base_model_id = next_model_id
        next_model_id += 1
        family += [fam_id] * n_base
        model += [base_model_id] * n_base
        kind += ["base"] * n_base
        variants = rnd.sample(variant_pool, k=min(n_variants, len(variant_pool)))
        while len(variants) < n_variants:
            variants.append(rnd.choice(variant_pool))
        for v in variants[:n_variants]:
            family.append(fam_id)
            model.append(next_model_id)
            next_model_id += 1
            kind.append(v)
    n = len(family)
    order = list(range(n))
    rnd.shuffle(order)
    family = [family[i] for i in order]
    model = [model[i] for i in order]
    kind = [kind[i] for i in order]
    return Structure(n=n, family=family, model=model, variant_kind=kind)


def generate_distance_matrix(
    struct: Structure,
    noise_scale: float,
    nuisance_scale: float,
    nuisance_fraction: float,
    rnd: random.Random,
) -> list[list[float]]:
    """Matrice de distance bruitée entre `struct.n` endpoints.

    Modèle additif, inspiré de reports/titouan/t2-statistics.md (section 2.5) :
      d_ij = bruit de mesure (toujours présent, hétéroscédastique)
           + signal de famille (si familles différentes, grand et quasi certain)
           + signal de modèle (si même famille, modèles différents : dépend du type de variante)
           + nuisance partagée (confondeurs comme prompt système/persona/température : affecte
             une fraction aléatoire des paires, avec une magnitude qui peut dépasser le signal
             des variantes discrètes (int8/fp8/LoRA faible), conformément à 2.5 : "les nuisances
             déplacent la distribution 10 à 300 fois plus que fp8/int8/LoRA 0.3").
    Toutes les distances sont tronquées à >= 0.
    """
    n = struct.n
    d = [[0.0] * n for _ in range(n)]
    # bruit par endpoint (variabilité propre à chaque point de mesure : latence, longueur
    # moyenne des réponses...) en plus du bruit par paire, pour une hétéroscédasticité réaliste
    endpoint_jitter = [rnd.gauss(0, noise_scale * 0.3) for _ in range(n)]
    # quelques endpoints "contaminés" par une nuisance partagée (ex. prompt système injecté)
    n_contaminated = round(n * nuisance_fraction)
    contaminated = set(rnd.sample(range(n), k=min(n_contaminated, n)))
    for i, j in _pairs(n):
        base_noise = abs(rnd.gauss(0, noise_scale)) + abs(endpoint_jitter[i]) + abs(endpoint_jitter[j])
        if struct.family[i] != struct.family[j]:
            signal = FAMILY_SIGNAL * (1.0 + rnd.gauss(0, 0.05))
        elif struct.model[i] != struct.model[j]:
            k_i, k_j = struct.variant_kind[i], struct.variant_kind[j]
            s_i = VARIANT_SIGNAL.get(k_i, 1.0)
            s_j = VARIANT_SIGNAL.get(k_j, 1.0)
            # deux variantes différentes de la même base : les signaux individuels par rapport
            # à "base" s'ajoutent à peu près (ils divergent chacun dans leur propre direction)
            signal = s_i + s_j if k_i != k_j or (k_i != "base") else max(s_i, s_j)
            signal = max(signal, 0.05)
        else:
            signal = 0.0
        nuisance = 0.0
        if (i in contaminated) != (j in contaminated):
            # paire où un seul des deux porte la nuisance : elle gonfle artificiellement la
            # distance mesurée, qu'ils soient du même modèle ou non (2.5 : mécanisme des FP)
            nuisance = abs(rnd.gauss(nuisance_scale, nuisance_scale * 0.4))
        val = max(0.0, base_noise + signal + nuisance)
        d[i][j] = d[j][i] = val
    return d


# Taux de divergence de génération gloutonne par type de variante, mesurés sur le jumeau local
# OLMo-2-1B (tracks/t2-model-equality/twin/exp1_divergence.py, repris dans
# reports/titouan/t2-mcc-simulation.md) : part des prompts dont le texte glouton diffère de la base.
DIVERGENCE_RATE = {
    "base": 0.32,          # plancher de bruit de serving entre deux copies du MÊME modèle (0,30-0,34)
    "int8": 0.50,
    "fp8": 0.85,           # 0,78-0,93
    "lora_0.3": 0.82,
    "nf4": 0.98,
    "lora_0.7": 0.98,
    "lora_1.5": 0.99,
    "other_quant": 0.90,
}


def generate_divergence_matrix(
    struct: Structure,
    n_responses: int,
    nuisance_scale: float,
    nuisance_fraction: float,
    rnd: random.Random,
) -> list[list[float]]:
    """Matrice de distance "réaliste" : d_ij = taux de divergence gloutonne observé sur
    `n_responses` prompts (bruit binomial), avec les taux mesurés sur le jumeau local
    (DIVERGENCE_RATE). Deux variantes différentes de la même base divergent entre elles au moins
    autant que la plus divergente des deux par rapport à la base. Les endpoints "contaminés" par
    un déguisement non neutralisé (prompt système, température) voient leur divergence avec tout
    le monde augmentée de `nuisance_scale` (ex. 0,1 à 0,3) : c'est le risque identifié dans
    reports/titouan/t2-mcc-simulation.md ("si la divergence réelle entre deux copies dépasse 0,42,
    int8 est perdu"). Les autres familles divergent à ~100 %."""
    n = struct.n
    d = [[0.0] * n for _ in range(n)]
    n_contaminated = round(n * nuisance_fraction)
    contaminated = set(rnd.sample(range(n), k=min(n_contaminated, n)))
    for i, j in _pairs(n):
        if struct.family[i] != struct.family[j]:
            p = 0.995
        else:
            k_i, k_j = struct.variant_kind[i], struct.variant_kind[j]
            if struct.model[i] == struct.model[j]:
                p = DIVERGENCE_RATE["base"]
            else:
                p = max(DIVERGENCE_RATE.get(k_i, 0.9), DIVERGENCE_RATE.get(k_j, 0.9))
            if (i in contaminated) != (j in contaminated):
                p = p + nuisance_scale * (1 - p) * rnd.uniform(0.5, 1.5)
        p = min(max(p, 0.0), 1.0)
        successes = sum(1 for _ in range(n_responses) if rnd.random() < p)
        d[i][j] = d[j][i] = successes / n_responses
    return d


GENERATORS = {
    "additive": "bruit gaussien additif + signaux par variante (VARIANT_SIGNAL), nuisances partagées",
    "divergence": "taux de divergence gloutonne binomial à N réponses (DIVERGENCE_RATE du jumeau)",
}


def make_distance(struct: Structure, generator: str, noise: float, nuisance: float, nuisance_fraction: float, rnd: random.Random):
    """`noise` = écart-type du bruit (additive) ou nombre de réponses N (divergence)."""
    if generator == "additive":
        return generate_distance_matrix(struct, noise_scale=noise, nuisance_scale=nuisance,
                                        nuisance_fraction=nuisance_fraction, rnd=rnd)
    if generator == "divergence":
        return generate_divergence_matrix(struct, n_responses=int(noise), nuisance_scale=nuisance,
                                          nuisance_fraction=nuisance_fraction, rnd=rnd)
    raise ValueError(f"générateur inconnu: {generator}")


# ---------------------------------------------------------------------------
# 3. Algorithme A : clustering hiérarchique à lien moyen, seuil optimisé
# ---------------------------------------------------------------------------

def _average_linkage_merges(dist: list[list[float]]) -> list[tuple]:
    """Dendrogramme par lien moyen (agglomératif, O(n^3), suffisant pour n <= quelques centaines).
    Retourne la liste des fusions (height, members_a, members_b) dans l'ordre croissant de height."""
    n = len(dist)
    clusters = {i: [i] for i in range(n)}
    active = set(range(n))
    merges = []

    def avg_dist(a, b):
        members_a, members_b = clusters[a], clusters[b]
        total = sum(dist[x][y] for x in members_a for y in members_b)
        return total / (len(members_a) * len(members_b))

    next_id = n
    while len(active) > 1:
        best = None
        for a, b in itertools.combinations(sorted(active), 2):
            d = avg_dist(a, b)
            if best is None or d < best[0]:
                best = (d, a, b)
        height, a, b = best
        merges.append((height, list(clusters[a]), list(clusters[b])))
        clusters[next_id] = clusters[a] + clusters[b]
        active.discard(a)
        active.discard(b)
        active.add(next_id)
        next_id += 1
    return merges


def hierarchical_partition_at_threshold(dist: list[list[float]], threshold: float) -> list[int]:
    """Coupe le dendrogramme à lien moyen à `threshold` : deux points sont dans le même cluster
    final si leur plus petite fusion commune a lieu à une hauteur <= threshold. Implémenté
    directement par union-find sur les fusions, sans reconstruire l'arbre complet."""
    n = len(dist)
    merges = _average_linkage_merges(dist)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for height, members_a, members_b in merges:
        if height <= threshold:
            union(members_a[0], members_b[0])
    roots = [find(i) for i in range(n)]
    remap = {}
    labels = []
    for r in roots:
        if r not in remap:
            remap[r] = len(remap)
        labels.append(remap[r])
    return labels


def hierarchical_oracle_threshold(dist: list[list[float]], true_labels: list[int], restrict_to: set | None = None) -> tuple[float, list[int]]:
    """Meilleur seuil POSSIBLE (triche en regardant les vraies étiquettes) : borne supérieure de
    ce que le lien moyen peut atteindre, utile en simulation pour juger la perte due au choix du
    seuil séparément de la perte due à la méthode elle-même."""
    merges = _average_linkage_merges(dist)
    candidate_heights = sorted({0.0} | {h for h, _, _ in merges} | {h + 1e-9 for h, _, _ in merges})
    best_score, best_t, best_labels = -1.0, 0.0, None
    for t in candidate_heights:
        labels = hierarchical_partition_at_threshold(dist, t)
        score = mcc_from_labels(true_labels, labels, restrict_to=restrict_to)
        if score > best_score:
            best_score, best_t, best_labels = score, t, labels
    return best_t, best_labels


@dataclass
class Mixture:
    """Mélange de deux gaussiennes 1D sur les distances : composante 0 = paires "même modèle"
    (petites distances), composante 1 = paires "modèles différents"."""
    pi0: float
    mu0: float
    sd0: float
    mu1: float
    sd1: float
    log_space: bool = True   # paramètres exprimés sur log(d + eps)
    eps: float = 1e-6

    def _x(self, d: float) -> float:
        return math.log(max(d, 0.0) + self.eps) if self.log_space else d

    def _inv(self, x: float) -> float:
        return max(0.0, math.exp(x) - self.eps) if self.log_space else x

    def log_odds_same(self, d: float, clip: float = 8.0) -> float:
        """log P(même|d) - log P(différent|d), borné pour éviter que quelques paires extrêmes ne
        dominent l'objectif du clustering de corrélation."""
        x = self._x(d)
        g0 = self.pi0 * _gauss_pdf(x, self.mu0, self.sd0)
        g1 = (1 - self.pi0) * _gauss_pdf(x, self.mu1, self.sd1)
        if g0 <= 0 and g1 <= 0:
            return 0.0
        if g0 <= 0:
            return -clip
        if g1 <= 0:
            return clip
        return max(-clip, min(clip, math.log(g0) - math.log(g1)))

    def threshold(self) -> float:
        """Point de croisement des deux densités pondérées (décision "même" en dessous), rendu
        dans l'unité d'origine des distances."""
        lo, hi = self.mu0, max(self.mu1, self.mu0 + 1e-6)
        best_t, best_gap = lo, float("inf")
        steps = 400
        for k in range(steps + 1):
            t = lo + (hi - lo) * k / steps
            gap = abs(self.pi0 * _gauss_pdf(t, self.mu0, self.sd0) - (1 - self.pi0) * _gauss_pdf(t, self.mu1, self.sd1))
            if gap < best_gap:
                best_gap, best_t = gap, t
        return self._inv(best_t)


def fit_mixture(values: list[float], pi0_init: float = 0.3, n_iter: int = 100, log_space: bool = True) -> Mixture:
    """EM à deux composantes gaussiennes sur une liste de distances. Sans étiquettes : utilisable
    en production (cf. reports/titouan/t2-statistics.md section 6.2, "version non supervisée").
    `pi0_init` ~ prévalence attendue des paires "même" (0,25 à 0,3 intra-famille sur ref).

    Par défaut l'ajustement se fait sur log(d + eps) : les distances "différent" ont une queue
    droite lourde (variantes faibles à ~1-2, fortes à ~4-6, autres familles à ~12) et en échelle
    linéaire la composante "différent" se cale sur la queue lointaine, laissant la composante
    "même" absorber les variantes moyennes (seuil beaucoup trop haut, observé en simulation :
    0,51 contre 0,94 pour le seuil oracle à bruit quasi nul). Le log comprime cette queue."""
    eps = 1e-6
    if log_space:
        pos = [v for v in values if v > 0]
        eps = max(1e-6, 1e-3 * statistics.median(pos)) if pos else 1e-6
        xs = [math.log(max(v, 0.0) + eps) for v in values]
    else:
        xs = list(values)
    if len(xs) < 4:
        m = statistics.median(xs) if xs else 0.0
        return Mixture(0.5, m, 1e-3, m, 1e-3, log_space, eps)
    mu0, mu1 = min(xs), max(xs)
    if mu0 == mu1:
        return Mixture(0.5, mu0, 1e-3, mu1, 1e-3, log_space, eps)
    values = xs
    sd0 = sd1 = max((mu1 - mu0) / 4, 1e-6)
    pi0 = pi0_init
    for _ in range(n_iter):
        w0, w1 = [], []
        for v in values:
            g0 = pi0 * _gauss_pdf(v, mu0, sd0)
            g1 = (1 - pi0) * _gauss_pdf(v, mu1, sd1)
            s = g0 + g1
            if s <= 0:
                w0.append(0.5)
                w1.append(0.5)
            else:
                w0.append(g0 / s)
                w1.append(g1 / s)
        sw0, sw1 = sum(w0), sum(w1)
        if sw0 < 1e-9 or sw1 < 1e-9:
            break
        new_mu0 = sum(w * v for w, v in zip(w0, values)) / sw0
        new_mu1 = sum(w * v for w, v in zip(w1, values)) / sw1
        new_sd0 = math.sqrt(sum(w * (v - new_mu0) ** 2 for w, v in zip(w0, values)) / sw0 + 1e-9)
        new_sd1 = math.sqrt(sum(w * (v - new_mu1) ** 2 for w, v in zip(w1, values)) / sw1 + 1e-9)
        pi0 = sw0 / len(values)
        if new_mu0 <= new_mu1:
            mu0, mu1, sd0, sd1 = new_mu0, new_mu1, new_sd0, new_sd1
        else:
            mu0, mu1, sd0, sd1, pi0 = new_mu1, new_mu0, new_sd1, new_sd0, 1 - pi0
    return Mixture(pi0, mu0, max(sd0, 1e-6), mu1, max(sd1, 1e-6), log_space, eps)


def hierarchical_auto_threshold(dist: list[list[float]]) -> float:
    """Seuil choisi SANS regarder les étiquettes (utilisable en production) : ajuste un mélange
    de deux gaussiennes 1D sur les distances hors-diagonale (mode "même" à faible distance, mode
    "différent" à grande distance — cf. reports/titouan/t2-statistics.md section 6.2, posterior
    P(même|T)) par EM, puis renvoie le point où les deux densités pondérées se croisent."""
    n = len(dist)
    values = [dist[i][j] for i, j in _pairs(n)]
    if not values:
        return 0.0
    return fit_mixture(values).threshold()


def _gauss_pdf(x, mu, sd):
    sd = max(sd, 1e-9)
    return math.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))


# ---------------------------------------------------------------------------
# 4. Algorithme B : clustering de corrélation par recherche locale
# ---------------------------------------------------------------------------

def calibrate_weights(dist: list[list[float]], mixture=None) -> list[list[float]]:
    """Transforme une matrice de distance en poids signés w_ij = log P(même|d) - log P(diff|d)
    (positif = évidence "même", négatif = évidence "différent"), nécessaires au clustering de
    corrélation (reports/titouan/t2-statistics.md 6.3 point 2). Le mélange est ajusté sur la
    matrice elle-même si non fourni ; en production, le fournir ajusté sur TOUTES les distances
    intra-famille poolées (plus de paires, même échelle) via `partition_from_distance_matrix`."""
    n = len(dist)
    if mixture is None:
        mixture = fit_mixture([dist[i][j] for i, j in _pairs(n)])
    w = [[0.0] * n for _ in range(n)]
    for i, j in _pairs(n):
        w[i][j] = w[j][i] = mixture.log_odds_same(dist[i][j])
    return w


def correlation_clustering(weights: list[list[float]], n_restarts: int, rnd: random.Random, size_bonus=None) -> list[int]:
    """Clustering de corrélation : maximise sum_{i<j, même cluster} w_ij (+ un bonus de taille
    optionnel `size_bonus(sizes_before_after)` pour encoder un prior, cf. section 5). Résolu par
    recherche locale multi-démarrage (KwikCluster pour l'initialisation + déplacements/fusions/
    scissions gloutonnes jusqu'à convergence), comme recommandé dans
    reports/titouan/t2-statistics.md section 6.3 point 2."""
    n = len(weights)
    best_labels, best_obj = None, -float("inf")
    for _ in range(n_restarts):
        labels = _kwik_cluster_init(weights, rnd)
        labels = _local_search_refine(weights, labels, size_bonus, rnd)
        obj = _correlation_objective(weights, labels, size_bonus)
        if obj > best_obj:
            best_obj, best_labels = obj, labels
    return best_labels


def _kwik_cluster_init(weights: list[list[float]], rnd: random.Random) -> list[int]:
    n = len(weights)
    remaining = list(range(n))
    rnd.shuffle(remaining)
    labels = [-1] * n
    cluster_id = 0
    remaining_set = set(remaining)
    for pivot in remaining:
        if labels[pivot] != -1:
            continue
        labels[pivot] = cluster_id
        for j in list(remaining_set):
            if labels[j] == -1 and j != pivot and weights[pivot][j] > 0:
                labels[j] = cluster_id
        remaining_set.discard(pivot)
        cluster_id += 1
    return labels


def _correlation_objective(weights, labels, size_bonus) -> float:
    n = len(labels)
    total = sum(weights[i][j] for i, j in _pairs(n) if labels[i] == labels[j])
    if size_bonus is not None:
        sizes = {}
        for l in labels:
            sizes[l] = sizes.get(l, 0) + 1
        total += sum(size_bonus(s) for s in sizes.values())
    return total


def _local_search_refine(weights, labels, size_bonus, rnd: random.Random, max_sweeps=25) -> list[int]:
    n = len(labels)
    labels = list(labels)
    next_cluster_id = max(labels) + 1
    for _ in range(max_sweeps):
        improved = False
        order = list(range(n))
        rnd.shuffle(order)
        for i in order:
            current = labels[i]
            candidates = set(labels) | {next_cluster_id}
            best_c, best_delta = current, 0.0
            for c in candidates:
                if c == current:
                    continue
                delta = _move_delta(weights, labels, i, current, c, size_bonus)
                if delta > best_delta + 1e-9:
                    best_delta, best_c = delta, c
            if best_c != current:
                labels[i] = best_c
                if best_c == next_cluster_id:
                    next_cluster_id += 1
                improved = True
        if not improved:
            break
    return _compact_labels(labels)


def _move_delta(weights, labels, i, old_c, new_c, size_bonus) -> float:
    n = len(labels)
    old_members = [j for j in range(n) if j != i and labels[j] == old_c]
    new_members = [j for j in range(n) if j != i and labels[j] == new_c]
    delta = sum(weights[i][j] for j in new_members) - sum(weights[i][j] for j in old_members)
    if size_bonus is not None:
        old_size = len(old_members) + 1
        new_size = len(new_members) + 1
        delta += size_bonus(old_size - 1) - size_bonus(old_size)
        delta += size_bonus(new_size + 1) - size_bonus(new_size)
    return delta


def _compact_labels(labels: list[int]) -> list[int]:
    remap = {}
    out = []
    for l in labels:
        if l not in remap:
            remap[l] = len(remap)
        out.append(remap[l])
    return out


# ---------------------------------------------------------------------------
# 5. Algorithme C : MAP bayésien avec prior sur la taille des clusters
# ---------------------------------------------------------------------------

def make_size_prior(n: int, gamma: float):
    """Prior en forme de U sur la taille des clusters : favorise les singletons ET un (ou
    quelques) gros cluster "base", défavorise les tailles intermédiaires. g(k) = -gamma*(k-1)*(n-k)
    vaut 0 aux deux extrêmes (k=1 et k=n) et est maximalement négatif vers k=n/2 : encode
    l'hypothèse du scénario "gros cluster base + quelques singletons variants"
    (reports/titouan/t2-statistics.md section 6.1, cahier de la tâche)."""
    def g(k: int) -> float:
        return -gamma * (k - 1) * max(n - k, 0)
    return g


def bayes_size_prior_clustering(weights: list[list[float]], n: int, n_restarts: int, rnd: random.Random, gamma: float = 0.15) -> list[int]:
    """MAP = argmax [ log-vraisemblance des paires (= objectif du clustering de corrélation avec
    poids log-odds) + log prior de taille ]. `gamma` règle la force du prior en unités de
    log-odds par paire "manquée" ; 0 redonne exactement le clustering de corrélation."""
    prior = make_size_prior(n, gamma)
    return correlation_clustering(weights, n_restarts, rnd, size_bonus=prior)


# ---------------------------------------------------------------------------
# 6. Pipeline complet : famille puis modèle, interface de production
# ---------------------------------------------------------------------------

@dataclass
class PartitionResult:
    family: dict      # endpoint_id -> label de famille
    model: dict        # endpoint_id -> label de modèle ("fam{f}_m{k}")
    family_labels_raw: list
    model_labels_raw: list


@dataclass
class GapCalibration:
    """Calibration sans modèle : seuil = milieu (géométrique) du plus grand trou entre distances
    triées (en log, entre les quantiles 10 % et 90 %) ; log-odds linéaires autour du seuil,
    normalisés par l'écart-type des distances. Même interface que `Mixture`."""
    t: float
    scale: float

    def threshold(self) -> float:
        return self.t

    def log_odds_same(self, d: float, clip: float = 8.0) -> float:
        return max(-clip, min(clip, (self.t - d) / self.scale))


def fit_gap(values: list[float]) -> GapCalibration:
    pos = sorted(v for v in values if v > 0)
    if len(pos) < 4:
        m = statistics.median(values) if values else 0.0
        return GapCalibration(m, 1.0)
    lo, hi = int(0.1 * len(pos)), max(int(0.9 * len(pos)), int(0.1 * len(pos)) + 2)
    window = pos[lo:hi]
    best_gap, best_t = -1.0, window[0]
    for a, b in zip(window, window[1:]):
        gap = math.log(b) - math.log(a)
        if gap > best_gap:
            best_gap, best_t = gap, math.sqrt(a * b)
    return GapCalibration(best_t, max(statistics.pstdev(pos), 1e-6))


@dataclass
class RefCalibration:
    """Calibration SUPERVISÉE par la référence : la distribution des distances entre copies du
    MÊME modèle est estimée hors des 33 ep, sur les paires ref-* dont on connaît la vérité
    (ex. ref-01 et ses déguisements), ou sur le jumeau local. Seuil = moyenne + k écarts-types
    (k=2 : règle "pire bruit mesuré + 2 SE" de reports/titouan/t2-mcc-simulation.md) ; log-odds
    linéaires en z-score. Ne dépend donc pas de la structure inconnue des ep."""
    null_mean: float
    null_sd: float
    k: float = 2.0

    def threshold(self) -> float:
        return self.null_mean + self.k * self.null_sd

    def log_odds_same(self, d: float, clip: float = 8.0) -> float:
        return max(-clip, min(clip, (self.threshold() - d) / max(self.null_sd, 1e-6)))


def fit_ref(same_model_distances: list[float], k: float = 2.0) -> RefCalibration:
    """`same_model_distances` : distances mesurées, avec la MÊME statistique et le même N, entre
    paires de référence dont on sait qu'elles sont le même modèle."""
    if not same_model_distances:
        raise ValueError("calibration 'ref' : il faut des distances de référence même-modèle")
    mu = statistics.mean(same_model_distances)
    sd = statistics.pstdev(same_model_distances) if len(same_model_distances) > 1 else 0.0
    return RefCalibration(mu, max(sd, 1e-3 * max(mu, 1e-6)), k)


# Spécification d'algorithme : "<décodeur>_<calibration>".
#   décodeur    : hier (lien moyen + seuil), corr (clustering de corrélation),
#                 bayes (corr + prior de taille gamma=0.15), bayesS (gamma=0.4)
#   calibration : lin (EM 2 gaussiennes sur d), log (EM sur log d), gap (plus grand trou),
#                 ref (null calibré sur des paires de référence même-modèle, fourni par l'appelant)
DECODERS = {"hier": None, "corr": 0.0, "bayes": 0.15, "bayesS": 0.4}
CALIBRATIONS = ("lin", "log", "gap", "ref")


def fit_calibration(values: list[float], calib: str, null_reference: list[float] | None = None):
    if calib == "ref":
        return fit_ref(null_reference or [])
    if calib == "lin":
        return fit_mixture(values, log_space=False)
    if calib == "log":
        return fit_mixture(values, log_space=True)
    if calib == "gap":
        return fit_gap(values)
    raise ValueError(f"calibration inconnue: {calib}")


def parse_algo(algo: str) -> tuple[str, str]:
    if algo == "hierarchical":           # alias historique
        return "hier", "log"
    if algo == "correlation":
        return "corr", "log"
    dec, _, cal = algo.partition("_")
    cal = cal or "log"
    if dec not in DECODERS or cal not in CALIBRATIONS:
        raise ValueError(f"algorithme inconnu: {algo} (attendu <{'|'.join(DECODERS)}>_<{'|'.join(CALIBRATIONS)}>)")
    return dec, cal


def _cluster_block(dist_block: list[list[float]], algo: str, rnd: random.Random, calibration=None, n_restarts: int = 12, null_reference=None) -> list[int]:
    """Partitionne une sous-matrice. `calibration` (Mixture, GapCalibration ou RefCalibration) :
    ajustée sur un ensemble de distances plus large que le bloc (pooling) ; si absente, ajustée
    sur le bloc seul."""
    if len(dist_block) <= 1:
        return [0] * len(dist_block)
    dec, cal = parse_algo(algo)
    if calibration is None:
        calibration = fit_calibration([dist_block[i][j] for i, j in _pairs(len(dist_block))], cal, null_reference)
    if dec == "hier":
        return hierarchical_partition_at_threshold(dist_block, calibration.threshold())
    w = calibrate_weights(dist_block, calibration)
    gamma = DECODERS[dec]
    if gamma == 0.0:
        return correlation_clustering(w, n_restarts, rnd)
    return bayes_size_prior_clustering(w, len(dist_block), n_restarts, rnd, gamma=gamma)


def partition_from_distance_matrix(
    dist: list[list[float]],
    endpoint_ids: list,
    algo: str = "corr_log",
    family_algo: str | None = None,
    seed: int = 0,
    null_reference: list[float] | None = None,
) -> PartitionResult:
    """Point d'entrée de production. `dist` : matrice n x n de distance (listes de listes,
    symétrique, diagonale nulle). `endpoint_ids` : identifiants (ex. noms d'endpoints), dans le
    même ordre que les lignes/colonnes de `dist`. `algo` s'applique à la séparation des modèles
    à l'intérieur de chaque famille ; `family_algo` (par défaut = `algo`) à la séparation des
    familles, qui est un problème beaucoup plus facile (signal très supérieur au bruit, 2.5) :
    le hiérarchique y suffit presque toujours et est le plus rapide. `null_reference` : distances
    même-modèle mesurées sur les ref-* (obligatoire pour la calibration "ref", recommandée).

    Étapes (reports/titouan/t2-statistics.md section 6.3, "famille puis modèle") :
      1. cluster grossier sur `dist` entière -> familles.
      2. un seul mélange à deux composantes est ajusté sur TOUTES les distances intra-famille
         prédites (poolées : même échelle de mesure, 3 à 5 fois plus de paires qu'une famille
         seule) -> seuil et log-odds P(même|d) partagés.
      3. à l'intérieur de chaque famille prédite, cluster fin sur la sous-matrice -> modèles.
      4. étiquette modèle = f"fam{F}_m{k}" pour respecter la contrainte "même modèle => même
         famille" imposée par le format du CSV de soumission.
    """
    n = len(dist)
    rnd = random.Random(seed)
    family_algo = family_algo or "hier_log"
    family_labels = _cluster_block(dist, family_algo, rnd)

    model_labels = [None] * n
    by_family: dict[int, list[int]] = {}
    for idx, f in enumerate(family_labels):
        by_family.setdefault(f, []).append(idx)

    pooled = [dist[i][j] for i, j in _pairs(n) if family_labels[i] == family_labels[j]]
    calibration = fit_calibration(pooled, parse_algo(algo)[1], null_reference) if (len(pooled) >= 4 or null_reference) else None

    for f, members in by_family.items():
        sub = [[dist[a][b] for b in members] for a in members]
        sub_labels = _cluster_block(sub, algo, rnd, calibration=calibration)
        for local_idx, global_idx in enumerate(members):
            model_labels[global_idx] = f"fam{f}_m{sub_labels[local_idx]}"

    family_out = {endpoint_ids[i]: family_labels[i] for i in range(n)}
    model_out = {endpoint_ids[i]: model_labels[i] for i in range(n)}
    return PartitionResult(family=family_out, model=model_out, family_labels_raw=family_labels, model_labels_raw=model_labels)


def partition_raw_labels(dist: list[list[float]], algo: str, family_algo: str, seed: int, null_reference=None) -> tuple[list, list]:
    """Variante interne qui renvoie les étiquettes brutes (family_labels, model_labels en labels
    entiers composites comparables par ==), utilisée par le harnais de simulation pour calculer
    le MCC directement sans repasser par des identifiants d'endpoints."""
    ids = list(range(len(dist)))
    res = partition_from_distance_matrix(dist, ids, algo=algo, family_algo=family_algo, seed=seed, null_reference=null_reference)
    return res.family_labels_raw, res.model_labels_raw


# ---------------------------------------------------------------------------
# 7. Harnais de simulation : scénarios, bruit, agrégation de robustesse
# ---------------------------------------------------------------------------

SCENARIOS = {
    # (n_familles, tailles de famille, fraction "base" par famille)
    "petit_2fam": (2, [6, 5], 0.7),
    "moyen_3fam": (3, [8, 7, 6], 0.65),
    "grand_4fam_eac": (4, [9, 8, 9, 7], 0.6),   # proche de l'ordre de grandeur réel (33 endpoints)
    "familles_desequilibrees": (3, [15, 5, 5], 0.55),
    "beaucoup_variantes": (2, [10, 10], 0.4),     # peu d'endpoints "base", beaucoup de singletons
}

ALGOS = ["hier_lin", "hier_log", "hier_gap", "hier_ref", "corr_lin", "corr_log", "corr_gap", "corr_ref",
         "bayes_lin", "bayes_log", "bayes_ref", "bayesS_log", "bayesS_ref"]
NOISE_LEVELS = [0.15, 0.4, 0.8, 1.3]          # additive : écart-type du bruit de mesure pur
NUISANCE_LEVELS = [0.0, 0.5, 1.5, 3.0]        # additive : magnitude des confondeurs partagés (2.5)
DIV_N_LEVELS = [48, 100, 300]                 # divergence : nombre de réponses par endpoint
DIV_NUISANCE_LEVELS = [0.0, 0.1, 0.2, 0.35]   # divergence : surcroît de divergence des contaminés
N_SEEDS = 40


def simulate_reference_null(generator: str, noise: float, nuisance: float, rnd: random.Random) -> list[float]:
    """Imite le jeu ref-* (9 endpoints : 5 copies du même modèle + 4 variantes, cf. CONTEXTE.md) mesuré
    avec la même statistique, le même N et les mêmes déguisements que les ep : renvoie les distances
    des 10 paires même-modèle, matière première de la calibration "ref"."""
    ref = make_structure(1, [9], 5 / 9, rnd)
    d = make_distance(ref, generator, noise, nuisance, 0.3, rnd)
    return [d[i][j] for i, j in _pairs(ref.n) if ref.model[i] == ref.model[j]]


def run_one(struct: Structure, noise: float, nuisance: float, nuisance_fraction: float, algo: str, seed: int, generator: str = "additive") -> dict:
    rnd = random.Random(seed)
    dist = make_distance(struct, generator, noise, nuisance, nuisance_fraction, rnd)
    null_ref = simulate_reference_null(generator, noise, nuisance, random.Random(seed + 7)) if algo.endswith("_ref") else None
    family_labels, model_labels = partition_raw_labels(dist, algo=algo, family_algo="hier_log", seed=seed, null_reference=null_ref)
    return composite_score(struct.family, struct.model, family_labels, model_labels)


def run_oracle_hierarchical(struct: Structure, noise: float, nuisance: float, nuisance_fraction: float, seed: int, generator: str = "additive") -> dict:
    """Variante "triche" du lien moyen : UN seuil global pour l'étape modèle, choisi en
    connaissant les vraies étiquettes (maximise le MCC_modèle total). Borne supérieure de ce
    qu'un seuil unique bien calibré peut donner ; sert à séparer la perte due au choix du seuil
    de la perte due à la méthode. Un seuil oracle PAR famille serait encore plus haut mais
    sur-ajuste le bruit (il choisit une coupure "chanceuse" différente dans chaque bloc) : non
    atteignable en pratique, donc non rapporté."""
    rnd = random.Random(seed)
    dist = make_distance(struct, generator, noise, nuisance, nuisance_fraction, rnd)
    n = len(dist)
    _, family_labels = hierarchical_oracle_threshold(dist, struct.family)
    by_family: dict[int, list[int]] = {}
    for idx, f in enumerate(family_labels):
        by_family.setdefault(f, []).append(idx)
    candidates = sorted({dist[i][j] for i, j in _pairs(n) if family_labels[i] == family_labels[j]} | {0.0})
    candidates = [c + 1e-9 for c in candidates]
    best = None
    for t in candidates:
        model_labels = [None] * n
        for f, members in by_family.items():
            sub = [[dist[a][b] for b in members] for a in members]
            sub_labels = hierarchical_partition_at_threshold(sub, t)
            for local_idx, global_idx in enumerate(members):
                model_labels[global_idx] = f"fam{f}_m{sub_labels[local_idx]}"
        r = composite_score(struct.family, struct.model, family_labels, model_labels)
        if best is None or r["score"] > best["score"]:
            best = r
    return best


def summarize(values: list[float]) -> dict:
    if not values:
        return {}
    sv = sorted(values)
    n = len(sv)
    return {
        "mean": statistics.mean(values),
        "stdev": statistics.pstdev(values) if n > 1 else 0.0,
        "p10": sv[max(0, int(0.10 * n) - 1)],
        "p50": sv[int(0.50 * n)],
        "p90": sv[min(n - 1, int(0.90 * n))],
        "worst": min(values),
    }


def run_full_simulation(scenarios=None, noise_levels=None, nuisance_levels=None, n_seeds=N_SEEDS, include_oracle=True, generator="additive") -> dict:
    scenarios = scenarios or SCENARIOS
    noise_levels = noise_levels or (NOISE_LEVELS if generator == "additive" else DIV_N_LEVELS)
    nuisance_levels = nuisance_levels or (NUISANCE_LEVELS if generator == "additive" else DIV_NUISANCE_LEVELS)
    results = {}
    for scen_name, (n_fam, sizes, base_frac) in scenarios.items():
        results[scen_name] = {}
        for noise in noise_levels:
            for nuisance in nuisance_levels:
                key = f"noise={noise}_nuisance={nuisance}"
                per_algo_scores = {a: [] for a in ALGOS}
                oracle_scores = []
                for seed in range(n_seeds):
                    struct_rnd = random.Random(zlib.crc32(f"{scen_name}:{seed}".encode()))
                    struct = make_structure(n_fam, sizes, base_frac, struct_rnd)
                    for algo in ALGOS:
                        r = run_one(struct, noise, nuisance, nuisance_fraction=0.3, algo=algo, seed=seed * 97 + 1, generator=generator)
                        per_algo_scores[algo].append(r["score"])
                    if include_oracle:
                        r = run_oracle_hierarchical(struct, noise, nuisance, nuisance_fraction=0.3, seed=seed * 97 + 1, generator=generator)
                        oracle_scores.append(r["score"])
                entry = {a: summarize(v) for a, v in per_algo_scores.items()}
                if include_oracle:
                    entry["oracle_hierarchical"] = summarize(oracle_scores)
                results[scen_name][key] = entry
    return results


# ---------------------------------------------------------------------------
# 8. CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default=None, help="nom de scénario (défaut : tous, voir SCENARIOS)")
    ap.add_argument("--noise", type=float, default=None, help="un seul niveau de bruit (défaut : NOISE_LEVELS)")
    ap.add_argument("--nuisance", type=float, default=None, help="un seul niveau de nuisance (défaut : NUISANCE_LEVELS)")
    ap.add_argument("--generator", default="additive", choices=sorted(GENERATORS), help="; ".join(f"{k}: {v}" for k, v in GENERATORS.items()))
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--out", default=None, help="chemin JSON pour écrire le détail complet")
    args = ap.parse_args()

    scenarios = {args.scenario: SCENARIOS[args.scenario]} if args.scenario else SCENARIOS
    noise_levels = [args.noise] if args.noise is not None else None
    nuisance_levels = [args.nuisance] if args.nuisance is not None else None

    results = run_full_simulation(scenarios=scenarios, noise_levels=noise_levels,
                                   nuisance_levels=nuisance_levels, n_seeds=args.seeds, generator=args.generator)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"résultat détaillé écrit dans {args.out}", file=sys.stderr)

    # résumé compact sur stdout : moyenne par algo, agrégée sur tous les scénarios/niveaux
    agg = {a: [] for a in ALGOS + (["oracle_hierarchical"] if any("oracle_hierarchical" in v for s in results.values() for v in s.values()) else [])}
    for scen in results.values():
        for entry in scen.values():
            for a, stats in entry.items():
                agg.setdefault(a, []).append(stats["mean"])
    print(json.dumps({a: round(statistics.mean(v), 4) for a, v in agg.items() if v}, indent=2))


if __name__ == "__main__":
    main()
