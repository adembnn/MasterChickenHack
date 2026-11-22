# T2 : courbe de calibration « force LoRA (s) -> divergence mesurée »

**Décision débloquée :** quel signal utiliser pour distinguer les trois forces LoRA (0.3 / 0.7 / 1.5),
qui comptent comme **trois modèles différents** dans `practice_answers.csv` (ref-09 = T2, ref-04 = T3,
ref-07 = T4), et avec quelle incertitude pour 100 réponses par endpoint.

Tout est local (jumeau OLMo-2-0425-1B-Instruct, bf16, A100 JURECA, jobs `t2calib` 15901613 et
`t2calib16` 15901633). **Aucune requête vers l'API du hackathon.** Code : `tracks/t2-model-equality/twin/`
(`calibration.py`, `analyze_calibration.py`, `*.sbatch`) ; résultats bruts `calib_results*.json`.

## 1. Protocole

- Base = OLMo-2 1B Instruct, génération gloutonne, 200 prompts de test (5 gabarits : calcul, capitale,
  explication, Python, reformulation), 64 nouveaux tokens max.
- LoRA (r=8, alpha=16, q_proj/v_proj, 150 pas, lr 3e-4) entraînée sur 100 exemples synthétiques
  « Sure thing! + réponse de la base ». **Trois corpus d'entraînement différents** (seeds 123/456/789)
  pour vérifier que la courbe ne dépend pas du corpus.
- Pour chaque LoRA, on multiplie son `scaling` par **s ∈ {0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5,
  0.7, 1.0, 1.5, 2.0}** (grille fine entre 0.05 et 0.5) et on compare à la base :
  `identical_frac` (part des générations identiques), `mean/median_first_div` (position du premier token
  différent, sur les prompts qui divergent).
- Plancher de bruit (base vs base, autre taille de lot ou dtype, mesuré dans exp1, n=300, 96 tokens) :
  **identical_frac = 0.66 à 0.70**, première divergence moyenne à ~27 tokens.

## 2. Courbe de calibration (rang 8, 3 corpus, n=200 prompts par corpus)

| s | identical_frac (moy.) | écart-type entre corpus | IC95 (600 prompts poolés) | mean_first_div (moy.) | médiane first_div |
|---|---|---|---|---|---|
| 0.05 | **0.577** | 0.025 | [0.537, 0.616] | 19.7 ± 1.6 | 16-17 |
| 0.10 | **0.440** | 0.017 | [0.401, 0.480] | 19.4 ± 1.9 | 13-18 |
| 0.15 | **0.360** | 0.018 | [0.323, 0.399] | 17.9 ± 1.7 | 11-14 |
| 0.20 | **0.297** | 0.010 | [0.262, 0.334] | 15.2 ± 1.1 | 9-11 |
| 0.25 | **0.242** | 0.010 | [0.209, 0.277] | 10.8 ± 1.2 | 4-8 |
| 0.30 | **0.193** | 0.015 | [0.164, 0.227] | 7.7 ± 1.1 | 0 |
| 0.35 | **0.155** | 0.009 | [0.128, 0.186] | 4.9 ± 1.3 | 0 |
| 0.40 | **0.123** | 0.015 | [0.099, 0.152] | 3.7 ± 0.7 | 0 |
| 0.50 | **0.082** | 0.013 | [0.062, 0.106] | 2.9 ± 0.6 | 0 |
| 0.70 | **0.043** | 0.040 | [0.030, 0.063] | 2.4 ± 1.6 | 0 |
| 1.00 | 0.000 | 0.000 | [0.000, 0.006] | 0.0 | 0 |
| 1.50 | 0.000 | 0.000 | [0.000, 0.006] | 0.0 | 0 |
| 2.00 | 0.000 | 0.000 | [0.000, 0.006] | 0.0 | 0 |

Lecture : à s=0.3, environ 1 réponse sur 5 reste identique à la base ; à s=0.7, moins de 1 sur 20 ;
à partir de s=1, plus aucune. La valeur s=0.3 mesurée ici (0.193) est la même que dans exp1 (0.18, autre
n, autres longueurs) : la mesure est reproductible.

**Stabilité au corpus d'entraînement : très bonne.** L'écart-type entre les trois corpus est ≤ 0.025 sur
toute la grille, sauf à s=0.7 (0.040 : 0.085 / 0.005 / 0.040, c'est la zone où la fraction tombe vers 0 et
où le bruit binomial sur 200 prompts domine). La courbe ne dépend donc pas du contenu exact des données
d'entraînement, à architecture LoRA fixée.

## 3. Forme analytique et estimateur inverse

`identical_frac(s) ≈ f0 · exp(−s / s0)` : avec le plancher f0 fixé à 0.68, **s0 = 0.240, k = 1.00**
(exponentielle pure, erreur quadratique 0.0009 sur 13 points). Plancher libre : f0 = 0.76, s0 = 0.21,
k = 0.90 (plancher plus haut parce que 64 tokens au lieu de 96 laissent moins d'occasions de diverger).

Inversion `ŝ = −s0 · ln(frac / f0)`, bande = min/max en ré-ajustant corpus par corpus :

| identical_frac observée | ŝ | bande inter-corpus |
|---|---|---|
| 0.50 | 0.07 | [0.07, 0.09] |
| 0.40 | 0.13 | [0.12, 0.14] |
| 0.30 | 0.20 | [0.19, 0.20] |
| 0.20 | 0.29 | [0.28, 0.31] |
| 0.15 | 0.36 | [0.34, 0.40] |
| 0.10 | 0.46 | [0.41, 0.52] |
| 0.05 | 0.63 | [0.53, 0.73] |
| 0.02 | 0.85 | [0.69, 1.01] |
| 0.01 | 1.01 | [0.80, 1.24] |

**Précision selon s :** entre 0.1 et 0.4, ŝ est connu à ± 0.02-0.03 ; à partir de 0.5 l'incertitude
explose (± 0.1 à 0.5) ; au-delà de 1.0, `identical_frac` vaut 0 et **ne porte plus aucune information**.

Position de première divergence : `mean_first_div ≈ 26 · exp(−s / 0.25)`. Elle est plate (~20 tokens) tant
que s ≤ 0.15, puis chute, et la **médiane vaut 0 dès s=0.3** : la LoRA apprise insère un préfixe
(« Sure thing! »), donc dès qu'elle gagne, elle gagne au premier token. Ce signal est donc **moins fin
que `identical_frac`** et dépend de la nature de la LoRA (une LoRA de style diverge plus tard qu'une LoRA de
préfixe). Il reste utile comme indice qualitatif : médiane 0 ⇒ s ≥ 0.3.

## 4. Pouvoir séparateur pour les trois forces attendues

Avec `identical_frac` seule (IC95 binomial recalculé pour **n=100 réponses**, budget réel par endpoint) :

| force | identical_frac attendue | IC95 à n=100 |
|---|---|---|
| 0.3 | 0.19 | [0.12, 0.28] |
| 0.7 | 0.04 | [0.01, 0.10] |
| 1.5 | 0.00 | [0.00, 0.04] |

- **0.3 vs 0.7 : séparables** (intervalles disjoints), à condition de connaître la base (un ref « plain »
  ou un endpoint que l'on croit être la base) et que le décodage soit déterministe des deux côtés.
- **0.7 vs 1.5 : NON séparables** par `identical_frac` ni par `first_div` (toutes deux ~0).

**Ce qui les sépare : une métrique continue sur le texte.** Calculée sur les sorties sauvegardées d'exp1
(n=300, 96 tokens, même type de LoRA) :

| variante | similarité de séquence vs base (SequenceMatcher) | Jaccard des tokens | longueur moyenne (tokens) | part qui diverge au token 0 |
|---|---|---|---|---|
| bruit (int8, pour échelle) | 0.855 ± 0.21 | 0.822 | 53.3 | 0.04 |
| LoRA 0.3 | 0.682 ± 0.27 | 0.634 | 55.9 | 0.40 |
| LoRA 0.7 | 0.670 ± 0.21 | 0.584 | 49.1 | 0.86 |
| LoRA 1.0 | 0.620 ± 0.22 | 0.539 | 39.4 | 1.00 |
| LoRA 1.5 | **0.512 ± 0.24** | **0.440** | **27.1** | 1.00 |

Base : 52.8 tokens. Entre 0.7 et 1.5 : similarité 0.670 → 0.512 (écart 0.16, erreur-type ≈ 0.02 à n=100,
soit ~7 écarts-types) et longueur 49 → 27 tokens. **Une LoRA sur-amplifiée (s > 1) dégrade la fluidité et
raccourcit les réponses** : c'est le signal pour s ≥ 0.7. En revanche la similarité ne distingue **pas**
0.3 de 0.7 (0.682 vs 0.670) : là, c'est `identical_frac` qui tranche.

## 5. Recette proposée pour un endpoint réel

1. Même jeu de prompts (déterministes, réponses longues) sur l'endpoint candidat et sur la base de référence.
2. `identical_frac` : > 0.5 ⇒ même modèle (bruit de calcul) ; 0.10-0.30 ⇒ LoRA faible (s ≈ 0.2-0.45, lire la
   table §3) ; < 0.05 ⇒ LoRA forte ou autre variante, passer à 3.
3. Similarité de séquence et longueur moyenne vs base : ≈ 0.65-0.70 et longueur ≈ base ⇒ s ≈ 0.7 ;
   ≈ 0.5 et longueur divisée par 2 ⇒ s ≈ 1.5. Les quantifications (fp8/int8/nf4) donnent une longueur
   inchangée : la longueur sépare « LoRA forte » de « quantifié ».
4. Incertitude à n=100 : ŝ à ± 0.03 dans [0.1, 0.4] ; classement 0.3 / 0.7 / 1.5 sans ambiguïté si le
   décodage est glouton. Si l'endpoint échantillonne (température > 0), `identical_frac` s'effondre pour
   toutes les variantes et seules les métriques continues restent utilisables.

## 6. Limites (honnêtes)

- **s est relatif à NOTRE adaptateur.** Les organisateurs appliquent 0.3/0.7/1.5 à leur propre LoRA, dont la
  norme et la cible (q/v ? tous les projecteurs ? r ?) sont inconnues. La courbe donne la forme (exponentielle
  en s, saturation vers s ≈ 1) et les ordres de grandeur, pas une correspondance numérique exacte ; §7 teste la
  sensibilité à l'architecture.
- Modèle 1B, 64-96 tokens, prompts synthétiques : le plancher f0 dépend de la longueur et du modèle. Il faut
  le remesurer sur la base réelle (ref « plain » vs ref « system prompt » qui est le même modèle S).
- `identical_frac` suppose un décodage déterministe des deux côtés ; ref-08 (température 1.0) montre que les
  organisateurs utilisent aussi l'échantillonnage.
- Toutes les LoRA testées ont la même « nature » (préfixe + reformulation) ; une LoRA de connaissances ou de
  style changerait la position de première divergence, moins `identical_frac`.

## 7. Sensibilité à l'architecture LoRA (rang 16, q/k/v/o, autres prompts de test)

Job `t2calib16` (15901633, 943 s) : LoRA r=16, alpha=32, sur q/k/v/o (4x plus de paramètres), 3 corpus,
200 autres prompts de test (seed 7). Même grille de s.

| s | identical_frac r=8 (q,v) | identical_frac r=16 (q,k,v,o) | écart |
|---|---|---|---|
| 0.05 | 0.577 | 0.522 | −0.055 |
| 0.10 | 0.440 | 0.398 | −0.042 |
| 0.20 | 0.297 | 0.247 | −0.050 |
| 0.30 | 0.193 | 0.153 | −0.040 |
| 0.40 | 0.123 | 0.078 | −0.045 |
| 0.50 | 0.082 | 0.065 | −0.017 |
| 0.70 | 0.043 | 0.035 | −0.008 |
| ≥ 1.0 | 0.000 | 0.000 | 0 |

- **Même forme** : exponentielle pure (k = 1.00 dans les deux cas), plancher libre f0 = 0.66 (vs 0.76),
  écart-type entre corpus ≤ 0.03 partout.
- **Échelle décalée** : s0 = **0.200** contre 0.240 : l'adaptateur plus gros agit ~20 % plus fort par unité
  de s. Concrètement, une `identical_frac` de 0.15 donne ŝ = 0.36 avec la courbe r=8 et ŝ = 0.30 avec la
  courbe r=16 : **l'incertitude liée à l'architecture inconnue de la LoRA des organisateurs (± 0.05 sur ŝ
  autour de 0.3) dépasse l'incertitude statistique (± 0.02-0.03)**. Il faut donc caler s0 sur les ref
  réels (ref-09 à 0.3, ref-04 à 0.7) plutôt que de réutiliser 0.24 tel quel.
- Le classement ordinal des trois forces est inchangé : 0.3 ⇒ identical_frac 0.15-0.19, 0.7 ⇒ 0.03-0.04,
  1.5 ⇒ 0. La recette du §5 tient avec les deux architectures.

## 8. Ce que ça débloque

- **Pour le pipeline T2** : utiliser `identical_frac` vs base comme estimateur de s dans [0.1, 0.5] (table §3,
  après recalage de s0 sur ref-09/ref-04), puis similarité de séquence + longueur moyenne pour séparer 0.7 de
  1.5. Les deux signaux sont calculables à partir du texte seul (pas de logprobs dans l'API).
- **Prochaine mesure utile (sans dépenser d'ep-\*)** : sur les ref, comparer ref-01 (plain) à ref-03 (system
  prompt, même modèle S) pour mesurer le plancher f0 réel, puis ref-09/ref-04/ref-07 pour lire s0 réel et
  vérifier que la longueur moyenne chute bien à 1.5. Cela coûte 4 x N réponses de ref seulement.
