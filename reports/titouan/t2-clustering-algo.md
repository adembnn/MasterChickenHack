# T2 : de la matrice de distances à la partition (famille, modèle), quel algorithme ?

Décision débloquée : quel décodeur et quel seuil utiliser pour transformer la matrice de distances entre les 33 `ep-*` (ex. taux de divergence gloutonne) en CSV `endpoint,family,model`, pour maximiser 0,1·MCC_famille + 0,9·MCC_modèle sur les 528 paires.
Code : `tracks/t2-model-equality/clustering.py` (stdlib pure, aucune requête API, aucun GPU). Simulations : 5 structures × 16 (ou 12) niveaux de bruit/nuisance × 30 tirages × 13 variantes d'algorithme + oracle, soit ~70 000 partitions.
Légende : [sim] mes simulations ; [hyp] hypothèse ; [calc] calcul.

## 0. Résumé (ce qui change nos décisions)
1. **Le décodeur compte peu, le seuil compte énormément.** À calibration égale, lien moyen, clustering de corrélation et MAP bayésien sont à ±0,01-0,02 l'un de l'autre [sim]. Passer d'un seuil choisi sans référence à un seuil calibré sur les `ref-*` rapporte +0,10 sur le générateur réaliste (0,79 → 0,89). L'oracle (meilleur seuil unique, choisi avec la vérité) reste à 0,96 : le gros du gain restant est dans le seuil, pas dans l'algorithme.
2. **Recommandation production** : `algo="corr_ref"` (clustering de corrélation, poids log-odds calibrés sur la loi nulle des paires `ref-*` même-modèle, seuil = moyenne + 2 écarts-types). Score moyen 0,892, p10 0,787 sur le générateur divergence ; `hier_ref` est équivalent (0,891) et plus simple à expliquer : prendre celui-là si on veut un seul paramètre lisible.
3. **Condition impérative** : la loi nulle de référence doit être mesurée avec **la même statistique, le même N et les mêmes déguisements/neutralisations** que les `ep-*`. Sinon la calibration `ref` devient la pire option : avec une nuisance forte absente de la référence, `*_ref` tombe à 0,22 contre 0,34-0,38 pour les seuils non supervisés [sim, générateur additif, nuisance 3,0].
4. **Le prior « gros cluster base + singletons »** (MAP bayésien) n'apporte rien de mesurable quand les poids sont déjà calibrés (+0,003 au mieux), et un prior fort dégrade (bayesS_ref 0,870 contre 0,892 ; pire cas à 0). À ne pas utiliser en production, sauf comme bris d'égalité.
5. **Calibration sans référence** (si les `ref-*` sont inutilisables) : EM à 2 gaussiennes sur les distances intra-famille poolées. En échelle log c'est le plus robuste sur le générateur additif (0,436-0,440) ; en échelle linéaire c'est légèrement meilleur sur les taux de divergence bornés dans [0, 1] (0,790 contre 0,781). Le « plus grand trou » est toujours dernier ou presque.
6. La famille (10 %) est réussie dans quasiment tous les tirages par un lien moyen coupé par EM : l'écart entre familles (~100 % de divergence) écrase le bruit.

## 1. Ce qui est implémenté (`clustering.py`)
Point d'entrée production :
```python
from clustering import partition_from_distance_matrix
res = partition_from_distance_matrix(dist, endpoint_ids, algo="corr_ref",
                                     null_reference=ref_same_model_distances)
res.family[ep], res.model[ep]   # étiquettes prêtes pour le CSV (sans virgule)
```
- `dist` : matrice n×n symétrique, diagonale nulle, « plus grand = plus différent ». `null_reference` : liste des distances entre paires `ref-*` connues comme même-modèle (ex. ref-01 contre ses 4 déguisements, et les déguisements entre eux = 10 paires).
- Pipeline (rapport t2-statistics.md §6.3) : (1) familles par lien moyen + seuil EM log ; (2) une seule calibration ajustée sur toutes les distances intra-famille **poolées** (3 à 5 fois plus de paires qu'une famille seule) ; (3) modèles décodés dans chaque famille ; (4) étiquette modèle = `fam{F}_m{k}`, donc « même modèle ⇒ même famille » par construction.
- Algorithmes, nommés `<décodeur>_<calibration>` :
  - Décodeurs : `hier` (dendrogramme à lien moyen coupé au seuil) ; `corr` (clustering de corrélation : maximise Σ w_ij sur les paires mises ensemble, initialisation KwikCluster puis déplacements gloutons, 12 redémarrages) ; `bayes` / `bayesS` (même objectif + log-prior de taille en U, g(k) = −γ(k−1)(n−k), γ = 0,15 / 0,4, qui favorise singletons et gros cluster et pénalise les tailles intermédiaires).
  - Calibrations : `lin` / `log` (EM à 2 gaussiennes sur d ou log d, seuil = croisement des densités, w_ij = log-odds a posteriori bornés à ±8) ; `gap` (milieu du plus grand trou entre distances triées) ; `ref` (seuil = μ₀ + 2σ₀ de la loi nulle de référence, w_ij = z-score centré sur ce seuil : c'est la règle « pire bruit + 2 SE » de t2-mcc-simulation.md).
- Oracle (`run_oracle_hierarchical`) : lien moyen avec le meilleur seuil **unique** choisi en connaissant la vérité. C'est une borne supérieure de ce qu'un seuil parfaitement calibré donnerait (un seuil oracle par famille sur-ajusterait le bruit, il n'est donc pas rapporté).
- Score : `composite_score` applique exactement la règle de l'énoncé (MCC_famille sur les 528 paires, MCC_modèle seulement sur les paires de **vraie** même famille, MCC = 0 si dénominateur nul).

## 2. Données simulées
Structures (`SCENARIOS`) : par famille, une fraction « base » (le même modèle, servi sous plusieurs déguisements) et le reste en singletons de variantes tirées sans remise parmi int8, fp8, nf4, LoRA 0,3/0,7/1,5, autre quantification.
| scénario | familles | tailles | fraction base |
|---|---|---|---|
| petit_2fam | 2 | 6, 5 | 0,70 |
| moyen_3fam | 3 | 8, 7, 6 | 0,65 |
| grand_4fam_eac | 4 | 9, 8, 9, 7 (= 33) | 0,60 |
| familles_desequilibrees | 3 | 15, 5, 5 | 0,55 |
| beaucoup_variantes | 2 | 10, 10 | 0,40 |
Deux générateurs de distances :
- **divergence** (le plus réaliste pour notre statistique) : d_ij = part des N prompts où le texte glouton diffère, tirage binomial. Taux tirés du jumeau OLMo-2-1B (twin/exp1, repris dans t2-mcc-simulation.md) : même modèle 0,32 ; int8 0,50 ; fp8 0,85 ; LoRA 0,3 0,82 ; nf4/LoRA ≥ 0,7 ≥ 0,98 ; autre famille 0,995. Deux variantes d'une même base divergent au moins autant que la plus divergente des deux. Nuisance : 30 % des endpoints portent un déguisement non neutralisé, qui ajoute +δ·(1−p) à leurs divergences (δ ∈ {0 ; 0,1 ; 0,2 ; 0,35}). N ∈ {48, 100, 300}.
- **additive** (stress test, statistique non bornée type MMD) : bruit demi-normal par paire + gigue par endpoint (hétéroscédasticité) + signal par variante (int8 0,15 … LoRA 1,5 4,0 ; famille 12) + nuisance partagée sur 30 % des endpoints, de magnitude ∈ {0 ; 0,5 ; 1,5 ; 3,0}, qui peut dépasser les signaux faibles (t2-statistics.md §2.5). Bruit σ ∈ {0,15 ; 0,4 ; 0,8 ; 1,3}.
La calibration `ref` reçoit les 10 distances même-modèle d'un jeu `ref-*` simulé (9 endpoints : 5 copies + 4 variantes), mesuré avec le même générateur, le même N et la même nuisance que les ep.

## 3. Résultats chiffrés [sim, 30 tirages par cellule]
### 3.1 Générateur divergence (réaliste)
| algo | score moyen | p10 moyen | pire cas | écart à l'oracle |
|---|---|---|---|---|
| oracle (seuil unique parfait) | 0,956 | 0,902 | 0,568 | — |
| **corr_ref** | **0,892** | 0,787 | 0,468 | 0,064 |
| **hier_ref** | **0,891** | 0,789 | 0,468 | 0,065 |
| bayes_ref | 0,888 | 0,794 | 0,509 | 0,068 |
| bayesS_ref | 0,870 | 0,763 | 0,000 | 0,086 |
| hier_lin / bayes_lin / corr_lin | 0,790 / 0,790 / 0,789 | 0,63 | 0,000 | 0,17 |
| bayesS_log / hier_log / bayes_log / corr_log | 0,782 / 0,781 / 0,780 / 0,779 | 0,61 | 0,000 | 0,18 |
| corr_gap / hier_gap | 0,755 | 0,578 | 0,194 | 0,20 |

Robustesse à la nuisance (score moyen) :
| algo | δ=0 | δ=0,1 | δ=0,2 | δ=0,35 |
|---|---|---|---|---|
| corr_ref | 0,953 | 0,935 | 0,863 | 0,819 |
| hier_ref | 0,953 | 0,935 | 0,860 | 0,817 |
| bayes_ref | 0,953 | 0,931 | 0,854 | 0,815 |
| hier_lin | 0,791 | 0,791 | 0,796 | 0,781 |
| hier_log | 0,793 | 0,786 | 0,790 | 0,755 |
| oracle | 0,994 | 0,991 | 0,966 | 0,874 |
Par N (48 / 100 / 300) : corr_ref 0,883 / 0,893 / 0,901 ; hier_lin 0,788 / 0,786 / 0,795. **Le budget compte peu ici (+0,02 de 48 à 300) ; le seuil compte 5 fois plus.**
Par structure : corr_ref de 0,843 (beaucoup_variantes) à 0,913 (moyen_3fam) ; les non supervisés tombent à 0,65-0,71 sur petit_2fam (trop peu de paires intra-famille, 25, pour que l'EM trouve deux modes : il coupe souvent dans la mauvaise vallée).
Cellules gagnées (hors oracle, 60 cellules) : hier_ref 24, corr_ref 11, bayes_ref 10, bayesS_ref 6, autres 9.

### 3.2 Générateur additif (stress test)
| algo | score moyen | p10 moyen | nuis=0 | nuis=0,5 | nuis=1,5 | nuis=3,0 |
|---|---|---|---|---|---|---|
| oracle | 0,637 | 0,489 | 0,761 | 0,668 | 0,575 | 0,546 |
| bayesS_log | 0,440 | 0,241 | 0,533 | 0,490 | 0,403 | 0,333 |
| bayes_log | 0,439 | 0,247 | 0,531 | 0,487 | 0,403 | 0,336 |
| corr_log | 0,436 | 0,248 | 0,526 | 0,478 | 0,401 | 0,338 |
| corr_ref | 0,435 | 0,267 | **0,614** | **0,541** | 0,357 | 0,228 |
| hier_log | 0,431 | 0,250 | 0,528 | 0,472 | 0,393 | 0,331 |
| hier_ref | 0,424 | 0,262 | 0,614 | 0,537 | 0,330 | 0,216 |
| corr_gap | 0,422 | 0,248 | 0,473 | 0,430 | 0,400 | **0,385** |
| hier_lin | 0,375 | 0,209 | 0,422 | 0,425 | 0,358 | 0,296 |
Lecture : quand la nuisance est faible, `ref` gagne largement (+0,09 sur log) ; quand elle est forte, la loi nulle estimée sur 10 paires ref (dont certaines seulement portent la nuisance) sous-estime la dispersion « même modèle » des ep, le seuil est trop bas, les copies déguisées sont isolées et le gros cluster base éclate. C'est le mode de défaillance n°1 (§0.3). Le prior bayésien gagne quelques millièmes ici, exactement quand le cluster base est fragmenté par la nuisance : c'est son seul usage défendable.
Par bruit σ (0,15 → 1,3) : tous les algorithmes perdent 0,12-0,18 ; l'oracle perd 0,21. Aucun décodeur n'est « plus robuste au bruit » qu'un autre à calibration égale.

### 3.3 Pourquoi le décodeur ne change presque rien [calc]
Avec des poids calibrés w_ij = log-odds(même | d_ij), le clustering de corrélation est le MAP exact sous indépendance des paires. Le lien moyen coupé au même seuil en est une très bonne approximation quand les clusters sont « un gros + singletons », car la moyenne des distances d'un singleton au gros cluster est dominée par le même signal que chaque paire. Les deux ne divergent qu'en cas d'incohérence de triangle (a~b, b~c, a≁c), rare ici : la structure vraie est presque ultramétrique (base ↔ variante ≈ variante ↔ variante). Le clustering de corrélation devient utile si la statistique produit beaucoup de violations de triangle (paires bruitées indépendamment, N petit) : garder `corr` comme défaut ne coûte rien.

## 4. Recommandations opérationnelles
1. **Étape 0 sur `ref-*` (gratuit)** : mesurer la statistique finale (divergence gloutonne après neutralisation, même N, mêmes prompts) sur les 10 paires même-modèle des ref, puis passer cette liste en `null_reference`. Vérifier la dispersion : si σ₀ < 0,02 avec 10 paires, le plancher de 2σ est fragile ; ajouter au besoin des paires jumeau (même modèle, autre lot/dtype) au null.
2. Soumission principale : `partition_from_distance_matrix(dist, eps, algo="corr_ref", null_reference=...)`. Contrôle croisé : `hier_ref` doit donner la même partition à ±1 endpoint ; sinon examiner à la main les endpoints qui changent (paires en violation de triangle).
3. Diagnostic nuisance avant de faire confiance à `ref` : comparer la médiane des distances intra-cluster base des ep (après décodage) à μ₀ des ref. Si elle dépasse μ₀ + 2σ₀, les ep portent une nuisance absente des ref : basculer sur `corr_log` (non supervisé) ou relever le k du seuil (`fit_ref(..., k=3)`), au lieu d'isoler la moitié du cluster base.
4. Ne pas utiliser `bayesS` (prior fort) ni `gap`. Le prior faible (`bayes_*`) peut servir de bris d'égalité entre deux partitions de score voisin.
5. La famille : lien moyen + EM log (`family_algo="hier_log"`, le défaut) ; aucun cas observé où un autre choix aide.

## 5. Limites
- Les taux de divergence viennent d'un jumeau 1B ; la famille réelle des ep peut avoir d'autres taux (int8 surtout). La conclusion « seuil ≫ décodeur » est robuste aux deux générateurs ; les niveaux absolus (0,89) ne le sont pas.
- La nuisance est simulée comme un décalage uniforme d'une fraction des endpoints ; une nuisance structurée (deux déguisements qui se ressemblent entre eux) favoriserait des sous-clusters que ni le prior ni la corrélation ne corrigent.
- 30 tirages par cellule : écart-type de la moyenne ~0,01-0,02, donc les écarts < 0,02 entre décodeurs sont du bruit de simulation.
- Pas d'ILP exact pour le clustering de corrélation (pulp absent) : la recherche locale à 12 redémarrages est probablement optimale à n ≤ 15 par famille, mais ce n'est pas prouvé.
- Reproduire : `python3 tracks/t2-model-equality/clustering.py --generator divergence --seeds 30 --out res.json` (~5 min sur un Mac) ; `--generator additive` pour le stress test.
