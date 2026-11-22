# T2 : simulation de bout en bout du MCC_modele attendu sur les 33 endpoints

Décision débloquée : quel seuil de décision appliquer au taux de divergence gloutonne, et quel MCC_modèle attendre avec le budget de 100 réponses par endpoint. Code : `tracks/t2-model-equality/mcc_simulation.py` (stdlib, ~20 s, `python3 mcc_simulation.py --trials 2000`). Aucune requête API. Entrée : les taux mesurés sur le jumeau OLMo-2-1B (`twin/results_exp1_a100.json`, n=300 prompts), pas d'hypothèse inventée.

## 0. Résumé
1. Seuil optimal : isoler un endpoint si **plus de 42 à 43 % de ses 100 réponses divergent** de la base de sa famille. Le même tau* (0,41 à 0,43) sort sur les 6 structures testées, et la règle a priori « pire bruit mesuré + 2 écarts-types binomiaux » = 0,43 est à moins de 0,003 MCC de l'oracle. Le seuil se fixe donc **avant** de voir les ep, sans ajustement sur le score public.
2. MCC_modèle attendu : **0,93 à 0,97 en moyenne** (p10 ≈ 0,85 à 0,92 ; médiane souvent 1,0) si la seule différence entre copies du même modèle est le bruit de serving mesuré sur le jumeau.
3. Méthode naïve A (seuil = moyenne du bruit 0,32, sans tenir compte de l'écart-type) : **0,50 à 0,54**. Naïve B (seuil majoritaire 0,50) : 0,70 à 0,92. Gain de notre règle sur la naïve B : +0,09 en moyenne, +0,21 sur la structure riche en int8.
4. Seul int8 est difficile (p = 0,50 contre bruit 0,30 à 0,34) : puissance 0,89 au seuil 0,43. Tout le reste (fp8, nf4, LoRA ≥ 0,3) est détecté à 100 %.
5. Coût du budget 100 au lieu de 300 : −0,03 MCC (−0,06 si beaucoup d'int8). Acceptable.
6. Risque principal : si le bruit de fond réel des ep dépasse celui du jumeau de +0,08 (0,42 au lieu de 0,34), int8 devient indiscernable et le MCC tombe à 0,64 à 0,85. La première chose à mesurer sur ref est donc **le taux de divergence entre deux copies du même modèle** (ref-01 vs ref-02/03/05/06/08 après neutralisation des déguisements).

## 1. Entrée : taux de divergence mesurés (p = 1 − identical_frac, n=300)
| type | p | sd(p̂) N=100 | sd(p̂) N=300 |
|---|---|---|---|
| noise_bs1 / bs8 / bs32 / float16 (même modèle) | 0,300 / 0,340 / 0,317 / 0,333 | 0,047 | 0,027 |
| int8 | 0,497 | 0,050 | 0,029 |
| fp8_e4m3 / fp8_e5m2 | 0,777 / 0,933 | 0,042 / 0,025 | 0,024 / 0,014 |
| nf4 | 0,983 | 0,013 | 0,007 |
| LoRA 0,3 / 0,7 / 1,0 / 1,5 | 0,820 / 0,983 / 1,000 / 1,000 | 0,038 / 0,013 / 0 / 0 | |

Bruit de fond : moyenne 0,323, max 0,340. Écart bruit→int8 = 0,16 soit ~3,3 sd à N=100 : séparable mais pas confortable.

## 2. Modèle de simulation
- Structure = familles ; chaque endpoint est soit une copie du même modèle (p tiré parmi les 4 bruits), soit une variante altérée distincte (p du type). Six structures : S1 3×11 (5 mêmes + 6 variantes : int8, fp8×2, nf4, LoRA 0,3, 0,7) ; S2 6 familles de 5-6 ; S3 une dominante de 15 + 6 triplets ; S4 11 triplets ; S5 3×11 avec variantes subtiles seulement (int8×3, fp8_e4m3×2, LoRA 0,3) ; S6 aléatoire (3 à 7 familles, 30 à 60 % de copies, redessinée à chaque tirage).
- Bruit d'échantillonnage : k ~ Binomial(100, p), p̂ = k/100 (échantillonneur exact par inversion de CDF).
- Décision : p̂ ≤ tau → « même modèle que la base », sinon singleton. Topologie en étoile (chaque endpoint comparé à la base de sa famille), ce que le budget permet.
- MCC sur les 528 paires, famille supposée juste (paires inter-familles = TN, comme dans `mcc.py` de t2-statistics §6.1). 2000 tirages par structure, 101 seuils.

## 3. Puissance par type à N=100 (P(isolé) ; pour les copies c'est un faux positif)
| type | p | tau 0,32 | 0,36 | 0,40 | **0,43** | 0,45 | 0,50 |
|---|---|---|---|---|---|---|---|
| noise_bs1 (FP) | 0,300 | 0,289 | 0,080 | 0,012 | **0,002** | 0,001 | 0,000 |
| noise_bs8 (FP, pire cas) | 0,340 | 0,620 | 0,296 | 0,086 | **0,024** | 0,009 | 0,000 |
| noise_float16 (FP) | 0,333 | 0,566 | 0,249 | 0,066 | **0,017** | 0,006 | 0,000 |
| int8 (puissance) | 0,497 | 1,000 | 0,996 | 0,967 | **0,891** | 0,798 | 0,434 |
| fp8_e4m3 et au-delà | ≥ 0,777 | 1,000 | 1,000 | 1,000 | **1,000** | 1,000 | 1,000 |

Lecture : au seuil 0,32 (moyenne du bruit), une copie sur deux est isolée à tort ; au seuil 0,50, un int8 sur deux est fusionné à tort. Le compromis se situe à 0,42-0,43.

## 4. MCC en fonction du seuil (moyenne sur 2000 tirages ; * = optimum)
| tau | S1 3×11 | S2 6 fam. | S3 dominante | S4 triplets | S5 subtil | S6 aléatoire |
|---|---|---|---|---|---|---|
| 0,30 | 0,367 | 0,351 | 0,336 | 0,312 | 0,367 | 0,330 |
| 0,32 | 0,535 | 0,529 | 0,505 | 0,500 | 0,535 | 0,505 |
| 0,36 | 0,815 | 0,812 | 0,793 | 0,796 | 0,813 | 0,796 |
| 0,40 | 0,947 | 0,950 | 0,943 | 0,942 | 0,931 | 0,944 |
| 0,42 | **0,964*** | 0,971 | 0,967 | 0,967 | 0,931 | 0,965 |
| 0,43 | 0,963 | **0,972*** | **0,970*** | **0,971*** | 0,915 | **0,967*** |
| 0,44 | 0,958 | 0,970 | 0,969 | 0,970 | 0,895 | 0,965 |
| 0,46 | 0,938 | 0,958 | 0,960 | 0,957 | 0,836 | 0,951 |
| 0,50 | 0,880 | 0,918 | 0,921 | 0,914 | 0,704 | 0,905 |
| 0,60 | 0,806 | 0,863 | 0,867 | 0,855 | 0,568 | 0,842 |
| S5 optimum | | | | | **0,934* (0,41)** | |

Paires positives : 30 / 18 / 36 / 11 / 30 / ~29 sur 528. La courbe est plate entre 0,40 et 0,45 (perte < 0,02) et s'effondre sous 0,36 : en cas de doute, arrondir **vers le haut** (un int8 raté coûte |cluster| FP ; une copie isolée coûte |cluster|−1 FN, mais le bruit frappe toutes les copies à la fois).

## 5. Notre méthode vs naïve (MCC moyen [p10 ; p90], N=100)
| structure | naïve A : tau = moy. bruit 0,32 | naïve B : tau = 0,50 | **nous, a priori 0,43** | nous, global 0,42 | oracle |
|---|---|---|---|---|---|
| S1 3×11 | 0,535 [0,36 ; 0,72] | 0,880 [0,80 ; 0,92] | **0,963 [0,92 ; 1,00]** | 0,964 | 0,964 (0,42) |
| S2 6 familles | 0,529 [0,33 ; 0,70] | 0,918 [0,86 ; 1,00] | **0,972 [0,92 ; 1,00]** | 0,971 | 0,972 (0,43) |
| S3 dominante | 0,505 [0,28 ; 0,75] | 0,921 [0,87 ; 0,99] | **0,970 [0,88 ; 1,00]** | 0,967 | 0,970 (0,43) |
| S4 11 triplets | 0,500 [0,30 ; 0,67] | 0,914 [0,85 ; 1,00] | **0,971 [0,92 ; 1,00]** | 0,967 | 0,971 (0,43) |
| S5 subtil | 0,535 [0,36 ; 0,72] | 0,704 [0,63 ; 0,79] | **0,915 [0,84 ; 1,00]** | 0,931 | 0,934 (0,41) |
| S6 aléatoire | 0,505 [0,30 ; 0,71] | 0,905 [0,79 ; 1,00] | **0,967 [0,88 ; 1,00]** | 0,965 | 0,967 (0,43) |

« Comparer les moyennes sans correction » (naïve A) divise le MCC par deux : avec sd = 0,047 à N=100, la moitié des copies dépassent la moyenne du bruit par pur hasard. La correction est entièrement dans le terme +2 SE.

## 6. Coût du budget : N=100 vs N=300 (seuil optimal de chaque N)
| structure | MCC N=100 | MCC N=300 | perte |
|---|---|---|---|
| S1 / S2 / S3 / S4 / S6 | 0,964 / 0,972 / 0,970 / 0,971 / 0,967 | 0,998 à 0,999 | −0,027 à −0,035 |
| S5 subtil | 0,934 | 0,998 | −0,064 |

À N=300 le problème est résolu ; à N=100 il reste 3 à 6 points, portés quasi exclusivement par int8. Si la ronde A ne consomme que 48 réponses par endpoint (plan de t2-statistics §5), le sd monte à 0,068 et il faut recalculer tau = 0,34 + 2×0,068 = 0,48 : l'int8 passe alors à ~55 % de puissance. **Recommandation : dépenser les 100 réponses sur la statistique de divergence pour tous les endpoints d'une famille où un int8 est plausible**, plutôt que garder 12 en réserve.

## 7. Sensibilité au bruit réel des endpoints (S1 et S5, seuil a priori recalculé sur le bruit décalé)
| bruit max réel | 0,34 (jumeau) | 0,38 | 0,42 | 0,46 | 0,50 |
|---|---|---|---|---|---|
| S1 a priori / oracle | 0,968 / 0,968 | 0,908 / 0,928 | 0,848 / 0,867 | 0,811 / 0,813 | 0,797 / 0,803 |
| S5 a priori / oracle | 0,922 / 0,937 | 0,765 / 0,861 | 0,644 / 0,740 | 0,579 / 0,606 | 0,560 / 0,563 |

Dès +0,08 de bruit, int8 est perdu et le plancher est « int8 fusionné dans S » (0,80 sur S1, 0,56 sur S5, cohérent avec le 0,55 de t2-statistics §6.1). Les autres variantes restent détectées quel que soit le bruit ≤ 0,50.

## 8. Limites (à lire avant de s'appuyer sur ces chiffres)
- Les p « même modèle » du jumeau ne couvrent que le bruit de serving (taille de lot, dtype). Les déguisements ref (prompt système, persona, filigrane, température) ne sont **pas** mesurés ici : s'ils ne sont pas neutralisés avant la comparaison gloutonne, leur divergence est proche de 100 % et cette simulation est sans objet. Elle vaut pour la statistique **après** neutralisation (extraction du prompt système, température 0 forcée si l'API le permet, ou feature invariante).
- Jumeau 1B ; les ep sont peut-être 7B ou 13B, dont la sensibilité à int8/fp8 diffère (MET : ×100 entre familles). Le tableau §7 donne la marge.
- Topologie en étoile : on suppose qu'on identifie une base par famille (l'endpoint de divergence intra-cluster minimale) et que deux variantes altérées ne sont jamais le même modèle. Si l'organisateur a mis deux ep sur le **même** int8, ils seraient à isoler ensemble, pas séparément : non simulé, coût ≤ 1 paire FN par doublon.
- Famille supposée juste (10 % du score, « presque gratuite » d'après t2-statistics). Une erreur de famille se propage ici en FP/FN non comptés.
- Décision par seuil dur ; un décodage à lien moyen ou MBR (t2-statistics §6.3) ne peut que faire mieux sur les cas à 0,40-0,45.

## 9. Ce que ça change pour le plan
1. Figer **tau = 0,43** dans le pipeline (ou `max(bruit ref) + 2·sqrt(p(1−p)/N)` recalculé sur le bruit réellement mesuré entre copies ref), posté sur l'issue #4 avant tout appel ep.
2. Étape 0 sur ref : mesurer le taux de divergence gloutonne ref-01 vs chaque copie déguisée après neutralisation. Si > 0,38 : prévenir que int8 sera probablement fusionné et viser 0,80-0,85 ; si ≤ 0,34 : viser 0,95.
3. Ne pas « comparer les moyennes » ; ne pas utiliser 0,50.
4. Reproduire : `python3 tracks/t2-model-equality/mcc_simulation.py --trials 2000 --seed 0` (sortie complète, 8 sections, dont une anatomie d'un tirage avec un int8 observé à 0,43 fusionné à tort : MCC 0,921).
