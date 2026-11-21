# T2 · Détecter int8 face au bruit de calcul GPU (jumeau local, zéro requête API)

Décision débloquée : peut-on séparer un endpoint int8 d'un endpoint fp du même modèle avec la comparaison gloutonne prompt par prompt ?
Modèle : OLMo-2-0425-1B-Instruct bf16, 300 prompts (`build_prompts`, seed 0), max_new=96, bs de référence=16.
Scripts : `tracks/t2-model-equality/twin/exp2_int8_detection.py`, `exp2b_power.py`. Résultats bruts : `results_exp2_a100.json`, `results_exp2_h200.json`, `results_exp2b_h200.json`.
Métriques : `identical_frac` (réponse entière identique, IC Wilson 95 %) et `token_match_frac` (tokens concordants avant la 1re divergence / tokens totaux).

## 1. Plancher de bruit (même modèle bf16)
| Comparaison | A100 identique | H200 identique |
|---|---|---|
| même bs, 2e passage | 100 % [98,7-100] | 100 % |
| déterminisme forcé (`use_deterministic_algorithms`) vs base | 100 % | 100 % |
| déterministe bs1 vs déterministe bs16 | 70 % | 73 % |
| bs1 / bs8 / bs32 / bs64 vs bs16 | 70 / 66 / 68 / 70 % | 73 / 76 / 95 / 94 % |
| même config, A100 vs H200 | 68 % [62,9-73,3] | |

Constats mesurés : (a) le « bruit » n'est pas aléatoire : à lot fixé, tout est reproductible à 100 %, et forcer le déterminisme n'enlève pas l'effet de la taille de lot (70 % reste 70 %). C'est la composition du lot (et le GPU) qui change le chemin de calcul. (b) Le plancher dépend du matériel et du lot : de 66 % à 95 %. Sur des endpoints réels on ne contrôle ni l'un ni l'autre, donc le seuil prudent est le pire cas mesuré, ~66-68 %.

## 2. Granularité int8 (vs base bf16, bs16)
| Variante | A100 identique | H200 identique | H200 token_match |
|---|---|---|---|
| int8 poids, par tenseur | 6,0 % | 6,0 % | 0,16 |
| int8 poids, par canal (= exp1) | 50,3 % [44,7-56,0] | 47,0 % | 0,67 |
| int8 poids, groupe 128 | 54,3 % | 56,3 % | 0,73 |
| int8 poids, groupe 64 | 54,0 % | 56,7 % | 0,72 |
| int8 poids, groupe 32 | 55,3 % | 57,0 % | 0,74 |
| W8A8 (poids canal + activations par token) | 9,7 % | 12,3 % | 0,26 |
| activations int8 seules | 13,7 % | 10,3 % | 0,28 |

Plus la quantification est fine, plus elle se cache dans le bruit (groupe 32 : 55-57 % contre plancher 66-68 %). Dès que les activations sont quantifiées (schéma réel W8A8 / SmoothQuant), la séparation est nette (10-12 %, IC très loin du plancher).

## 3. Générations plus longues (int8 par canal vs bruit bs32)
| max_new | A100 bruit / int8 | H200 bruit / int8 |
|---|---|---|
| 96 | 68,3 / 50,3 % | 95,3 / 47,0 % |
| 192 | 65,7 / 48,7 % | 93,7 / 46,7 % |
| 384 | 65,0 / 48,3 % | 94,0 / 45,7 % |
Allonger ne change presque rien : l'écart reste ~17 points (A100). La médiane de 1re divergence int8 est à 17-22 tokens : tout se joue au début. Pas la peine de dépenser des tokens longs.

## 4. Puissance au budget réel (H200, bootstrap B=2000)
Paire « bruit » = endpoint bs8 vs endpoint bs32 (même modèle) ; paire « int8 » = int8 bs16 vs fp bs32, sur prompts tirés indépendamment. Seuil = 5e centile du bruit (faux positif 5 %).
| n prompts | int8 canal | int8 groupe128 | int8 groupe32 | W8A8 |
|---|---|---|---|---|
| 50 (identique) | 99,95 % | 95,1 % | 95,5 % | 100 % |
| 100 (identique) | 100 % | 100 % | 100 % | 100 % |
| 50 (token_match) | 98,7 % | 79,4 % | 74,4 % | 100 % |
Réserve : ce bruit H200 bs8/bs32 vaut 77 % ; le pire plancher mesuré (A100 vs H200, ou bs8 sur A100) vaut 66-68 %. Avec un plancher à 68 %, l'écart à int8 groupe32 (55-57 %) n'est plus que ~11-13 points ; la puissance n'a pas été remesurée dans ce cas.

## 5. Réserves mesurées
- W8A8 est chaotique : le même modèle W8A8 sur A100 vs H200 ne donne que 8,7 % d'identique (contre 68 % pour bf16). Deux endpoints W8A8 sur des machines différentes ressembleraient à deux modèles différents : risque de scinder un vrai groupe.
- int8 poids par canal sur A100 vs H200 : 64 % identique, int8 par tenseur : 59 %.
- Un seul modèle (1B) ; un 7B pourrait être plus ou moins robuste. Prompts fixes (5 familles) : la recherche de prompts discriminants est traitée par la session « mining de prompts ».

## Conclusion
- int8 W8A8 (le plus courant en production) : détectable de façon fiable (10-12 % vs ≥66 %).
- int8 poids seuls par canal ou groupe : détectable avec ≥100 prompts si le plancher ressemble au nôtre (puissance 100 % à 5 % de FP sur H200), mais l'écart au pire plancher (66-68 %) n'est que de 10-20 points ; ne pas utiliser de seuil fixe, mais calibrer le seuil sur des paires de bruit mesurées (ref-* ou jumeau).
- Allonger les générations ou forcer le déterminisme n'aide pas.
