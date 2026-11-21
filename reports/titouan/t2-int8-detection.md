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

## Conclusion (exp2)
- int8 W8A8 (le plus courant en production) : détectable de façon fiable (10-12 % vs ≥66 %).
- int8 poids seuls par canal ou groupe : détectable avec ≥100 prompts si le plancher ressemble au nôtre (puissance 100 % à 5 % de FP sur H200), mais l'écart au pire plancher (66-68 %) n'est que de 10-20 points ; ne pas utiliser de seuil fixe, mais calibrer le seuil sur des paires de bruit mesurées (ref-* ou jumeau).
- Allonger les générations ou forcer le déterminisme n'aide pas.

# Exp3 : un meilleur signal que « texte identique » ? (OLMo-1B + Qwen2.5-1.5B)
Script `exp3_margin.py`, 300 prompts, max_new=96. L'API ne donne que du texte (pas de logprobs), donc tous les signaux
se calculent depuis deux textes + un modèle local « scoreur » : `margin` = |écart de logits| entre les deux tokens au
premier point de divergence (scoreur fp32 = le modèle cible, oracle) ; `dnll_self` = |ΔNLL| des deux réponses sous ce
scoreur ; `dnll_for` = |ΔNLL| sous un scoreur ÉTRANGER (Qwen pour OLMo, OLMo pour Qwen : le cas réaliste).
Bootstrap B=2000 ; seuil = quantile 5 % de la paire bruit la PLUS défavorable ; on donne la puissance MINIMALE sur toutes
les paires int8 (int8 contre chaque configuration fp). Bruit OLMo = lots 1/8/16/32 + A100 vs H200 ; Qwen = lots 1/8/16/32.

## 3a. Puissance à n=100 prompts, faux positifs 5 %, pire paire bruit
| Modèle (GPU) | Signal | int8 canal | int8 g128 | int8 g32 |
|---|---|---|---|---|
| OLMo (H200) | texte identique | 97 % | 38 % | 33 % |
| | margin | 98 % | 28 % | 8 % |
| | dnll_self | 77 % | 33 % | 34 % |
| | dnll_for | 70 % | 10 % | 6 % |
| OLMo (A100) | texte identique | 90 % | 40 % | 63 % |
| | margin | 95 % | 42 % | 36 % |
| | dnll_for | 59 % | 13 % | 16 % |
| Qwen (A100) | texte identique | 71 % | 38 % | 30 % |
| | margin | 22 % | 20 % | 4 % |
| | dnll_self | 26 % | 15 % | 5 % |
| | dnll_for | 14 % | 7 % | 2 % |
**Résultat négatif mesuré : aucun signal par score ne bat « texte identique ».** Raison visible dans les moyennes :
le bruit et int8 bifurquent tous deux à des quasi-égalités (OLMo : margin moyen 0,01-0,02 bruit vs 0,02-0,04 int8 ;
aucune divergence > 1 logit). Sur Qwen, le bruit bf16 lui-même bifurque à > 1 logit dans 11-17 % des prompts (plancher
55-62 % identique), ce qui noie tout. Le scoreur étranger est le pire (il ajoute son propre désaccord de style).

## 3b. Ce qui marche : choisir des prompts stables face au bruit (`exp3b_stable_prompts.py`, CPU seul)
On garde les prompts dont toutes les paires bruit de SÉLECTION sont identiques, puis on évalue sur des paires bruit
TENUES À L'ÉCART (jamais vues à la sélection) et sur les paires int8 (jamais regardées). Puissance minimale, 5 % de FP :
| Cas | prompts gardés | n=100 tous prompts (g128 / g32 / canal) | n=100 prompts stables (g128 / g32 / canal) |
|---|---|---|---|
| OLMo H200, sélection lots H200, éval A100 vs H200 | 192/300 | 38 / 32 / 97 % | **87 / 75 / 100 %** |
| OLMo A100, sélection sans lot 1, éval paires lot 1 | 158/300 | 68 / 85 / 98 % | **99 / 100 / 100 %** |
| Qwen A100, sélection sans lot 1, éval paires lot 1 | 139/300 | 31 / 23 / 60 % | **91 / 87 / 100 %** |
| **Transfert : prompts stables d'OLMo → Qwen** | 149/300 | 31 / 24 / 61 % | **20 / 25 / 32 %** (pire) |
Sur prompts stables, le bruit tenu à l'écart remonte à 85 % identique (OLMo et Qwen) alors qu'int8 reste à 53-75 % (OLMo H200).
**Mais la stabilité d'un prompt est propre au modèle** : les prompts stables choisis sur OLMo n'aident pas sur Qwen
(bruit Qwen sur ces prompts : 60-71 %, à peine mieux que 55-62 %). On ne peut donc pas présélectionner les prompts
sur un jumeau local d'une autre famille.

## Conclusion exp3 (décision pour la méthode T2)
1. Garder « texte identique » comme statistique ; abandonner margin / NLL croisée (mesurés inférieurs, 2 modèles).
2. Le levier qui marche est la sélection de prompts stables, mais elle doit se faire sur le modèle lui-même. Piste à
   mesurer ensuite : estimer la stabilité sur les réponses réelles (accord entre endpoints déjà regroupés, ou ref-* si la
   limite est levée), puis ne tester int8 que sur ces prompts. Risque à quantifier : réutiliser les mêmes réponses pour
   choisir et pour tester.
3. W8A8 reste séparé quel que soit le signal (Qwen : 12 % identique vs 55-62 % de bruit).
Limites : modèles 1-1,5B seulement ; fausse quantification simulée (pas de noyaux int8 réels type bitsandbytes/vLLM).
