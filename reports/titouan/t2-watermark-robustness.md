# T2 · Un filigrane de génération casse-t-il notre comparaison de texte glouton ? Mesures sur le jumeau local

**Décision débloquée :** ne pas conclure « modèles différents » quand deux endpoints de même famille divergent dès les premiers tokens ; remplacer (ou compléter) la comparaison exacte des textes gloutons par une statistique de lecture (log-vraisemblance croisée) partout où un modèle peut relire le texte de l'autre, et par la position de première divergence en boîte noire.

Code : `tracks/t2-model-equality/twin/watermark_robust.py`. Résultats bruts : `tracks/t2-model-equality/twin/results_wm_a100.json`. Job JURECA `t2wm` n° 15901652 (1 A100, 162 s). Aucune requête vers l'API du hackathon.

## Ce qui a été testé
- Modèle : OLMo-2-0425-1B-Instruct, bf16, décodage glouton, 60 prompts (5 familles : calcul, capitale, explication, code, reformulation), 64 tokens max.
- Filigrane KGW (Kirchenbauer et al. 2023, « A Watermark for Large Language Models ») : à chaque pas, le token précédent et une clé secrète tirent une permutation du vocabulaire ; les 25 % premiers tokens (« liste verte », γ = 0,25) reçoivent un bonus δ sur les logits. **Les poids ne changent pas**, seule la règle de choix du token change. Deux clés (k1 = 111, k2 = 222), δ ∈ {0, 1, 2, 4, 8}.
- Trois paires « même modèle » : sans filigrane vs k1 ; k1 vs k2 ; et des paires « modèles vraiment différents » : base vs int8, fp8 e4m3, fp8 e5m2, NF4 (fausse quantification des Linear, comme dans l'exp. 1), chacune avec et sans filigrane.
- Trois statistiques : (1) comparaison exacte des textes (notre méthode actuelle, `exp1_divergence.py`) ; (2) z-score KGW de détection du filigrane ; (3) NLL croisée : chaque modèle relit, en forward enseigné et **sans biais**, le texte produit par l'autre, et on compare à la NLL de l'auteur sur son propre texte (`gap = NLL_B(texte_A) − NLL_A(texte_A)`, dans les deux sens).

## 1. Le filigrane détruit la comparaison exacte
Part de prompts au texte strictement identique (plancher de bruit de l'exp. 1, même modèle à lot/dtype différents : 66–70 %).

| Paire (mêmes poids) | identique | 1re divergence (médiane) |
|---|---|---|
| sans filigrane vs δ=1 | 6,7 % | 9 |
| sans filigrane vs δ=2 | 0 % | 4 |
| sans filigrane vs δ=4 | 0 % | 3 |
| sans filigrane vs δ=8 | 0 % | 1 |
| k1 vs k2, δ=1 | 1,7 % | 8 |
| k1 vs k2, δ=2 | 0 % | 3 |
| k1 vs k2, δ=4 | 0 % | 0 |
| k1 vs k2, δ=8 | 0 % | 0 |

Référence « poids vraiment différents », sans filigrane : int8 61,7 % (méd. 16), fp8 e4m3 20 % (13), fp8 e5m2 6,7 % (11), NF4 1,7 % (10).

Lecture : dès δ = 2 (une force modeste, le papier utilise δ = 2 et plus), deux instances **identiques** du modèle n'ont plus un seul texte en commun. Pour la comparaison exacte, un jumeau filigrané ressemble à une quantification NF4. Une seule des deux API filigranée suffit (ligne « sans filigrane vs δ »). Le filigrane est donc un déguisement qui annule notre signal principal, pas un simple bruit.

Un indice reste exploitable en boîte noire : la **position de la première divergence**. Avec un filigrane elle est médiane 0–4 tokens (le biais agit dès le premier pas), alors que des poids modifiés divergent après 10–16 tokens (les premiers tokens d'une réponse sont très sûrs et résistent à une perturbation des poids). Une divergence quasi immédiate entre deux endpoints d'une même famille sent le déguisement de décodage, pas le modèle différent.

## 2. Détecter le filigrane
**Avec la clé (z-score KGW, 64 tokens)** : moyenne du z sur les 60 textes filigranés k1, et part des textes avec z > 4.

| δ | vraie clé | mauvaise clé (k2) | clé aléatoire | texte sans filigrane, vraie clé |
|---|---|---|---|---|
| 1 | 1,2 (0 %) | 0,2 | 0,5 | 0,24 (0 %) |
| 2 | 2,4 (8 %) | 0,1 | 0,5 | 0,24 |
| 4 | 7,0 (88 %) | −0,4 | 0,6 | 0,24 |
| 8 | 11,2 (100 %) | −0,3 | 0,3 | 0,24 |

Le test marche à partir de δ ≈ 4 sur 64 tokens (il faudrait concaténer plusieurs réponses pour δ = 2) et ne se déclenche jamais sans la clé. Il ne nous sert donc que si le schéma de hachage et la clé sont connus ou énumérables : inutilisable en pratique sur les endpoints.

**Sans clé : taux de « hors-argmax »** quand l'auteur relit son propre texte sans biais (part des tokens générés qui ne sont pas le token le plus probable du modèle). Texte glouton sans filigrane : **0,4 %**. Filigrané : δ=1 → 6–8 %, δ=2 → 16–17 %, δ=4 → 35–37 %, δ=8 → 61 %. Quand un modèle **différent** (quantifié) relit un texte sans filigrane : 1,5 % (int8), 3–4 % (fp8 e4m3), 7 % (fp8 e5m2), 10–11 % (NF4). Un taux au-dessus de ~15 % ne peut pas venir d'une différence de poids de ce type : c'est une règle de décodage manipulée (filigrane, ou autre biais sur les logits). Ce détecteur exige de pouvoir relire le texte avec le modèle supposé : il vaut pour nos jumeaux et pour tout endpoint qui renvoie des logprobs, pas pour l'API du hackathon (pas de logprobs, cf. issue #4).

## 3. La statistique robuste : NLL croisée
`gap` moyen et t de Student apparié (60 prompts × 2 sens) ; `gap = 0` attendu si les deux endpoints partagent les poids.

| Paire | sans filigrane : gap moyen / t | avec filigrane δ=8 (k1 vs k2) : gap / t |
|---|---|---|
| mêmes poids | **0,000 / 0** (identique au bit, pour tout δ) | **0,000 / 0** |
| base vs int8 | 0,001 / 1,4 | 0,003 / 2,6 |
| base vs fp8 e4m3 | 0,006 / 4,7 | 0,008 / 2,9 |
| base vs fp8 e5m2 | 0,024 / 10,8 | 0,030 / 7,0 |
| base vs NF4 | 0,059 / 11,0 | 0,114 / 10,0 |

Pourquoi c'est exactement 0 pour le jumeau : le filigrane modifie le **choix** du token au moment de générer, pas la **probabilité** que le modèle attribue au texte quand il le relit. Deux copies des mêmes poids relisent n'importe quel texte avec les mêmes probabilités, filigrane ou pas. Au contraire, des poids différents sont plus surpris par le texte de l'autre que son auteur (gap > 0), et le filigrane ne détruit pas ce signal (t de 2,6 à 10 avec δ = 8 des deux côtés, contre 0 pour le jumeau). La NLL absolue monte avec le filigrane (0,40 → 2,47 nats/token à δ = 8), mais le gap, lui, est insensible.

Limites : int8 reste au bord de la détection (t 1,4 sans filigrane), comme dans l'exp. 1 où int8 était déjà quasi indistinguable du plancher de bruit par comparaison exacte ; 60 prompts, un seul modèle de 1B, un seul schéma de filigrane (fenêtre 1, hachage simple). Un filigrane qui modifierait aussi la lecture (p. ex. échantillonnage avec graine) n'est pas couvert : avec du sampling, l'exact match est déjà mort et seule la NLL croisée tient.

## Ce qu'on en déduit pour les 33 endpoints
1. **Si un endpoint peut relire un texte** (il renvoie des logprobs, ou c'est un jumeau local), la NLL croisée est la statistique à utiliser : elle vaut 0 pour des poids identiques quel que soit le déguisement de décodage, et sépare les quantifications dès fp8.
2. **En boîte noire pure** (API actuelle, texte seul) : ne jamais déduire « modèles différents » d'un taux d'identité nul si la première divergence est très précoce (médiane ≤ 4 tokens) ; c'est la signature d'un biais de décodage. Dans ce cas, comparer les réponses par un score tolérant au niveau du token (similarité d'n-grammes ou sémantique, ou position de première divergence sur de nombreux prompts courts et très sûrs) plutôt que par égalité stricte.
3. **Détection préalable du filigrane** : impossible sans la clé en boîte noire ; possible sans clé dès qu'on a les logprobs (taux hors-argmax > 15 %). Choisir la méthode de comparaison en fonction de ce que l'endpoint expose, pas d'un test de filigrane.

Prochaine expérience utile : ajouter au jumeau un déguisement « température > 0 avec graine » et un filigrane à fenêtre de contexte plus large, et mesurer si la position de première divergence et un score d'n-grammes (boîte noire) séparent encore « mêmes poids déguisés » de « NF4 ».
