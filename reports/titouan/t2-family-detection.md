# T2 · Notre méthode sépare-t-elle deux familles de modèles différentes ? Mesures sur jumeaux locaux

**Décision débloquée :** la détection de famille (10 % du score, et première étape avant de séparer les modèles) peut s'appuyer sur la comparaison de textes gloutons **à condition d'utiliser des prompts ouverts (réponses longues)**. Sur des prompts fermés à réponse courte, deux familles différentes donnent le même texte dans 60 à 76 % des cas : ne jamais décider « même famille » sur un prompt fermé, ni à la majorité de prompts fermés.

Code : `tracks/t2-model-equality/twin/family_detection.py` (+ `family.sbatch`). Résultats bruts : `results_family_h200.json` (VM H200, version finale) et `results_family_a100.json` (JURECA, job `t2family` n° 15901781, 1 A100, 102 s). Les deux machines donnent les mêmes chiffres à ±0,03. **Aucune requête vers l'API du hackathon.**

## Ce qui a été testé
- Trois modèles de taille voisine, de **trois familles différentes** (architecture, tokenizer et données d'entraînement différents) : OLMo-2-0425-1B-Instruct (AllenAI, vocabulaire 100 k), Qwen2.5-1.5B-Instruct (Alibaba, 152 k), SmolLM2-1.7B-Instruct (Hugging Face, 49 k). bf16, décodage glouton, template de chat de chaque modèle.
- Références « même famille » : OLMo contre lui-même avec une autre taille de lot (bruit de calcul) et OLMo contre OLMo int8 (fausse quantification des Linear, comme dans l'exp. 1 : la variante la plus difficile à distinguer).
- **Prompts ouverts** : les 100 prompts d'exp. 1 (calcul, capitale + anecdote, explication, code, reformulation), 64 tokens max.
- **Prompts fermés** (le cas difficile) : 170 questions à réponse unique et courte avec consigne « réponds seulement par… », 10 catégories (addition, table de multiplication, capitale en un mot, pair oui/non, traduction FR d'un mot, symbole chimique, antonyme, jour suivant, suite numérique, première lettre), 16 tokens max.
- Comparaison texte (les tokenizers diffèrent, donc sur la chaîne décodée) : identique strict, identique après normalisation (casse, espaces, ponctuation finale), première divergence en mots.
- NLL croisée (méthode de `t2-watermark-robustness.md`) adaptée aux vocabulaires différents : chaque modèle relit le texte de l'autre **re-tokenisé avec son propre tokenizer** ; le gap est exprimé en **nats par caractère** pour être indépendant du tokenizer.

## 1. Prompts ouverts : la séparation est écrasante

| Paire | texte identique | 1re divergence (médiane, mots) | gap NLL croisée (nats/car) | t apparié | prompts avec gap > 0 |
|---|---|---|---|---|---|
| OLMo vs OLMo (autre lot) | **80 %** | 13 | 0,000 | 0 | 0 % |
| OLMo vs OLMo int8 | **56 %** | 13 | 0,0002 | 1,8 | 55 % |
| OLMo vs Qwen | **0 %** | 3 | 0,163 | 14,1 | 100 % |
| OLMo vs SmolLM2 | **0 %** | 2 | 0,137 | 23,2 | 97 % |
| Qwen vs SmolLM2 | **0 %** | 1 | 0,137 | 16,0 | 99,5 % |

Lecture : sur 300 comparaisons inter-familles (3 paires × 100 prompts), **aucun texte identique**, même après normalisation (borne supérieure à 95 % : ~1 % par prompt). Dans la même famille, même la variante la plus proche (int8) garde 56 % de textes identiques. La première divergence arrive après 1 à 3 mots entre familles (souvent dès la tournure : « The capital of France is Paris, known for… » contre « …and it's also known as… »), contre 13 mots dans la famille. La NLL croisée est 700 fois plus grande entre familles qu'entre OLMo et OLMo int8 (0,14–0,16 contre 0,0002 nats/car), avec un t de 14 à 23.

Point annexe utile : le même modèle sur A100 et sur H200 ne donne le même texte ouvert que sur 62 à 71 % des prompts (fermés : 96–100 %). Le matériel de serving fait partie du plancher de bruit, au même niveau que la taille de lot.

## 2. Le cas difficile : les prompts fermés collisionnent massivement

| Paire | identique strict | identique normalisé |
|---|---|---|
| OLMo vs OLMo int8 (même famille) | 96 % | 99 % |
| OLMo vs Qwen | **71 %** | **76 %** |
| OLMo vs SmolLM2 | **58 %** | **60 %** |
| Qwen vs SmolLM2 | **63 %** | **71 %** |

Taux de collision inter-familles (normalisé) par catégorie :

| Catégorie | OLMo/Qwen | OLMo/Smol | Qwen/Smol |
|---|---|---|---|
| jour suivant, capitale en un mot, symbole chimique | 95–100 % | 95–100 % | 95–100 % |
| antonyme, première lettre | 90 % | 95–100 % | 85–90 % |
| addition, table de multiplication | 100 % | 55 % | 55 % |
| pair oui/non | 60 % | 0 % | 40 % |
| traduction FR d'un mot | 25 % | 30 % | 40 % |
| suite numérique | 0 % | 0 % | 85 % |

Lecture : sur une question à réponse unique, des modèles compétents donnent **la bonne réponse, donc la même** (« 24 », « Paris », « Au », « cold », « Tuesday »). Les collisions ne disparaissent que quand la réponse a une **forme libre** (casse et article : « Cat » / « chat » / « cat » ; « 60 » / « 12 times 5 is 60. ») ou quand un modèle se trompe (« 70 » contre « 60 » pour 100, 90, 80, 70). Un prompt type « 2+2 = ? » ne dit donc presque rien sur la famille : 76 % de chances de collision entre OLMo et Qwen, contre 99 % dans la famille.

Même dans ce cas, la NLL croisée garde un signal si on peut relire le texte (logprobs) : sur les textes **strictement identiques**, l'écart absolu |NLL_B(t) − NLL_A(t)| vaut en médiane 0,015 à 0,038 nats/car entre familles, contre 0,0018 entre OLMo et OLMo int8 (8 à 20 fois plus). Attention : le gap signé utilisé jusqu'ici s'annule par construction sur deux textes identiques (les deux sens sont opposés) ; il faut l'écart absolu. Sans logprobs (API actuelle, texte seul), ce signal n'existe pas.

## 3. Garde-fou : combien de prompts exiger ?
Règle « même famille si les K prompts fermés donnent tous le même texte » : probabilité de faux négatif famille (deux familles différentes jugées identiques), K prompts fermés tirés au hasard parmi les 170 (Monte-Carlo, 20 000 tirages) :

| K | OLMo/Qwen | OLMo/Smol | Qwen/Smol |
|---|---|---|---|
| 1 | 76 % | 60 % | 71 % |
| 3 | 45 % | 21 % | 36 % |
| 5 | 26 % | 7 % | 18 % |
| 8 | 11 % | 1,4 % | 6 % |
| 10 | 6 % | 0,4 % | 3 % |

Règle « à la majorité » sur K fermés : faux négatif 63 à 94 % pour K = 3 à 10, **pire qu'un seul prompt**. À proscrire. Et ces chiffres supposent des prompts tirés au hasard : si on choisit les K dans les catégories à 100 % (capitales, jours, symboles), le faux négatif reste ~100 % quel que soit K.

**Garde-fou proposé, dans l'ordre :**
1. **Décider la famille sur des prompts ouverts**, réponses de 40 à 64 tokens, pas sur des questions fermées. Collision inter-familles mesurée : 0/300 ; intra-famille : ≥ 56 % (int8) et ≥ 62 % (changement de GPU).
2. **Exiger l'accord sur plusieurs prompts à la fois** : « même famille » si au moins 2 textes ouverts identiques sur 20 (ou un taux ≥ 10 %). Faux négatif : avec un taux de collision ≤ 1 % par prompt, P(≥ 2 sur 20) ≤ 1,7 % ; faux positif (même famille jugée différente) avec 56 % de collisions : P(< 2 sur 20) < 10⁻⁵.
3. **Un indicateur bon marché en plus** : la 1re divergence. Entre familles, médiane 1–3 mots ; dans la famille, 13 mots.
4. **Si les prompts fermés sont quand même utilisés** (budget), ne les compter que pour **séparer** (un désaccord est informatif : 1 % de désaccord intra-famille seulement), jamais pour **réunir** ; préférer les catégories à forme libre (traduction, suites, oui/non avec justification) où la collision tombe à 0–40 %.
5. **Avec logprobs** (jumeau, endpoint qui les renvoie) : NLL croisée en nats/car, gap signé sur textes différents, écart absolu sur textes identiques.

## Limites et piège à connaître
- **Le piège inverse : un déguisement de décodage fait passer un même modèle pour une autre famille.** Avec un filigrane KGW δ ≥ 2 (`t2-watermark-robustness.md`), deux copies des mêmes poids ont 0 % de textes identiques et divergent dès 0–4 tokens, exactement comme deux familles différentes ici. La règle « 0 texte ouvert identique » prouve donc « autre famille » **seulement si** on a écarté un biais de décodage (température, filigrane, prompt système). Avant de conclure à un changement de famille, vérifier qu'un même endpoint interrogé deux fois redonne le même texte, et regarder le style (longueur, tournures d'ouverture, formatage) qui, lui, reste propre à la famille.
- Trois familles de 1–2 B seulement. Des modèles plus gros, distillés l'un de l'autre ou fine-tunés sur les mêmes données synthétiques peuvent collisionner davantage sur l'ouvert. Le template de chat par défaut a été gardé (Qwen et SmolLM2 injectent un prompt système qui les nomme) ; un endpoint qui le change déplace un peu les textes mais pas la famille.
- Les prompts ouverts mesurés ici sont ceux d'exp. 1 ; les prompts discriminants du mining (`prompt_mining.py`, réponses de 20 tokens) n'ont pas été testés pour la collision inter-familles. À vérifier avant de les utiliser pour la famille.

## Ce qu'on en déduit pour les 33 endpoints
La première étape (familles) est triviale **si** on lui donne les bons prompts : 20 prompts ouverts par endpoint suffisent à séparer des familles différentes avec une erreur de l'ordre du pourcent, à condition d'avoir d'abord neutralisé les déguisements de décodage. Les questions fermées ne servent à rien pour réunir deux endpoints dans une famille. Prochaine mesure utile : refaire le tableau 1 avec un filigrane ou une température sur l'un des deux côtés, et vérifier qu'un score d'n-grammes ou de style sépare encore « même famille déguisée » de « autre famille ».
