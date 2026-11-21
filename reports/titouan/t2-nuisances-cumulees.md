# T2 · NLL croisée avec TOUS les déguisements empilés (filigrane + system prompt + persona + température)

**Décision débloquée :** la NLL croisée reste utilisable sous déguisements cumulés pour les variantes grossières (fp8 e5m2, LoRA, NF4), mais **plus pour int8 / fp8 e4m3 quand le system prompt caché est inconnu**. Il faut tester le gap contre un plancher de bruit mesuré dans les mêmes conditions (jamais contre 0), et chercher à reconstruire le contexte caché de l'endpoint.

Code : `tracks/t2-model-equality/twin/nuisances_cumulees.py` (v2), `analyse_cumul.py`. Résultats : `results_cumul_v1_{a100,h200}.json` (v1, 60 prompts), `results_cumul_v2_h200.json` (v2, 240 prompts). Jobs : JURECA 15901761 (v1, A100, 185 s), H200 `~/eac-t2` (v1 et v2). Aucune requête vers l'API du hackathon.

## Protocole
OLMo-2-0425-1B-Instruct, 64 tokens. Endpoint A : filigrane KGW clé 111, δ=8, system prompt « assistant laconique » + persona « Nova », T=0,8 (échantillonnage). Endpoint B : clé 222, δ=8, system prompt « tuteur chaleureux » + persona « Sunny », T=0,8, autre graine. gap = NLL_B(texte_A) − NLL_A(texte_A) et symétrique (lecture en forward enseigné, sans biais). Trois conditions sur les mêmes paires :
- **nu** : glouton, sans system prompt ni filigrane (référence) ;
- **ctx** : textes déguisés, relus avec le system prompt de l'auteur (contexte caché connu) ;
- **blind** : mêmes textes, relus avec le prompt utilisateur seul (cas réaliste : contexte caché inconnu).

## 1. « Mêmes poids » : gap ≡ 0 est une identité, pas une mesure
v1 (base vs base dans le même processus, 3 tirages, A100 et H200) : gap = 0,00000, sd 0 sous tous les déguisements. C'est vrai par construction (même fonction, même entrée de lecture), comme dans le rapport filigrane. Le vrai plancher de bruit vient d'une **autre instance** des mêmes poids : v2 compare base bf16 à base fp16.

| plancher (bf16 vs fp16, 240 prompts) | gap moyen | sd du gap | t vs 0 |
|---|---|---|---|
| nu | +0,0001 | 0,003 | 0,4 |
| ctx (3 tirages) | +0,0007 à +0,0011 | 0,008–0,009 | 1,8 à 2,6 |
| blind (3 tirages) | −0,0012 à −0,0017 | 0,011–0,012 | −2,3 à −3,3 |

Les déguisements multiplient l'écart-type du bruit par **3 (ctx) à 4 (blind)**, et surtout le décalent : avec un seuil naïf « t > 2 contre 0 », deux copies des mêmes poids seraient déclarées différentes dans 2 tirages ctx sur 3.

## 2. Poids différents sous déguisements cumulés
v2, 240 prompts (480 gaps), t apparié contre 0 et Welch contre le plancher de la même condition et du même tirage :

| variante | nu : t / Welch | ctx : t / Welch (3 tirages) | blind : t / Welch (3 tirages) |
|---|---|---|---|
| int8 | 2,5 / 2,1 | 3,1–4,4 / 2,1–3,4 | 0,3–1,1 / 1,5–1,9 |
| fp8 e4m3 | 8,4 / 8,1 | 1,9–6,1 / 1,6–5,7 | 1,0–1,7 / 1,6–2,1 |

v1, 60 prompts, ctx, A100 / H200 (même code, seul le tirage change) :

| variante | gap | t A100 | t H200 |
|---|---|---|---|
| int8 | −0,002 / +0,006 | −1,1 | 3,0 |
| fp8 e4m3 | 0,002 / 0,011 | 0,5 | 3,0 |
| fp8 e5m2 | 0,056 / 0,057 | **7,8** | **6,6** |
| LoRA 0.3 | 0,082 / 0,115 | 2,5 | 3,2 |

e5m2 et LoRA 0.3 restent séparés du plancher (le gap e5m2 est ~40× l'écart-type du plancher ctx / √n). La v2 a planté sur e5m2 et LoRA (mémoire GPU saturée sur la H200 partagée, ~14 processus) ; la réplique JURECA v2 (job 15901949) les couvre.

## 3. Dégradation du signal/bruit
Prompts nécessaires pour t = 3 (à partir de la moyenne et de l'écart-type mesurés) pour fp8 e4m3 : **31 (nu) → 58–580 (ctx) → 750–2300 (blind)**, soit ×2 à ×75. Pour int8 : 360 (nu) → 110–230 (ctx) → 1700–18 000 (blind). Trois causes : (a) l'échantillonnage à T=0,8 et le filigrane δ=8 produisent des textes peu probables, dont la NLL varie beaucoup d'un prompt à l'autre (sd du gap ×3–5) ; (b) le tirage aléatoire change le t d'un facteur 3 entre deux répétitions (e4m3 ctx : 1,9 à 6,1) ; (c) relire sans le system prompt de l'auteur ajoute un biais systématique, présent même pour les mêmes poids.

## 4. Corrections proposées
1. **Tester contre le plancher, jamais contre 0** : z = (gap − moyenne_plancher) / sd_plancher·√(1/n_var + 1/n_pl), plancher mesuré avec une seconde instance des mêmes poids (autre dtype ou autre lot) **sous les mêmes déguisements**. Sans ça, faux positifs « différents » pour des jumeaux.
2. **Reconstruire le contexte caché** : passer de blind à ctx multiplie le t par ~3–4. Pistes : demander à l'endpoint de répéter ses instructions, ou choisir parmi des system prompts candidats celui qui minimise la NLL du lecteur sur les réponses de l'endpoint.
3. **Plus de prompts et regroupement des tirages** : la variance du gap par prompt décide tout ; 240 prompts suffisent en ctx pour int8/e4m3, pas en blind. Avec 100 réponses par endpoint, int8 et fp8 e4m3 ne sont **pas** séparables en blind : ne viser que les variantes grossières, ou combiner avec d'autres indices.
4. **Réduire la variance** : prompts courts et très contraints (calcul, capitale) où la sortie échantillonnée reste proche du glouton ; tronquer la NLL aux premiers tokens ; médiane du gap au lieu de la moyenne.

## Limites
Un seul modèle 1B, un seul jeu de déguisements, δ=8 (fort), T=0,8. v2 sans e5m2/LoRA (OOM, à compléter avec JURECA 15901949). Plancher fondé sur un seul type d'instance (fp16) ; un vrai serveur ajouterait du bruit de lot/noyau.
