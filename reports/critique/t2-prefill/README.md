# T2 · Sonde « préfixe assistant + max_tokens=1 » pour voir fp8 sous déguisement

**Décision débloquée** : comment dépenser 25 des 75 réponses `ep` restantes pour séparer une copie fp8 de la base.
Zéro requête. Jumeau SmolLM2-360M (h200c), positions prises dans les 225 vraies réponses `ref` du bloc 0.

## Principe
L'API respecte un préfixe `assistant` imposé et accepte `max_tokens=1`. On impose donc un texte de réponse coupé
juste avant un token où la base et sa copie fp8 ne prédisent pas la même chose, et on tire un token.
La fp8 se **reproduit en local** (poids e4m3 per-tensor, avec ou sans activations dynamiques) : on connaît sa
distribution exacte à chaque position. Chaque tirage x donne LLR = log p_fp8,T(x) − log p_base,T(x), qu'on somme
sur les 25 tirages. Somme > 0 → fp8. T = T_log du guichet, mesuré sur le bloc 0.

## Mesures sur 27 688 positions (T = 0,75), divergence de Jeffreys J entre la base bf16 et un autre lecteur
| lecteur | médiane | p99 | p99,9 | max |
|---|---|---|---|---|
| fp8 poids seuls (W8) | 0,010 | 0,20 | 0,47 | 1,03 |
| fp8 poids + activations (W8A8) | 0,021 | 0,39 | 0,96 | 1,64 |
| base fp32 (bruit de noyau) | 0,001 | 0,011 | 0,022 | 0,18 |
| prompt « helpful assistant » | 0,006 | 0,19 | 0,97 | 3,7 |
| prompt Acme | 0,008 | 0,39 | 1,83 | 11,8 |
| persona pirate | 0,018 | 1,78 | 5,46 | 15,8 |
Le déguisement déplace la distribution autant ou plus que la fp8, mais **pas aux mêmes endroits**. Positions
retenues (règle fixée a priori) : score = min(J_W8, J_W8A8), avec bruit fp32 < score/4 et les 3 prompts système
< score/2. 1649 positions passent ce filtre sur 9285 ; au plus 2 par réponse.

## Puissance simulée, 25 tirages par guichet (1500 essais, `prefill_sim.out`)
P(LLR > 0), soit la part des essais où le guichet est jugé fp8, avec le lecteur W8A8 :
| plan | fp8 W8A8 | fp8 W8 | fp8 à T = 1,0 | base | base fp32 | 3 prompts système | filigrane δ=2 | base à T = 1,0 |
|---|---|---|---|---|---|---|---|---|
| 25 positions × 1 (filtre robuste) | **0,98** | 0,89 | 0,96 | 0,01 | 0,02 | 0,01 | 0,05 | 0,04 |
| 5 positions × 5 | **1,00** | 0,96 | 0,98 | 0,00 | 0,01 | 0,00 à 0,01 | 0,00 | 0,02 |
| 12 positions × 2 | 0,99 | 0,93 | 0,96 | 0,01 | 0,01 | 0,00 à 0,02 | 0,01 | 0,04 |
Lecteur W8 au lieu de W8A8 : même tableau à quelques points près (fausses alertes ≤ 0,09).

## Ce qui n'est PAS couvert (réserves, par ordre de gravité)
1. **Toute la puissance repose sur l'hypothèse que la fp8 du serveur est la nôtre** (per-tensor, e4m3). Une fp8
   per-channel ou par blocs déplacerait d'autres positions. D'où la validation sur `ref-02` avant tout `ep`.
2. **LoRA 0,3** : sa direction est inconnue (ce n'est pas notre LoRA locale), la sélection ne la vise donc pas. Non
   mesuré. Piste : prendre les positions où une LoRA forte de la même famille (ref-07 sur les ref ; ep-14 ou ep-07 si
   la LoRA forte se confirme) choisit un token improbable sous la base. LoRA 0,3 ne déplacerait qu'environ 1/5 de ce
   logit, donc z ≈ 1 en 25 tirages selon un calcul à la main : à mesurer sur les ref, pas à dépenser sur les ep.
3. Les prompts système des ep sont inconnus. Les 3 testés ne trompent pas le test, mais une persona très éloignée
   pourrait le faire.
4. **Temps** : la latence mesurée est d'environ 23 s par requête, avec 4 en parallèle. 225 requêtes (9 ref × 25)
   prennent environ 22 min, si `max_tokens=1` ne change pas cette latence.

## Plan concret
1. **Ref, validation (225 requêtes)** : envoyer `prefill_360_25x1.jsonl` (25 lignes, champ `messages` + `max_tokens: 1`)
   aux 9 ref. Option : `prefill_360_5x5.jsonl` (25 lignes : 5 positions × 5), soit 225 requêtes de plus.
2. Score : `python score_prefill.py journal.jsonl --temps ref-01=0.76,ref-02=0.72,ref-03=0.72,ref-04=0.82,ref-05=0.73,ref-06=0.74,ref-07=0.92,ref-08=1.0,ref-09=0.75`
   (h200c, `prefill_scan_lib.py` à côté). Journal attendu : une ligne par tirage avec `endpoint`, `messages`, `response`.
   **Critère pré-enregistré** : ref-02 LLR > 0 ; ref-01, 03, 05, 06 et 08 LLR < 0. Si ref-02 est < 0, notre fp8
   n'est pas celle du serveur : ne rien dépenser sur les ep.
3. **Ep (25 par guichet)** : refaire le scan, puis `make_prefill_probes.py` avec le jumeau de chaque famille
   (Qwen2.5-3B, Qwen2.5-1.5B, Llama-3.2-1B) et les réponses `ep` du bloc 0 de cette famille comme textes.
   ~20 min de GPU par famille. Même règle, mêmes seuils.

[agent critique · T2]
