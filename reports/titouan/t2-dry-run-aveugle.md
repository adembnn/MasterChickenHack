# T2 · Dry-run aveugle de bout en bout (jumeau local, budget strict)

Décision débloquée : quelle méthode T2 garder, et à partir de combien de réponses par guichet elle devient fiable, mesuré **à l'aveugle** et non sur des données de mise au point.
Aucune requête vers l'API du hackathon : tout est local (JURECA, h200b, h200c).

## 1. Protocole (ce qui rend le test honnête)

- **Plan caché** : 33 faux guichets générés au hasard à chaque essai, sauvés dans `verite_terrain.json`. Le solveur ne reçoit qu'un objet serveur qui renvoie du **texte** ; la vérité n'est relue qu'après l'écriture de `submission.csv`.
- **Serveur simulé** (`dryrun_blind_test.py`, classe `Server`) : 256 tokens générés max, 1024 tokens d'entrée max, **budget strict** par guichet (`BudgetExceeded` au-delà, comme un HTTP 429). Bruit de serving reproduit (ordre et taille des lots tirés au hasard). Jamais dépassé dans aucun essai (vérifié par assertion).
- **Structure calquée sur le corrigé** `practice_answers.csv` : par famille, un gros cluster « base » déguisé (system prompt, persona, filigrane KGW, T=1,0, seuls ou combinés) et des variantes qui changent l'identité (fp8, int8, nf4, LoRA 0,3/0,7/1,5), en singletons ou en paires.
- **Score** : score officiel 0,1·MCC_famille + 0,9·MCC_modèle (MCC < 0 → 0, MCC_modèle restreint aux paires de vraie même famille) **et** le vrai `score_practice.py` exécuté tel quel (ep-XX renommés ref-XX).
- **Graines neuves** pour toute mesure rapportée ; un détecteur qui a des seuils réglés (variante « norm ») est calibré sur des graines de dev distinctes.
- **Budgets emboîtés** : une collecte au budget max ; le plan à budget b n'utilise que ses b premières réponses (sous-ensemble strict, vérifié par assertion). Statistiquement identique à une collecte séparée à b, et 6 fois moins cher.

## 2. Résultats mesurés

### 2.1 Méthode v3 (texte exact + seuil 0,43), familles OLMo-1B / Qwen-1.5B / SmolLM2-1.7B, guichets gloutons
| Budget | Plans | Score officiel | score_practice.py |
|---|---|---|---|
| 8 | 3 | 0,175 / 0,090 / 0,081 (moy. 0,115) | 0,199 / 0,177 / 0,079 |
| 100 | 1 lu (JURECA, job 15902520) | 0,105 | 0,156 |

**Cause, vue après révélation** : l'étape famille. Chaque copie déguisée de la base (persona, température, filigrane, system prompt) produit des textes trop différents ; on prédit 17 à 23 « familles » pour 2-3 vraies. Comme un modèle doit rester dans une famille, toutes les paires « même modèle » entre copies déguisées sont perdues. Ce n'est pas un problème de budget : à 100 réponses, même effondrement.

### 2.2 Méthode v4 (relecture par une copie locale de la base), même monde glouton, 1 plan
Score officiel : 8 → 0,51 ; 15 → 0,23 ; 20 → 0,43 ; **25 → 0,42** ; 33 → 0,31 ; 50 → 0,35. Non monotone : un seul plan, le bruit domine. Erreurs : 2 guichets dans la mauvaise famille, copies déguisées classées « variante ».

### 2.3 Jumeau SmolLM2-360M, tirage pur à T=0,8 (base mesurée des ref-*), 1 plan, 1 famille
| Détecteur | 25 | 50 | 100 |
|---|---|---|---|
| v4 | 0,000 | 0,000 | 0,000 |
| Rapport de vraisemblance sous lecteurs locaux | **0,542** | **0,745** | **0,715** |
| Statistiques de NLL sous la base | 0,005 | 0,000 | 0,000 |
| Répartitions des réponses à choix forcé | 0,000 | 0,047 | 0,078 |

Détail par type de paire (lecteurs locaux, budget 100) : copies déguisées 435/435 regroupées, base vs LoRA 1,5 60/60 séparées, **base vs fp8 per-tensor 0/30 séparées**. fp8 per-tensor reste invisible même avec un lecteur bit-identique à la variante cachée : sur ce modèle, l'écart fp8 est noyé dans la variance d'échantillonnage à 100 réponses. v4 regroupe aussi les copies (406/435 à 25) mais ne sépare ni fp8 ni la LoRA forte en régime échantillonné.

### 2.4 Banc final (3 familles × 11 : Qwen2.5-3B, Qwen2.5-1.5B, Llama-3.2-1B, T=0,8)
**En attente.** `bench3.py` (règle des lecteurs `~/casser-fp8/ep_readers.py` + v4 de repli, budgets 25/50/100, variantes en clair ou déguisées, tirage pur vs `generation_config` du modèle) a été lancé sur h200c puis arrêté à la demande de la session principale pour libérer la GPU pendant l'analyse du test 1, avant qu'un essai n'aboutisse. À relancer.

## 3. Ce qu'on en déduit
1. **v3 est à abandonner** : son effondrement vient de la famille, pas du budget.
2. En **régime échantillonné** (le cas réel), seuls les lecteurs par vraisemblance tiennent ; v4, NLL brute et répartitions de réponses ne voient pas les variantes.
3. **fp8 per-tensor est le point faible** : 0/30 paires base-fp8 séparées même dans le cas le plus favorable. C'est la première chose à améliorer (plus de tokens par réponse, prompts à faible marge, ou accepter de fusionner fp8 avec la base).
4. Une seule mesure par condition sur 2.2 et 2.3 : ce sont des indications, pas des moyennes. Le tableau moyenné vient du banc 2.4.

## 4. Limites (à lire avant d'utiliser ces chiffres)
- Familles = vrais modèles différents, mais l'hypothèse « la vraie base est dans nos candidats locaux » est forte.
- nf4 approché par une quantification 4 bits par blocs, pas le codebook bitsandbytes ; LoRA cachées entraînées par nous (cible inconnue dans le vrai serveur).
- Lecteurs de quantification bit-identiques aux variantes cachées : optimiste pour fp8/int8.
- JURECA : connexion expirée (MFA) en cours de route, les essais 0, 2, 3 et 4 du run v3 budget 100 n'ont pas été lus.

## 5. Fichiers
`tracks/t2-model-equality/dryrun/` : `dryrun_blind_test.py` (plan caché, serveur, v3), `blind_v4.py` (v4 adaptée, multi-famille), `twin_detectors.py` (4 détecteurs, jumeau SmolLM2-360M), `bench3.py` (banc 3 familles), `analyze_small_budget.py` (variante « réponse normalisée », calibrée sur dev).
Dépendances hors `main` : `clustering.py` (branche t2/clustering-algo), `method_v4.py` (t2/method-v4), `~/casser-fp8/` sur h200c (lecteurs fp8 et LoRA distillée).
