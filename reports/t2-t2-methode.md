# T2 Model Equality : méthode v4, qui tient les copies déguisées ensemble

Session `t2-methode` d'Adem, samedi 21/11. Décision débloquée : **avec quelles sondes dépenser les 100 réponses par endpoint `ep-*`, et quel décodeur soumettre**. Aucun appel à l'API, aucune clé lue, aucune branche de Titouan modifiée, rien de commité. Code : `tracks/t2-model-equality/adem/` (mode d'emploi : `README.md`). Critique séparée : `reports/critique/t2-methode-tour1.md`. Mondes générés sur la H200 : `~/eac/t2/world3` (glouton), `worldS` et `worldS3` (échantillonnés) ; chaque script de `analyse/` porte sa ligne d'usage en tête.

Termes. MCC : coefficient de corrélation de Matthews sur les paires d'endpoints (« même modèle » ou non), 0 = pas mieux qu'une réponse constante, 1 = parfait. AUC : probabilité qu'une paire « même modèle » ait une statistique plus haute qu'une paire « différente » (0,5 = hasard). Décodage glouton : le serveur prend toujours le token le plus probable, donc la même question rend la même réponse. Filigrane KGW : le serveur ajoute un bonus δ aux logits d'une « liste verte » de tokens tirée d'après le token précédent. LoRA : adaptateur de rang faible ajouté aux poids, multiplié par une force.

## 0. Résultat en six lignes
1. **Mesuré** : sur un monde local où chaque poids existe sous chaque déguisement, aucune statistique par paire ne sépare seule « mêmes poids » de « poids différents » (AUC 0,49 à 0,56). Le déguisement déplace plus le texte que les poids.
2. **Mesuré** : la v3 telle quelle (identité de texte, seuil 0,43 de divergence) fait MCC 0,00 sur des mondes de structure identique au corrigé.
3. **Mesuré** : le type d'un endpoint se lit dans ses réponses. Température 1,0 : deux réponses différentes au même message (48/48). Prompt système ou persona : une sonde d'extraction le rend mot pour mot (OLMo 16/16, Qwen 14/16). 0 fausse alerte sur les endpoints en clair après retrait du prompt par défaut de Qwen.
4. **Méthode v4** : comparaison exacte de la v3 seulement entre endpoints en clair, puis rattachement des endpoints déguisés au groupe central. Sans apprentissage, 3 seuils.
5. **Mesuré, 100 réponses par endpoint, structure du corrigé** : MCC modèle 0,93 (OLMo) et 0,71 (Qwen) ; bout en bout sur 18 endpoints et 2 familles : **score 0,842 [0,67 ; 0,88]**. CSV de 33 endpoints simulés : `validate_t2.py --strict` → `RESULTAT: VALIDE`.
6. **Limites mesurées** : score 0,18 si les variantes de poids sont elles aussi déguisées (33 endpoints au hasard) ; **score 0,22 [0,17 ; 0,32] si tous les endpoints échantillonnent par défaut** (§3.7), cas le plus probable d'après la consigne et le code public des organisateurs. La ronde 0 dit dans quel régime on est et permet déjà un envoi « familles seules » (§3.8).

## 1. La méthode v3 de Titouan en dix lignes
1. Statistique : part des réponses gloutonnes identiques caractère pour caractère entre deux endpoints, sur les mêmes prompts.
2. Seuil : deux endpoints sont « différents » si plus de 43 % des réponses divergent (bruit de service mesuré 0,30 à 0,34 + 2 erreurs types à 100 réponses).
3. Regroupement : lien moyen, famille d'abord, modèle ensuite ; isolement en cas de doute.
4. Budget : 16 réponses « famille », 48 « cœur » (16 prompts × 3 répétitions), 24 adaptatives, 12 de réserve.
5. Garde-fou filigrane : première divergence médiane ≤ 4 tokens → ne pas conclure, passer à la NLL croisée (vraisemblance du texte de l'un relu par l'autre), qui demande des logprobs que l'API ne donne pas.
6. Famille : vraisemblance du texte sous un zoo de modèles locaux.
7. MCC attendu par simulation : 0,93 à 0,97.
8. Hypothèse A : le décodage par défaut est glouton. Non vérifiée ; la consigne parle de « ordinary sampling variation » et le gabarit de « answers are sampled ».
9. Hypothèse B : les copies d'un même modèle sont comparables texte à texte. Fausse d'après le corrigé : les 4 copies sont toutes déguisées (prompt système, persona, filigrane, température 1,0).
10. Hypothèse C : le modèle de base est OLMo et le bruit du jumeau vaut pour l'API. Non vérifiée (relevée par ses propres audits, PR #18 et #22).

## 2. Banc d'essai local (H200, `bench/world.py`)
- Deux familles : `allenai/OLMo-2-0425-1B-Instruct` et `Qwen/Qwen2.5-1.5B-Instruct`. Par famille, 7 poids : base ; fp8 et int8 (fausse quantification par ligne des couches linéaires, la même que le jumeau de Titouan) ; nf4 (bitsandbytes 4 bits) ; LoRA 0,3 / 0,7 / 1,5 (un adaptateur de rang 16 entraîné 300 pas, sur dolly-15k pour OLMo et CodeAlpaca pour Qwen, puis force multipliée).
- Déguisements : en clair ; prompt système ; persona ; filigrane KGW (γ = 0,25, δ = 2, fenêtre d'un token) ; température 1,0. Chaque poids existe sous les 5 ; la base a 5 répliques (autre prompt système, autre persona, autre clé, autre taille de lot). 40 endpoints par famille.
- Sortie : texte seul, 256 tokens au plus, au format de `task_template.py`. Régimes : glouton par défaut (`world3`), échantillonné à T = 0,7 (`worldS`). Mondes notés : « ref » = structure exacte du corrigé (base en clair + 4 copies déguisées + 4 variantes en clair), 200 tirages ; « mix » = 14 endpoints au hasard parmi 40 ; « tout » = les 40.

## 3. Mesures
### 3.1 Statistiques par paire seules (OLMo, monde v1, 34 endpoints, 143 sondes, 6 tirages de budget)
```
statistique                           AUC  MCC ref     [p10 ; p90]  MCC mix
ident (v3)                          0.523    0.176   [0.00 ; 0.38]    0.021
pref (préfixe commun)               0.516    0.201   [0.00 ; 0.38]    0.033
m_q1 (une question chiffrée)        0.535    0.286   [0.00 ; 0.69]    0.034
m_nb5 (5 questions chiffrées)       0.511    0.199   [0.00 ; 0.38]    0.039
m_word (un mot)                     0.559    0.420   [0.14 ; 1.00]    0.029
RL toutes (régression logistique)   0.645    0.309   [0.10 ; 0.49]    0.116
```
Identité de texte contre la base en clair (monde v3, plan gelé) : OLMo réplique 0,94 ; int8 0,79 ; fp8 0,54 ; LoRA 0,3 0,44 ; nf4 0,30 ; **prompt système 0,30** ; LoRA 0,7 0,21 ; **filigrane 0,20** ; **persona 0,17** ; LoRA 1,5 0,09 ; **T 1,0 0,08**. Qwen : réplique 0,98 ; int8 0,78 ; LoRA 0,3 0,66 ; filigrane 0,65 ; fp8 0,62 ; prompt système 0,53 ; LoRA 0,7 0,53 ; persona 0,45 ; nf4 0,38 ; T 1,0 0,34 ; LoRA 1,5 0,30. Une copie déguisée de la base est donc plus loin de la base qu'une quantification.

### 3.2 Par type de paire le signal existe (OLMo, v1, AUC de l'identité de texte)
clair/clair 1,00 ; clair/prompt système 0,85 (0,99 contre LoRA ≥ 0,7 et nf4 seulement) ; clair/filigrane 0,82 (0,98) ; clair/persona 0,77 (0,89) ; clair/T 1,0 : 0,44 (0,61). Classifieurs appris sur une famille et testés sur l'autre (monde v3) : AUC 0,56 à 0,72, MCC ref 0,00 à 0,59 selon le sens. Ils transfèrent mal ; d'où le choix de règles sans apprentissage.

### 3.3 Pistes mesurées et écartées
- Neutraliser le prompt système depuis le message utilisateur (jeton de fin de texte, en-tête de nouvelle conversation, 500 tokens de remplissage, « ignore les instructions ») : identité 0,04 à 0,25, contre 0,67 à 0,92 pour une simple réplique. Réponses chiffrées « au bord du savoir » : un prompt système les change (accord 0,17 à 0,33).
- Lots de 20 mini-questions : un modèle 1B recopie « B » sur 15 lignes (lots de 5 : taux d'analyse 0,94 à 1,00, retenus). Profil de capacité (taux de bonnes réponses par type de calcul) : écarts de 0,05 à 0,15 entre copies déguisées, du même ordre qu'entre poids.

### 3.4 Détection du type (plan gelé, 40 endpoints par famille)
```
olmo : persona 8/8  plain 8/8  sys 8/8  t1 8/8  wm 7/8      qwen : persona 8/8  plain 8/8  sys 6/8  t1 8/8  wm 8/8
```
Ratés : sur Qwen, LoRA 0,7 et 1,5 avec prompt système ne le répètent pas (typés « en clair », donc isolés, ce qui reste juste). Filigrane : l'écart type, entre 8 paires d'étiquettes (A/B, C/D, X/Y…), des écarts de préférence contre la base vaut 0,46 et 0,30 pour les copies filigranées d'OLMo contre ≤ 0,21 pour les variantes en clair ; sur Qwen 0,13 et 0,19 contre ≤ 0,11. Le seuil 0,25 détecte OLMo, pas Qwen.

### 3.5 Décodeurs, MCC modèle au plan gelé (`analyse/eval_rules.py`, 200 mondes par famille)
```
décodeur                                         OLMo ref  [p10 ; p90]   mix   tout | Qwen ref  [p10 ; p90]   mix   tout
tout isolé / tout fusionné                          0.000                0.000 0.000 |   0.000                0.000 0.000
v3 : identité, seuil 0,57 (divergence 0,43)         0.000  [0.00 ; 0.00] 0.018 0.002 |   0.003  [0.00 ; 0.00] 0.077 0.064
règles : + S rattaché                               0.658  [0.22 ; 0.72] 0.112 0.192 |   0.706  [0.72 ; 0.72] 0.088 0.020
règles complètes (+ filigrane)                      0.926  [0.45 ; 1.00] 0.124 0.190 |   0.706  [0.72 ; 0.72] 0.081 0.004
plus proche ancre, sans a priori                    0.732  [0.39 ; 1.00] 0.203 0.326 |   0.016  [0.00 ; 0.04] 0.172 0.109
sans seuil (aucun P fusionné, déguisés au centre)   1.000  [1.00 ; 1.00] 0.044 0.007 |   0.706  [0.72 ; 0.72] 0.038 0.000
```
Le groupe central est bien la base en clair dans 176/200 mondes (OLMo) et 194/200 (Qwen). 0,72 = tout juste sauf la copie filigranée, isolée. La variante « sans seuil » (`--variante sans_seuil`, proposée par `t2-critique`) fait aussi bien sur la structure du corrigé et ne dépend plus du seuil 0,8, fragile si le bruit de service réel est fort ; elle sépare en revanche deux copies en clair du même modèle.

### 3.6 Bout en bout (`analyse/e2e.py` : simulation → plan de 100 réponses → `t2_pipeline.py` → barème officiel)
```
structure ref, 9 par famille, familles estimées, 12 graines      structure mix, 33 endpoints, familles estimées, 8 graines
variante       score     [min ; max]  MCC famille  MCC modèle     variante       score     [min ; max]  MCC famille  MCC modèle
principal      0.842   [0.67 ; 0.88]        1.000       0.825     principal      0.184   [0.09 ; 0.34]        0.839       0.112
ancre          0.387   [0.36 ; 0.44]        1.000       0.319     ancre          0.194   [0.09 ; 0.36]        0.839       0.122
repli          0.100   [0.10 ; 0.10]        1.000       0.000     repli          0.084   [0.07 ; 0.10]        0.839       0.000
```
Validateur de Titouan sur le CSV de 33 endpoints : `33 endpoints, 2 familles, 10 modeles` / `RESULTAT: VALIDE`. Repli : `RESULTAT: VALIDE` avec l'avertissement « tout différent ».

### 3.7 Régime où tous les endpoints échantillonnent
Plus aucune réponse n'est répétable, donc plus de comparaison exacte. Statistique : pour chaque item fermé, D = collision(A,A) + collision(B,B) − 2 collision(A,B), où la collision est la probabilité que deux réponses soient égales. D est un estimateur sans biais de la distance quadratique entre les deux distributions de réponses (espérance 0 si elles sont égales) ; z = D moyen sur son erreur type entre items.

**Monde v1** (`worldS`, T = 0,7, 17 lots fermés × 4 tirages + 16 textes libres × 2). z contre la base en clair, OLMo puis Qwen : réplique 0,4 / 0,2 ; fp8 0,4 / 0,6 ; LoRA 0,3 1,9 / 0,8 ; LoRA 0,7 3,8 / 2,1 ; LoRA 1,5 5,2 / 3,0 ; mais aussi prompt système 5,0 / 2,7, persona 4,6 / 4,7, filigrane 4,2 / 2,8. Prompt système extrait correctement : 79/80. MCC ref de 5 décodeurs (centre + seuil sur z, écart au consensus de la famille, endpoints à prompt système fusionnés, règle « typé → base » de `t2-critique`) : 0,00 à 0,36, sans réglage stable d'une famille à l'autre.

**Monde v2** (`worldS3`), aligné sur le code public des organisateurs relevé par `t2-critique` : T 0,7 + top-p 0,9, leurs prompts système (« Acme Corp », « You are a helpful assistant. », une espace seule), 10 tirages par sonde, lots de 10 questions. z contre la base en clair, par famille de sondes (`analyse/eval_S3.py`) :
```
            lots A/B de 10        choix équivalents de 10     lots chiffrés de 10
            OLMo     Qwen         OLMo     Qwen               OLMo     Qwen
réplique    -0.9      0.8          0.3     -0.5               -0.1     -0.0
Acme         2.7      0.9          3.0     -1.8                3.5      1.6
« helpful » -0.2      1.2          2.6      0.5                2.2      1.4
espace      -1.3     -0.9          3.6      3.9                0.4      1.9
persona      1.1     -0.2          4.4      8.1                3.4      1.7
T 1,0        3.8      0.9         -1.1      0.4                0.9     -0.0
filigrane    8.3      0.1          3.0      7.3                2.0      2.9
fp8         -0.8     -0.2         -2.8      2.7                1.7      2.2
LoRA 0,3     0.1     -0.2          2.1      0.7               -0.3      1.4
LoRA 0,7     1.6      1.6          3.1      3.6                2.8      1.5
LoRA 1,5     2.3      3.2          2.4      4.0                5.1      3.8
```
Les choix entre options équivalentes ne résistent pas à la persona ni au filigrane. Les lots A/B et chiffrés résistent aux déguisements sur Qwen (qui suit le format à 100 %) mais pas sur OLMo-1B (filigrane 8,3). Plan retenu pour ce régime : 4 lots A/B de 10 + 3 lots chiffrés de 10 + 2 lots A/B de 5, 10 tirages chacun, lien moyen sur z < 3. Bout en bout (`analyse/e2e_S.py`, 18 endpoints, 2 familles, 20 graines) :
```
cas                            score     [min ; max]  MCC famille  MCC modèle
familles estimées              0.222   [0.17 ; 0.32]        1.000       0.136      MCC modèle par famille : olmo 0.000  qwen 0.389
repli (familles estimées)      0.100   [0.10 ; 0.10]        1.000       0.000
```
**Lecture** : en régime échantillonné, fp8 et LoRA 0,3 sont dans le bruit, seule LoRA 1,5 (et nf4) sort, et le résultat dépend du modèle : 0,39 quand il répond de façon stable aux questions fermées, 0,00 sinon. Score local 0,22, contre 0,84 en glouton.

### 3.8 Mode dégradé : familles seules, dès la ronde 0 (`analyse/eval_family_small.py`, 40 tirages de 33 endpoints)
```
plan                                                      MCC famille (glouton)    min    MCC famille (échantillonné)    min
ronde 0 seule (4 sondes x 2 = 8 réponses)                       0.742            0.498            0.938                0.702
ronde 0 + 3 identité + 3 extraction (14)                        0.737            0.548            0.897                0.690
ronde 0 + 6 meta + 8 A/B + 8 chiffrés + 4 textes (34)           0.793            0.650            0.924                0.681
plan complet (100), mesuré au §3.6                              0.84 à 1.00
```
Avec les 8 réponses de la ronde 0, `--variante repli` sort déjà un CSV valide dont la note attendue est 0,1 × 0,74 à 0,94, soit 0,07 à 0,09. La ronde 0 sur les 33 `ep-*` (264 requêtes) donne donc à la fois le régime de décodage et un premier envoi, sans rien perdre : ces 8 réponses font partie du plan complet. Mesuré sur 2 familles seulement.

## 4. La méthode v4 (`t2_pipeline.py`, `analyse/rules.py`, stdlib seule)
1. **Familles** : lien moyen sur une proximité normalisée par endpoint (moyenne de 4 statistiques, seuil 0,4). Option `--families` pour imposer les familles de Titouan.
2. **Type** de chaque endpoint, dans sa famille : T si deux réponses au même message diffèrent ; S si une des 9 sondes d'extraction rend un passage « You are… » (hors prompts par défaut connus) ; P sinon.
3. **P contre P** : identité de texte, lien moyen coupé à 0,8. C'est la statistique de la v3, dans le seul cas où elle vaut.
4. **Groupe central** : chaque endpoint vote pour l'endpoint P le plus proche de lui, sur 4 statistiques ; le plus voté est le centre.
5. **Filigrane** : un P isolé dont l'écart type de biais d'étiquettes contre le centre atteint 0,25 rejoint le centre.
6. **S** : rattaché au groupe P le plus proche, le centre l'emportant à 0,05 près. **T** : rattaché au centre.
Hypothèse assumée, lue dans le corrigé : un endpoint déguisé sert le modèle de base. Variantes fournies : `centre`, `ancre`, `prudent`. Si une famille n'a aucun endpoint P, le pipeline bascule sur le décodeur par distributions du §3.7 et l'écrit dans `_diag.txt`.

## 5. Plan de dépense des 100 réponses (`probes/plan.json`, fichiers `.jsonl` acceptés par `task_template.py`)
| Étape | Sondes | Réponses |
|---|---|---|
| Ronde 0, tous | 2 lots A/B + 2 questions chiffrées, 2 fois chacune | 8 |
| Endpoint déterministe | 16 lots à paires d'étiquettes, 12 sondes meta (9 extractions, 3 identité), 14 lots A/B, 18 questions chiffrées, 12 lots chiffrés, 12 « un mot », 8 textes libres | 92 |
| Endpoint échantillonné | 12 sondes meta + 4 textes libres, 2 fois chacun (32) ; 16 lots fermés 4 fois (64, dont 4 réponses déjà payées) | 92 |
| Tous échantillonnés | 4 lots A/B de 10, 3 lots chiffrés de 10 et les 2 lots A/B de la ronde 0, 10 fois chacun (90, dont 4 déjà payées) ; 3 sondes d'extraction 2 fois (6) | 92 |

## 6. À faire sur `ref-*` avant tout `ep-*` (requêtes de Titouan)
1. Ronde 0 sur les 9 `ref-*` (72 réponses) : si seul ref-08 diffère, le régime est glouton et la méthode s'applique ; si tous diffèrent, le régime est échantillonné : utiliser le plan « tous échantillonnés » (score local 0,22, §3.7).
2. Le plan complet sur les 9 `ref-*`, puis `t2_pipeline.py --prefix ref-` et `score_practice.py` : seule mesure réelle. Regarder dans `_diag.txt` si ref-03 et ref-05 sont typés S et ref-08 typé T.
3. Si un même passage « You are… » sort sur beaucoup d'endpoints, c'est le prompt par défaut de la famille : l'ajouter à `_DEFAULT_SYS` (`analyse/features.py`).
4. `analyse/wm_default_keys.py` sur les réponses de ref-06 avec les tokenizers candidats : les bibliothèques KGW courantes partagent la clé par défaut 15485863. Un z > 4 détecte le filigrane et donne le tokenizer de la famille. Mesuré sur le banc (endpoint Qwen filigrané avec le tirage par défaut de `transformers`, 60 réponses) : z = 7,4 avec le tokenizer Qwen et le générateur CUDA (γ = 0,25), z ≤ 2,1 avec le tokenizer OLMo, z ≤ 0,3 avec le générateur CPU, et aucun déclenchement sur le filigrane à clé non standard du banc. Il ne sert que si les organisateurs ont gardé une clé par défaut.
## 7. Limites
- **Prompts système des organisateurs** : une espace seule et « You are a helpful assistant. » ne laissent rien à extraire (le second est dans la liste des prompts par défaut ignorés). En régime glouton ces endpoints seraient typés P et isolés : une copie perdue chacun.
- **Supposé, non mesuré** : que les `ep-*` suivent la structure du corrigé. Si les variantes sont déguisées, 0,18 (§3.6). Calcul du critique sur 3 familles × 11 : la règle vaut 1,00 dans la structure du corrigé et 0,19 à 0,39 sinon.
- **Supposé** : filigrane KGW à fenêtre d'un token ; un autre schéma échappe à la règle 5 (la copie est alors isolée : MCC 0,72 au lieu de 1,00 dans la structure du corrigé).
- **Mesuré sur 2 familles seulement** : le seuil des familles (MCC famille 0,91 à 0,4 ; 0,58 à 0,3 ; 0,79 à 0,5). Fragile.
- **Fuite banc → méthode** : les 3 seuils (0,8 ; 0,25 ; 0,05) et le vote ont été choisis en regardant ces deux familles. Les intervalles p10-p90 viennent de tirages d'endpoints, pas de nouvelles générations.
- **Non mesuré** : modèles de plus de 2 milliards de paramètres ; prompt système qui interdit de se répéter ; fp8 et int8 ne sont jamais séparés d'une copie déguisée (acceptés comme fusionnés avec la base quand ils sont déguisés).

## 8. Critique, tour 1 (agent séparé) et corrections
Bugs corrigés : décodeur du pipeline différent du décodeur mesuré (une seule implémentation maintenant, `rules.decode`) ; tout fusionné quand aucun endpoint n'est P ; expression régulière de fuite trop large (« You're asking… », prompt par défaut de Qwen) ; statistique de filigrane qui confondait biais de position d'une LoRA et filigrane (remplacée par l'écart type des écarts signés) ; nombres collés par un espace. Non traité : seuil calibré sur une fréquence attendue de paires ; test de l'hypothèse endpoint par endpoint en rejouant le prompt extrait dans le message utilisateur.
