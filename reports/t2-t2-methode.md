# T2 Model Equality : méthode v4, qui tient les copies déguisées ensemble

Session `t2-methode` d'Adem, samedi 21/11. Décision débloquée : **avec quelles sondes dépenser les 100 réponses par endpoint `ep-*`, et quel décodeur soumettre**. Aucun appel à l'API, aucune clé lue, aucune branche de Titouan modifiée, rien de commité. Code : `tracks/t2-model-equality/adem/` (mode d'emploi : `README.md`). Critique séparée : `reports/critique/t2-methode-tour1.md`.

Termes. MCC : coefficient de corrélation de Matthews sur les paires d'endpoints (« même modèle » ou non), 0 = pas mieux qu'une réponse constante, 1 = parfait. AUC : probabilité qu'une paire « même modèle » ait une statistique plus haute qu'une paire « différente » (0,5 = hasard). Décodage glouton : le serveur prend toujours le token le plus probable, donc la même question rend la même réponse. Filigrane KGW : le serveur ajoute un bonus δ aux logits d'une « liste verte » de tokens tirée d'après le token précédent. LoRA : adaptateur de rang faible ajouté aux poids, multiplié par une force.

## 0. Résultat en six lignes
1. **Mesuré** : sur un monde local où chaque poids existe sous chaque déguisement, aucune statistique par paire ne sépare seule « mêmes poids » de « poids différents » (AUC 0,49 à 0,56). Le déguisement déplace plus le texte que les poids.
2. **Mesuré** : la v3 telle quelle (identité de texte, seuil 0,43 de divergence) fait MCC 0,00 sur des mondes de structure identique au corrigé.
3. **Mesuré** : le type d'un endpoint se lit dans ses réponses. Température 1,0 : deux réponses différentes au même message (48/48). Prompt système ou persona : une sonde d'extraction le rend mot pour mot (OLMo 16/16, Qwen 14/16). 0 fausse alerte sur les endpoints en clair après retrait du prompt par défaut de Qwen.
4. **Méthode v4** : comparaison exacte de la v3 seulement entre endpoints en clair, puis rattachement des endpoints déguisés au groupe central. Sans apprentissage, 3 seuils.
5. **Mesuré, 100 réponses par endpoint, structure du corrigé** : MCC modèle 0,93 (OLMo) et 0,71 (Qwen) ; bout en bout sur 18 endpoints et 2 familles : **score 0,842 [0,67 ; 0,88]**. CSV de 33 endpoints simulés : `validate_t2.py --strict` → `RESULTAT: VALIDE`.
6. **Limites mesurées** : score 0,18 si les variantes de poids sont elles aussi déguisées (33 endpoints au hasard) ; **MCC 0,02 à 0,27 si tous les endpoints échantillonnent par défaut** (§3.7). Le premier fait à établir sur `ref-*` est donc le régime de décodage : il décide si cette méthode s'applique.

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
- Sortie : texte seul, 256 tokens au plus, au format de `task_template.py`. Deux régimes : glouton par défaut (monde `world3`), échantillonné à T = 0,7 par défaut (monde `worldS`).
- Mondes notés : « ref » = structure exacte du corrigé (base en clair + 4 copies déguisées + 4 variantes en clair), 200 tirages ; « mix » = 14 endpoints au hasard parmi 40 ; « tout » = les 40.

## 3. Mesures
### 3.1 Statistiques par paire seules (OLMo, monde v1, 34 endpoints, 143 sondes, 6 tirages de budget)
```
statistique                           AUC  MCC ref     [p10 ; p90]  MCC mix
ident (v3)                          0.523    0.176   [0.00 ; 0.38]    0.021
pref (préfixe commun)               0.516    0.201   [0.00 ; 0.38]    0.033
bow (mots communs)                  0.518    0.047   [0.00 ; 0.16]    0.035
m_q1 (une question chiffrée)        0.535    0.286   [0.00 ; 0.69]    0.034
m_ab5 (5 questions A/B)             0.521    0.052   [0.00 ; 0.16]    0.030
m_nb5 (5 questions chiffrées)       0.511    0.199   [0.00 ; 0.38]    0.039
m_word (un mot)                     0.559    0.420   [0.14 ; 1.00]    0.029
a_list / a_seq (listes, suites)     0.509 / 0.497
RL toutes (régression logistique)   0.645    0.309   [0.10 ; 0.49]    0.116
```
Identité de texte contre la base en clair (monde v3, plan gelé) : OLMo réplique 0,94 ; int8 0,79 ; fp8 0,54 ; LoRA 0,3 0,44 ; nf4 0,30 ; **prompt système 0,30** ; LoRA 0,7 0,21 ; **filigrane 0,20** ; **persona 0,17** ; LoRA 1,5 0,09 ; **T 1,0 0,08**. Qwen : réplique 0,98 ; int8 0,78 ; LoRA 0,3 0,66 ; filigrane 0,65 ; fp8 0,62 ; prompt système 0,53 ; LoRA 0,7 0,53 ; persona 0,45 ; nf4 0,38 ; T 1,0 0,34 ; LoRA 1,5 0,30. Une copie déguisée de la base est donc plus loin de la base qu'une quantification.

### 3.2 Par type de paire le signal existe (OLMo, v1, AUC de l'identité de texte)
clair/clair 1,00 ; clair/prompt système 0,85 (0,99 contre LoRA ≥ 0,7 et nf4 seulement) ; clair/filigrane 0,82 (0,98) ; clair/persona 0,77 (0,89) ; clair/T 1,0 : 0,44 (0,61). Classifieurs appris sur une famille et testés sur l'autre (monde v3) : AUC 0,56 à 0,72, MCC ref 0,00 à 0,59 selon le sens. Ils transfèrent mal ; d'où le choix de règles sans apprentissage.

### 3.3 Pistes mesurées et écartées
- Neutraliser le prompt système depuis le message utilisateur (jeton de fin de texte, en-tête de nouvelle conversation, 500 tokens de remplissage, « ignore les instructions ») : identité 0,04 à 0,25 avec persona ou prompt système, contre 0,67 à 0,92 pour une simple réplique.
- Lots de 20 mini-questions numérotées : un modèle 1B recopie « B » sur 15 lignes. Lots de 5 : taux d'analyse 0,94 à 1,00, retenus.
- Profil de capacité (taux de bonnes réponses par type de calcul) : écarts de 0,05 à 0,15 entre copies déguisées, du même ordre qu'entre poids ; inutilisable à 100 réponses.
- Réponses chiffrées « au bord du savoir » : un prompt système les change (accord 0,17 à 0,33).

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
règles : P seuls, S/T/filigrane isolés              0.000  [0.00 ; 0.00] 0.026 0.091 |   0.000  [0.00 ; 0.00] 0.026 0.091
règles : + T rattaché au centre                     0.240  [0.00 ; 0.27] 0.041 0.000 |   0.264  [0.27 ; 0.27] 0.036 0.000
règles : + S rattaché                               0.658  [0.22 ; 0.72] 0.112 0.192 |   0.706  [0.72 ; 0.72] 0.088 0.020
règles complètes (+ filigrane)                      0.926  [0.45 ; 1.00] 0.124 0.190 |   0.706  [0.72 ; 0.72] 0.081 0.004
plus proche ancre, sans a priori                    0.732  [0.39 ; 1.00] 0.203 0.326 |   0.016  [0.00 ; 0.04] 0.172 0.109
```
Le groupe central est bien la base en clair dans 176/200 mondes (OLMo) et 194/200 (Qwen). 0,72 = tout juste sauf la copie filigranée, isolée.

### 3.6 Bout en bout (`analyse/e2e.py` : simulation → plan de 100 réponses → `t2_pipeline.py` → barème officiel)
```
structure ref, 9 par famille, familles estimées, 12 graines      structure mix, 33 endpoints, familles estimées, 8 graines
variante       score     [min ; max]  MCC famille  MCC modèle     variante       score     [min ; max]  MCC famille  MCC modèle
principal      0.842   [0.67 ; 0.88]        1.000       0.825     principal      0.184   [0.09 ; 0.34]        0.839       0.112
ancre          0.387   [0.36 ; 0.44]        1.000       0.319     ancre          0.194   [0.09 ; 0.36]        0.839       0.122
prudent        0.100   [0.10 ; 0.10]        1.000       0.000     prudent        0.102   [0.07 ; 0.23]        0.839       0.020
repli          0.100   [0.10 ; 0.10]        1.000       0.000     repli          0.084   [0.07 ; 0.10]        0.839       0.000
```
Validateur de Titouan sur le CSV de 33 endpoints : `33 endpoints, 2 familles, 10 modeles` / `RESULTAT: VALIDE`. Repli : `RESULTAT: VALIDE` avec l'avertissement « tout différent ».

### 3.7 Régime où tous les endpoints échantillonnent (monde `worldS`, T = 0,7 par défaut, endpoint « T 1,0 » à 1,0)
Plus aucune réponse n'est répétable, donc plus de comparaison exacte. Statistique testée : pour chaque item fermé, D = collision(A,A) + collision(B,B) − 2 collision(A,B), où la collision est la probabilité que deux réponses soient égales. D est un estimateur sans biais de la distance quadratique entre les deux distributions de réponses ; z = D moyen sur son erreur type. Même construction sur les textes libres (noyau = cosinus des sacs de mots) et sur la longueur. Plan : 17 lots fermés × 4 + 16 sondes libres × 2 = 100 réponses.
```
z max contre la base en clair (OLMo) : réplique 0.4  fp8 0.4  int8 0.5  LoRA0.3 1.9  nf4 2.9  T1,0 3.4 / 2.1  LoRA0.7 3.8
                                       filigrane 4.2 / 4.2  persona 4.6 / 5.7  système 5.0 / 3.7  LoRA1.5 5.2
z max contre la base en clair (Qwen) : réplique 0.2  fp8 0.6  T1,0 0.7 / 0.7  LoRA0.3 0.8  LoRA0.7 2.1  int8 2.2
                                       système 2.7 / 4.0  filigrane 2.8 / 3.0  LoRA1.5 3.0  nf4 3.6  persona 4.7 / 3.9
prompt système détecté correctement : olmo 39/40, qwen 40/40      endpoint T 1,0 repéré par son taux de collision : AUC 0.91 / 0.84
décodeur (MCC ref, 200 mondes)                    OLMo   [p10 ; p90]     Qwen   [p10 ; p90]
centre + z < 3 (fermées + textes + longueur)      0.021  [0.02 ; 0.02]   0.064  [0.00 ; 0.33]
centre + z < 3 (fermées seules)                   0.000  [0.00 ; 0.00]   0.165  [0.00 ; 0.33]
endpoints à prompt système fusionnés, reste isolé 0.037  [0.04 ; 0.04]   0.273  [0.27 ; 0.27]
```
**Lecture** : à 100 réponses échantillonnées, fp8, int8 et LoRA 0,3 sont dans le bruit (z < 2), les copies déguisées en sortent autant que les LoRA fortes (z 3 à 5), et le centre n'est retrouvé que dans 76/200 et 15/200 mondes. **Dans ce régime la méthode ne marche pas : MCC 0,02 à 0,27.** La détection du prompt système, elle, tient (79/80).

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

## 6. À faire sur `ref-*` avant tout `ep-*` (requêtes de Titouan)
1. Ronde 0 sur les 9 `ref-*` (72 réponses) : si seul ref-08 diffère, le régime est glouton et la méthode s'applique ; si tous diffèrent, le régime est échantillonné, la méthode ne tient pas (§3.7) et il faut d'abord chercher sur `ref-*` un plan à beaucoup plus de répétitions par sonde.
2. Le plan complet sur les 9 `ref-*`, puis `t2_pipeline.py --prefix ref-` et `score_practice.py` : seule mesure réelle. Regarder dans `_diag.txt` si ref-03 et ref-05 sont typés S et ref-08 typé T.
3. Si un même passage « You are… » sort sur beaucoup d'endpoints, c'est le prompt par défaut de la famille : l'ajouter à `_DEFAULT_SYS` (`analyse/features.py`).
4. `analyse/wm_default_keys.py` sur les réponses de ref-06 avec les tokenizers candidats : les bibliothèques KGW courantes partagent la clé par défaut 15485863. Un z > 4 détecte le filigrane et donne le tokenizer de la famille. Mesuré sur le banc (endpoint Qwen filigrané avec le tirage par défaut de `transformers`, 60 réponses) : z = 7,4 avec le tokenizer Qwen et le générateur CUDA (γ = 0,25), z ≤ 2,1 avec le tokenizer OLMo, z ≤ 0,3 avec le générateur CPU, et aucun déclenchement sur le filigrane à clé non standard du banc. Il ne sert que si les organisateurs ont gardé une clé par défaut.
La consigne (l. 109) dit que `temperature` et `seed` sont ignorés et les messages `system` refusés : inutile de les tester.

## 7. Limites
- **Supposé, non mesuré** : que les `ep-*` suivent la structure du corrigé. Si les variantes sont déguisées, 0,18 (§3.6). Calcul du critique sur 3 familles × 11 : la règle vaut 1,00 dans la structure du corrigé et 0,19 à 0,39 sinon.
- **Supposé** : filigrane KGW à fenêtre d'un token ; un autre schéma échappe à la règle 5 (la copie est alors isolée : MCC 0,72 au lieu de 1,00 dans la structure du corrigé).
- **Mesuré sur 2 familles seulement** : le seuil des familles (MCC famille 0,91 à 0,4 ; 0,58 à 0,3 ; 0,79 à 0,5). Fragile.
- **Fuite banc → méthode** : les 3 seuils (0,8 ; 0,25 ; 0,05) et le vote ont été choisis en regardant ces deux familles. Les intervalles p10-p90 viennent de tirages d'endpoints, pas de nouvelles générations.
- **Non mesuré** : modèles de plus de 2 milliards de paramètres ; prompt système qui interdit de se répéter ; fp8 et int8 ne sont jamais séparés d'une copie déguisée (acceptés comme fusionnés avec la base quand ils sont déguisés).

## 8. Critique, tour 1 (agent séparé) et corrections
Bugs corrigés : décodeur du pipeline différent du décodeur mesuré (une seule implémentation maintenant, `rules.decode`) ; tout fusionné quand aucun endpoint n'est P ; expression régulière de fuite trop large (« You're asking… », prompt par défaut de Qwen) ; statistique de filigrane qui confondait biais de position d'une LoRA et filigrane (remplacée par l'écart type des écarts signés) ; nombres collés par un espace. Non traité : seuil calibré sur une fréquence attendue de paires ; test de l'hypothèse endpoint par endpoint en rejouant le prompt extrait dans le message utilisateur.

## 9. Reproduire
```
python probes/make_probes.py --v3 --n_fs 0 --out probes/probes_v3.jsonl ; python probes/geler.py probes/probes_v3.jsonl probes/probes_meta2.jsonl
python bench/world.py --model allenai/OLMo-2-0425-1B-Instruct --fam olmo --regime G --probes probes/probes_v3.jsonl --lora lora_olmo --out world3/olmo_G_base.jsonl
python analyse/eval_rules.py --probes probes/probes_gelees.jsonl --plan probes/plan.json --world world3 --fams olmo,qwen
python analyse/e2e.py --world world3 --structure ref --seeds 12
```
