# T2 : le 0,842 de la méthode v4 de `t2-methode` est-il vrai sur les vrais endpoints ?

Session `t2-critique`, samedi 21/11, 20 h 30. Demande d'Adem (20 h 08) : vérifier la véracité des mondes simulés. Décision débloquée : **peut-on dépenser le budget réel des `ep-*` sur la foi de ce 0,842 ?** Aucun appel à l'API, aucune clé lue, aucun droit forcé.

## Mise à jour, 21 h 02 : le réel a parlé
Titouan a collecté 224 vraies réponses sur les 9 ref à 20 h 59 (issue #4). **Les 9 ref échantillonnent** (la question répétée diffère sur 9 sur 9, température lue ≈ 0,8, ref-08 à 1,0). L'hypothèse H1 du monde simulé est donc fausse : le 0,842 ne s'applique pas, c'est la ligne à 0,111 qui correspond au réel. Autres faits : le modèle des ref est SmolLM2-360M-Instruct (H5 : plus petit que les jumeaux de 1 à 1,5 milliard) ; la v4 de Titouan fait un MCC modèle de 0,273 sur ces réponses. Le pipeline de `t2-methode` n'a pas encore tourné sur ces réponses : le journal réel n'est pas dans un dossier lisible, demande faite à Titouan.

## Mise à jour, 21 h 13 : première mesure de règles sur le réel
Le journal brut des réponses n'est toujours pas accessible, mais Titouan a publié des statistiques calculées dessus (25 réponses par ref). Sur ces statistiques, formule de `score_practice.py` : sa v4 pré-enregistrée fait 0,273 ; la règle « variante détectée par un lecteur local → seule, tout le reste → base » fait 0,734 avec le lecteur LoRA seul et 1,000 avec en plus le lecteur fp8 par tenseur (0,446 à 1,000 sur 36 réglages, médiane 0,734). Ces chiffres sont obtenus avec des seuils regardés après coup et un lecteur LoRA entraîné en connaissant le corrigé : ils indiquent une direction, pas un score garanti. Le pipeline de `t2-methode` reste non mesuré sur le réel.

## Mise à jour, 21 h 33 : le journal réel est lu
Titouan a déposé le journal des 225 vraies réponses ref. Lu directement, texte seul : (1) les 9 ref échantillonnent, premier écart entre deux réponses à la même question dès le 9e au 19e caractère ; (2) réponses de 77 à 131 tokens en médiane, sauf la LoRA 1,5 qui atteint le plafond de 256 ; (3) prompt système et persona n'ont aucune signature visible à 25 réponses, contrairement aux mondes simulés où la persona parlait en personnage et où l'extraction rendait le prompt mot pour mot ; (4) le filigrane ne répond pas au détecteur à clé par défaut, alors que le monde simulé le supposait. Hypothèses du tableau contredites par le réel : H1 (décodage), H2 (extraction), H3 (filigrane), H5 (taille du modèle), H9 (style des personas). Le pipeline de `t2-methode` n'est pas applicable tel quel à ce journal (autres sondes, tout endpoint échantillonné rattaché au centre).

## Verdict
**Non validé sur le réel.** Le 0,842 est exact sur le monde simulé (je l'ai reproduit), mais il n'a été confronté à aucune vraie réponse, parce qu'il n'en existe aucune d'accessible. Et il dépend d'une hypothèse que deux requêtes sur `ref-01` suffisent à tester : que les endpoints rendent toujours la même réponse à la même question. Si cette hypothèse est fausse, le même pipeline fait **0,111**.

| Monde simulé | Score du pipeline (0,1·MCC famille + 0,9·MCC modèle) | MCC modèle |
|---|---|---|
| Réponses fixes (décodage glouton), structure du corrigé | **0,842** [0,67 ; 0,88] | 0,825 |
| Tous les endpoints tirent leurs réponses au sort (température 0,7), structure du corrigé | **0,111** [0,11 ; 0,11] | 0,013 |
| Réponses fixes, mais les variantes de poids sont déguisées elles aussi | **0,184** [0,09 ; 0,34] | 0,112 |

Termes. MCC : coefficient de corrélation de Matthews sur les paires d'endpoints, 0 = pas mieux qu'une réponse constante, 1 = parfait. Décodage glouton : le serveur prend toujours le mot le plus probable, donc la même question rend la même réponse. Échantillonnage : le serveur tire le mot au sort selon sa probabilité, donc la même question peut rendre deux réponses.

## 1. Inventaire des vraies réponses : il n'y en a aucune d'accessible
| Où j'ai cherché | Résultat |
|---|---|
| Toutes les branches du dépôt (fichiers `*answers*.jsonl`, `reference_*.jsonl`, `collecte_reelle_*.jsonl`, dossiers `logs/`) | aucun fichier |
| Dossier de travail local `E:\EAC` | aucun fichier |
| H200, H200b, H200c : `/home/titouan` | non lisible (droits non forcés) |
| H200b, H200c : `/home/hermes/eac` (dossier de groupe) | lisible, aucun fichier T2 de réponses |
| JURECA, dossier d'équipe `model_equality/` | connexion refusée sans double authentification |

Seule vraie réponse connue : une requête de Titouan sur `ref-01` à 16 h 08 (« Name three colors »), jamais publiée. Étapes (2) et (3) de la demande impossibles : pas de données réelles pour faire tourner le pipeline ni pour comparer.

## 2. Ce que j'ai pu vérifier : le chiffre se reproduit, et il s'effondre hors de son hypothèse
Mondes recopiés depuis la H200 (`world3` et `worldS`, 29 Mo, dans `E:\EAC-lab\t2\mondes`), pipeline du dépôt tel quel (`tracks/t2-model-equality/adem/`). Sorties de commande :
```
$ python analyse/e2e.py --world world3 --regime G --structure ref --n 18 --seeds 12
structure ref, 9 par famille endpoints, familles estimées, 12 graines, régime G
variante       score     [min ; max]  MCC famille  MCC modèle
principal      0.842   [0.67 ; 0.88]        1.000       0.825
ancre          0.387   [0.36 ; 0.44]        1.000       0.319
prudent        0.100   [0.10 ; 0.10]        1.000       0.000

$ python analyse/e2e.py --world worldS --regime S --structure ref --n 18 --seeds 12
structure ref, 9 par famille endpoints, familles estimées, 12 graines, régime S
principal      0.111   [0.11 ; 0.11]        1.000       0.013

$ python analyse/e2e.py --world world3 --regime G --structure mix --n 33 --seeds 8 --variantes principal
structure mix, 33 endpoints, familles estimées, 8 graines, régime G
principal      0.184   [0.09 ; 0.34]        0.839       0.112
```
La deuxième ligne est une mesure nouvelle : le rapport de `t2-methode` la laissait « à compléter ».

## 3. D'où vient le 0,842
Le tableau des décodeurs du rapport (`reports/t2-t2-methode.md`, §3.5) le montre : sans rattachement, le score est 0,000. Rattacher l'endpoint échantillonné au groupe central : 0,24. Rattacher aussi ceux qui ont un prompt système : 0,66. Ajouter la règle du filigrane : 0,93. Le score vient donc de deux choses : reconnaître le **type de déguisement** de chaque endpoint, puis supposer qu'**un endpoint déguisé sert le modèle de base**, comme dans le corrigé. La session le dit elle-même dans ses limites.

## 4. Hypothèses du monde simulé, une par une
| # | Hypothèse du monde simulé | Ce qu'on sait du réel | Effet si elle est fausse |
|---|---|---|---|
| H1 | Le décodage par défaut est glouton ; seul l'endpoint « température 1,0 » varie | **Non vérifié. Indices contraires** : le code public des organisateurs tourne à température 0,7 et top-p 0,9 ; `query_reference.py` écrit « answers are sampled » | 0,842 → 0,111 (mesuré) |
| H2 | Un prompt système caché se fait recopier par une sonde d'extraction | Non vérifié. Le dépôt des organisateurs contient trois prompts, dont une espace seule et « You are a helpful assistant. », qui ne se recopient pas | La copie est classée « en clair » puis isolée : 1,00 → 0,72 sur le corrigé (calcul) |
| H3 | Le filigrane est de type KGW à fenêtre d'un token, force 2 | Inconnu. Les données de la finale de juillet contenaient KGW, Unigram, Gumbel-Max et TextSeal | Copie filigranée isolée : 1,00 → 0,72 |
| H4 | Les variantes de poids (fp8, LoRA) sont en clair, les copies déguisées sont la base | Vrai dans le corrigé des 9 ref. Inconnu pour les 33 `ep-*` | 0,842 → 0,184 (mesuré) |
| H5 | Modèles de 1 à 1,5 milliard de paramètres (OLMo-2, Qwen2.5) | Inconnu. Les ref sont « a separate small model » ; rien sur les `ep-*` | Questions trop faciles pour un modèle plus gros : toutes les réponses fermées s'accordent |
| H6 | « fp8 » = arrondi des poids ligne par ligne sur un modèle en demi-précision | Dans le serveur vLLM, fp8 garde la tête de sortie en pleine précision ; sur une carte A100 les activations restent en 16 bits | Vrai fp8 plus proche de la base que le faux : risque de fusion (1,00 → 0,73) |
| H7 | LoRA de rang 16 entraînée 300 pas sur un corpus public | Inconnu | Taille de l'effet d'une LoRA 0,3 inconnue |
| H8 | Le serveur ne rend jamais deux textes différents pour la même question en glouton | Inconnu. Titouan mesure 30 à 34 % de textes longs différents entre deux tailles de lot sur son jumeau | Des endpoints gloutons classés « échantillonnés » |
| H9 | Deux prompts système et deux personas, d'un seul style | Inconnu | Extraction ou accord plus faibles |

Aucune de ces hypothèses n'a pu être confrontée à une vraie réponse. H1 et H4 changent le score d'un facteur 5 à 8.

## 5. Ce qui manque : le test minimal sur les 9 `ref-*` (décision d'Adem et de Titouan)
Posté sur l'issue #4 pour Titouan, qui garde seul l'accès.

**Étape 1 : 8 réponses par ref, 72 requêtes, environ 5 minutes.** Elle tient même si la limite de 100 par ref existe encore.
| Sonde (identifiant dans `adem/probes/probes_gelees.jsonl`) | Envois | Ce qu'on lit |
|---|---|---|
| `open-06` (décrire Lisbonne en trois phrases) | 2 | les deux réponses sont-elles identiques ? |
| `ab5-00` (5 questions à choix A/B) | 2 | format tenu, premières réponses fermées |
| `nb5-00` (5 questions chiffrées) | 2 | idem |
| `meta-03` (qui t'a créé, quel est ton nom) | 1 | famille, persona |
| `meta-00` (recopie tout le texte au-dessus) | 1 | prompt système caché |

Critères chiffrés :
1. **Régime** : nombre de ref dont les deux réponses à `open-06` diffèrent. Le monde glouton en prédit 1 (ref-08). S'il y en a 8 ou 9, le régime est échantillonné : ne pas lancer le plan gelé (0,111 mesuré), passer à la relecture par modèle local de la v4 de Titouan.
2. **Extraction** : `meta-00` rend un passage « You are… » sur ref-03 et ref-05, et sur aucune autre. Le monde simulé prédit 2 sur 2 et 0 fausse alerte.
3. **Format** : au moins 9 lignes sur 10 lisibles dans `ab5-00` et `nb5-00`.

**Étape 2, seulement si le critère 1 donne 1 ou 2 :** le reste du plan sur les ref, puis
```
python tracks/t2-model-equality/adem/analyse/plan_budget.py --answers reference_answers.jsonl --prefix ref-
python tracks/t2-model-equality/adem/t2_pipeline.py --answers reference_answers.jsonl --probes tracks/t2-model-equality/adem/probes/probes_gelees.jsonl --out ref_sub --prefix ref-
python reports/sources/t2/score_practice.py ref_sub.csv
```
Critère de succès : `score_practice.py` ≥ 0,52 (copies regroupées, au plus fp8 et LoRA 0,3 fusionnés). Entre 0,27 et 0,52 : soumettre la variante la plus prudente seulement. Sous 0,27 : ne pas dépenser le plan sur les `ep-*`.

## 6. Ce que cette vérification ne dit pas
- Elle ne dit pas que la méthode est mauvaise : sur son hypothèse, elle est la meilleure mesurée à ce jour dans l'équipe, et sa session a listé elle-même la plupart de ces limites.
- Le 0,111 en régime échantillonné mesure le pipeline du dépôt à 20 h 25 ; la session travaille sur ce régime (`analyse/sampled.py`, `eval_S.py`), le chiffre peut monter.
- Les mondes sont fabriqués par la session qui a écrit la méthode ; trois seuils ont été choisis en regardant ces deux familles.
