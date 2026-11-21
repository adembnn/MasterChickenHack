# T2 Model Equality : critique sévère (tour hors série, 21/11 vers 23 h 30)

Critique indépendant, Opus, lecture seule. **0 requête API, 0 soumission, aucun fichier de Titouan modifié, aucun `.env` lu.**
Décision débloquée : **comment dépenser les 75 réponses restantes par `ep-*`, et quelle méthode remplace les « lecteurs fp8 + LoRA estimée ».**
Lu : énoncé `reports/sources/Model_Equality_Testing.txt`, `practice_answers.csv`, `score_practice.py`, issue #4 (25 derniers commentaires), PR #24, #28, #30, #33 (descriptions), `reports/critique/t2-suivi.md`, `reports/t2-recherche-agentprov.md`.
Calculs rejouables : `reports/critique/t2-tour-severe-paliers.py` (MCC exact sous des structures cachées, sans requête).

Termes. **MCC** (coefficient de corrélation de Matthews) : corrélation entre « paire dite pareille » et « paire vraiment pareille », de −1 à 1, ramenée à 0 si négative. **Copie** : même poids que la base, seul le déguisement change (prompt système, persona, filigrane, décodage). **Variante** : poids différents (quantification, LoRA). **LoRA** : petite correction de rang faible ajoutée aux poids ; « force 0,7 » = correction multipliée par 0,7. **NLL** : log-vraisemblance négative d'un texte sous un modèle local (plus bas = texte plus probable). **Jumeau** : copie locale des poids, servant à simuler l'endpoint.

## 0. État réel (à corriger dans la question)
Le meilleur score T2 n'est plus 0,10 : **soumission 1882 = 0,40 public** (22 h 10 UTC ; familles + ep-05, ep-15, ep-22 isolés, signal fp8). Un envoi « 1882 + {ep-17, ep-18} regroupés » était annoncé pour 23 h 39. La critique porte sur l'écart 0,40 → 1,00.

## 1. Ce que demande la colonne `model`, et le calcul du 0,10
- **Sémantique exacte** : la colonne `model` ne nomme pas un type de modification. C'est une **étiquette de groupe** : deux endpoints ont la même étiquette si et seulement si on les dit « mêmes poids » (même checkpoint, même quantification, même LoRA, même force). `score_practice.py` et le serveur comparent le couple (famille, modèle) **paire par paire** : MCC binaire « même / différent », pas un MCC multi-classe.
- **Paires comptées** : MCC modèle sur les 3 × C(11,2) = **165 paires internes aux familles** seulement ; MCC famille sur les 528 paires. Le public ne voit que 30 % des paires (≈ 50 paires internes), affiché arrondi à 0,05 (`t2-suivi.md`).
- **Notre CSV respecte la sémantique** : copies sous une étiquette commune, variantes chacune la leur. Deux écarts de fond :
  1. **ep-05 et ep-15 (Llama) portent deux étiquettes différentes alors que tous deux sont « fp8 ».** Dans le corrigé ref il y a un seul fp8 par famille. Soit l'un des deux n'est pas fp8 (LoRA 0,3 ressemble à fp8 vu d'un lecteur fp8), soit c'est le même fp8 sous deux déguisements et il faut **une seule** étiquette. S'ils sont le même modèle : 0,40 → 0,42 en les réunissant (script, base de 7).
  2. **Le groupe {ep-17, ep-18} en UNE étiquette est faux par construction si ce sont deux forces de LoRA** : dans le corrigé ref, LoRA 0,3 / 0,7 / 1,5 = trois modèles distincts. Coût faible (−0,007) si les deux sont des variantes, mais c'est une erreur de sémantique.
- **Le 0,10 était-il « familles justes, modèle au hasard » ?** Oui. Familles justes ⇒ MCC famille = 1 ⇒ 0,1. Tout fusionner par famille ⇒ dénominateur nul ⇒ MCC modèle = 0 ⇒ **0,100 exactement**. Étiquettes au hasard ⇒ MCC modèle ≈ 0 (espérance positive minuscule à cause du `max(0, ·)`) ⇒ ≈ 0,10.
- **Erreur d'inférence commise à 22 h 48** (« T\* élevé ⇒ copie ») : 1803 isolait 5 endpoints. Parce que le MCC négatif est ramené à 0, **0,10 reste compatible avec 2 vraies variantes sur 5** (base de 6 : 9 combinaisons ; base de 7 : 0,108 avec 2). Le 0,10 ne disait donc presque rien sur ep-14, 23, 07, 28, 16. On a conclu trop vite.

## 2. Les paliers des autres équipes : quelle structure d'erreur
Hypothèse la plus simple, cohérente avec le corrigé (5 copies + 4 variantes sur 9) et avec nos deux scores : **par famille, 6 à 7 copies + 4 à 5 variantes** (fp8, LoRA 0,3 / 0,7 / 1,5, peut-être une 5ᵉ). Score sur toutes les paires (script) :

| Ce qu'on rend (dans chaque famille) | base 6 + 5 var | base 7 + 4 var |
|---|---|---|
| tout fusionné / tout isolé | 0,10 | 0,10 |
| 1 variante juste par famille | 0,36 | 0,43 |
| **1882 : 2 + 1 + 0 variantes justes** | **0,35** | **0,43** (public 0,40) |
| 2 variantes par famille | 0,50 | 0,61 |
| toutes sauf LoRA 0,3 et fp8 | 0,64 | 0,61 |
| toutes sauf LoRA 0,3 | 0,80 | 0,80 |
| toutes les variantes, mais en 1 seul groupe | 0,70 | 0,82 |
| toutes justes + 1 copie isolée à tort | 0,79 | 0,80 |
| tout juste | 1,00 | 1,00 |

- **0,70** ≈ deux types de variante ratés par famille (typiquement fp8 et LoRA 0,3, les plus proches de la base), ou toutes trouvées mais mises dans un seul groupe. Ces équipes ont donc les **LoRA 0,7 et 1,5**, les plus faciles. **Nous, c'est l'inverse : nous avons le fp8 (le difficile) et aucune LoRA (le facile).** C'est le symptôme central.
- **0,85** ≈ un seul type raté par famille (LoRA 0,3), ou une copie déguisée isolée à tort.
- **1,00** sur 50 paires publiques ⇒ aucune erreur visible : chaque variante isolée, chaque copie déguisée rattachée, **y compris fp8 et LoRA 0,3**, avec au plus 100 réponses. Ce qu'une équipe à 1,00 a forcément : (a) une statistique que la température, la persona et le filigrane ne bougent pas ; (b) un détecteur de LoRA qui marche sans connaître l'adaptateur ; (c) une règle de regroupement des copies. On ne sait pas laquelle ; trois équipes l'ont trouvée, donc le signal existe.

## 3. Les failles de la v4 et des lecteurs de Titouan
1. **Confusion température / poids, non résolue, seulement rapiécée.** Le lecteur LoRA « distillé » est appris sur le texte de l'endpoint : il apprend d'abord la température (corr(z, NLL) = −0,82 à −0,92, reconnu à 22 h 00). La correction actuelle (régression z ~ NLL, puis « relu à son T\* ») est un ajustement après coup, pas un modèle. Une température ne change pas l'**ordre** des tokens à chaque position ; une LoRA ou une quantification le change. Il faut un lecteur dont la température est un paramètre libre (voir §4).
2. **Seuils non transférables.** Seuils z > 2, 2,4, 2,8, 3 réglés sur les ref (SmolLM2-360M, 25 réponses, un autre modèle, d'autres longueurs). Un z dépend du nombre de tokens, de l'écart fp8/base propre à chaque modèle (Qwen-3B : rien trouvé, sans doute écart plus petit, pas absence) et de la température. PR #32 le montre : v4 ≈ 0 en tirage pur, v3 0,105 à 100 réponses. Le « 1,000 sur les ref » des lecteurs a été obtenu **avec le seuil calé sur les mêmes 25 réponses** (dit dans PR #33) : ce n'est pas une validation.
3. **Pas de loi nulle mesurée dans les conditions réelles.** Aucune expérience ne génère, avec le jumeau Qwen2.5-3B / 1.5B / Llama-3.2-1B, 200 faux endpoints « copie » aux mêmes prompts, mêmes longueurs, même T\*, pour lire la vraie distribution du z sous « mêmes poids ». Sans ça, « z 2,4 limite » n'a pas de p-valeur. Le jumeau existe : c'est 20 min de GPU par famille.
4. **Pas de test entre endpoints.** Tout est « endpoint contre base locale ». Aucun test à deux échantillons entre deux endpoints d'une même famille (MMD de Gao et al. : distance entre deux nuages de réponses aux mêmes prompts ; ou vraisemblance croisée). Or c'est la question posée : ep-05 = ep-15 ? ep-17 = ep-18 ? Sans ce test, on ne sait ni regrouper deux variantes identiques, ni séparer deux LoRA.
5. **Dépense du budget mal rentabilisée.** Le journal de collecte (PR #24) n'enregistre que le texte, `finish_reason` et l'heure de départ : ni durée de réponse, ni en-têtes, ni corps JSON complet. La latence par token (un adaptateur LoRA ralentit le décodage dans vLLM, une fp8 l'accélère) était gratuite sur chaque réponse payée ; elle est perdue pour les 25 premières. Le plan « préfixe + `max_tokens = 1` » paie une réponse pour un token : au p99 des positions, ≈ 0,39 nat d'écart fp8/base ; une réponse de 200 tokens à 0,021 de médiane en donne ≈ 4. **Préfixe imposé oui, `max_tokens = 1` non** sur les `ep-*`.
6. **Pas de pré-enregistrement réel sur des données neuves.** Chaque règle a été vue sur les ref avant d'être figée ; les ref ont un quota d'un million, et on n'a toujours pas de score sur ref 26-100 frais.
7. **Pilotage par le leaderboard.** 5 envois T2 en 2 h, chacun ajusté après le précédent. L'énoncé dit noir sur blanc que le public (30 % des paires, arrondi 0,05) peut diverger fortement du final. Un endpoint isolé touche ≈ 3 paires publiques : une sonde par envoi mesure surtout du bruit.

## 4. Protocole proposé (75 réponses restantes par endpoint)
Principe : **un rapport de vraisemblance généralisé par endpoint, la température étant un paramètre libre, calibré par des faux endpoints simulés, puis un regroupement par tests entre endpoints.**

**Étape A (gratuit, 30 min, ref seulement, ce soir)** : enregistrer le corps JSON complet, les en-têtes et la durée de chaque requête ; essayer `stream: true` (temps entre tokens), `logprobs`, `top_logprobs`, `n`. Requêtes invalides non décomptées, ref illimités. 100 requêtes identiques (256 tokens) sur ref-01, 02, 04, 07, 09 : si le temps par token sépare fp8 et LoRA des copies avec une AUC (aire sous la courbe ROC, 1 = séparation parfaite) ≥ 0,95, on le lit gratuitement sur les 75 réponses `ep-*` à venir. Un `logprobs` accepté changerait tout le plan (vraisemblance exacte côté serveur). Lire une fuite de nom de modèle dans un message d'erreur serait de l'exploitation de faille : à ne pas faire sans l'accord des organisateurs ; latence et `usage` sont de l'audit normal.

**Étape B (GPU, 0 requête) : le lecteur à température libre.** Pour chaque endpoint e et chaque candidat de poids m ∈ {base ; fp8 per-tensor (reproduit à l'identique, déjà fait) ; autres quantifications publiques : int8 W8A8, GPTQ-Int8/Int4 et AWQ officiels de Qwen2.5 ; base + s·Δ}, maximiser la vraisemblance des réponses de e en (T, top-p) : ℓ(e, m) = max\_T Σ log softmax(logits\_m / T). Statistique : Λ(e, m) = ℓ(e, m) − ℓ(e, base). La température est absorbée par le profil, pas par une régression.
- **Δ (direction LoRA) sans connaître l'adaptateur** : apprendre un LoRA sur les réponses de l'endpoint le plus écarté de la famille **avec un facteur de température appris en même temps** (un scalaire 1/τ sur les logits), pour que l'adaptateur n'apprenne que ce que la température n'explique pas. Valider la dose-réponse sur les ref : Δ appris sur ref-07 doit donner un s estimé ≈ 0,7/1,5 = 0,47 pour ref-04 et ≈ 0,2 pour ref-09, et s ≈ 0 pour les 5 copies.
- **Persona et prompt système** : relire chaque endpoint sous le prompt caché extrait (préfixe imposé, déjà réussi sur ep-29 et ep-02) avant de calculer Λ ; sinon une persona se fait passer pour une LoRA (cas ep-02).
- **Filigrane** : un filigrane KGW pousse des tokens choisis par hachage du token précédent ; un LoRA ne peut pas l'apprendre et Λ(base+sΔ) reste ≈ 0. Garde-fou : un endpoint à NLL élevée dont aucun candidat n'améliore ℓ est classé copie déguisée.
- **Calibration** : par famille, 200 faux endpoints simulés avec le jumeau, aux mêmes prompts, mêmes longueurs, au T\* ajusté : copies (loi nulle de Λ) et fp8 / base + sΔ (puissance). p-valeur = rang. Contrôle croisé gratuit : les ep-33, ep-30, ep-31 (T = 1,0) et les endpoints au prompt extrait doivent tomber dans la loi nulle.
- **Critère de passage pré-enregistré** : sur 3 tirages frais de 25 réponses par ref (réponses 26-100, jamais vues), `score_practice.py` ≥ 0,90 en moyenne, et 0 copie isolée. Sinon on ne dépense rien.

**Étape C (0 requête)** : appliquer B aux 25 réponses déjà payées de chaque `ep-*`. Puis **tests entre endpoints** dans chaque famille, pour toutes les paires de suspects : vraisemblance croisée (le Δ appris sur ep-i explique-t-il ep-j au même s ?) et MMD à noyau de Hamming sur les 50 premiers caractères de réponses aux mêmes prompts. Copies = groupe indiscernable (gros, ≥ 5 membres) ; deux variantes indiscernables entre elles = une seule étiquette.

**Étape D (dépense 50 / endpoint)** : 10 prompts ouverts fixes, identiques pour les 33 endpoints, × 5 réponses, réponses longues (consigne « 300 mots » + préfixe imposé qui saute la formule d'introduction), 256 tokens, durée enregistrée. Domaines variés (code, science, récit, conseil, liste), car une LoRA ne bouge que certains domaines. 4 requêtes en parallèle, journal complet. **Jamais `max_tokens = 1`.**

**Étape E (dépense ≤ 25, ciblée)** : refaire B et C sur 75 réponses. Les 25 dernières ne vont qu'aux endpoints dont la p-valeur est entre 0,01 et 0,2, ou dont l'appartenance à un groupe est incertaine (test séquentiel). Les autres gardent leur budget.

**Critère d'arrêt** : on fige le CSV quand chaque endpoint a une décision à p < 0,01 (variante) ou p > 0,2 (copie) sous la loi nulle simulée, ou n'a plus de budget ; dans le doute, **rattacher à la base** (isoler une copie coûte ≈ 2 fois ce que rapporte une variante juste : −0,10 contre +0,05, `t2-structure-posterieur.py`). Un seul CSV final, issu de la règle figée, envoyé une fois ; le public n'est qu'un contrôle grossier.

Ordre et temps : A (30 min) ∥ B (2 h GPU, 3 familles en parallèle sur les H200) → validation ref (30 min) → C (30 min) → D (≈ 40 min de collecte) → E → CSV. Dépenser D avant que B passe le critère est la seule erreur irréversible possible.

## 5. Trois erreurs de méthode à ne plus refaire
1. **Conclure d'un score public arrondi et tronqué.** 0,10 est compatible avec 2 vraies variantes sur 5 isolées ; ±0,05 à 30 % des paires ne valide ni n'invalide un endpoint. Chaque décision doit venir d'une p-valeur locale calibrée.
2. **Régler un seuil sur les données qui servent à le juger, et sur un autre modèle.** Tout seuil se fixe sur des faux endpoints simulés de la bonne famille, puis se vérifie sur des réponses ref jamais vues.
3. **Lire une différence de texte comme une différence de poids sans retirer d'abord la température et le déguisement.** Le modèle statistique doit contenir la température (et le prompt caché) comme paramètre ; un lecteur appris sur le texte de l'endpoint mesure d'abord sa chaleur.
Corollaire budgétaire : chaque réponse payée doit rapporter le maximum (256 tokens, durée, corps complet), et ne partir qu'après un test passé sur les ref.

## Limites de cette critique
Structure cachée supposée (6-7 copies + 4-5 variantes par famille), non observée. Je n'ai pas relu le code des lecteurs (`fp8_llr.py`, `ep_readers.py`) ligne à ligne, ni les journaux bruts sur les H200. La piste latence n'est pas mesurée : c'est une hypothèse à tester en 30 min, pas un fait.
