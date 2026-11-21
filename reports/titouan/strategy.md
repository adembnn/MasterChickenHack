# Stratégie transversale pour gagner le classement combiné (T1 + T2 + T3)

Date : sam. 21/11/2026 ~16:00. Ce que ce rapport débloque : comment répartir 3 personnes, les GPU et 20 h entre les trois tâches, quand soumettre, quand abandonner une piste, quoi capturer pour le pitch.
Légende : [SOURCE] lu dans le dépôt ou le web ; [SIM] sortie de ma simulation (hypothèses en 2.4, scripts sim.py/sim2.py dans le dossier de travail) ; [HYP] jugement non mesuré. Aucun endpoint appelé, rien téléchargé.

## 0. Verdict en 12 lignes

1. L'agrégation exacte n'est écrite nulle part dans le dépôt, les slides ni les 3 PDF [SOURCE : grep rank/combin/aggreg sur slides.txt et les .txt : rien]. Seule source externe : la page de presse de la finale 2026 du même championnat dit que le classement final repose sur « le score combiné des trois tâches » ET sur la présentation devant le jury (https://eng.pw.edu.pl/News/WUT-representatives-on-the-podium-at-the-CISPA-hackathon ; résumé via recherche web, non vérifié sur la page officielle). Le pitch peut donc compter, à confirmer au bureau d'accueil.
2. Quelle que soit l'agrégation plausible, la bonne stratégie est la même : **équilibre d'abord, pic ensuite**. Un zéro sur une tâche (porte T3 < 70 %, CSV invalide, .pt rejeté) coûte 8 à 14 places au classement combiné [SIM], plus que n'importe quel gain de précision sur les deux autres.
3. Ordre d'effort : (a) soumission valide et non nulle sur les 3 tâches avant 19:00 ; (b) la tâche où l'écart de score par heure de travail est le plus fort d'abord (T1 : méthode B ; T3 : plan A), T2 reste limité par le budget de requêtes, pas par les heures ; (c) seulement après, creuser la tâche où l'on est le mieux placé.
4. Public 30 % / privé 70 % : le bruit de classement est modeste sur T1 et T3 (± 2-3 places au milieu du classement), élevé sur T2 (± 5-7 places, davantage si peu de paires « même modèle »). Règle : on choisit la soumission finale sur la validation locale, le public sert de garde-fou.
5. **Point à vérifier d'urgence** : le classement affiche « le meilleur résultat par équipe » et ne met pas à jour si la nouvelle soumission est plus basse [SOURCE : Watermark_Localization.txt l.162-165, Unlearning.txt l.198-200, Model_Equality_Testing.txt l.191]. Il est probable que le classement privé utilise la soumission au meilleur score PUBLIC (ou la dernière). Demander aux organisateurs (Discord) laquelle compte. Si c'est « meilleure au public », chaque soumission « de plus » est un tirage de sélection sur du bruit : on soumet peu et seulement des candidats déjà validés en local.
6. Scores réalistes (section 4) : T1 médian 0,65 (rang ~14), T2 médian 0,45-0,50 (rang ~12), T3 médian 0,35 (rang ~14). Probabilité d'être dans les 6 premiers au combiné : ~15-30 %, dans les 3 : ~5-10 % [SIM + HYP]. Très incertain : on ne connaît aucune équipe adverse.
7. Le pitch : 6 équipes seulement ; il faut donc d'abord être dans les 6, mais la préparation du pitch se fait en continu (captures toutes les 2-3 h, section 7).

## 1. Hypothèses d'agrégation et ce que chacune implique

| # | Hypothèse | Plausibilité [HYP] | Conséquence sur l'effort |
|---|---|---|---|
| H1 | Somme (ou moyenne) des rangs par tâche, le plus petit gagne | 40 % | Un rang vaut autant sur chaque tâche. Investir là où 1 point de score fait gagner le plus de rangs (champ dense). Éviter les queues de classement (zéro = rang ~40-50). |
| H2 | Scores normalisés (min-max, z-score, ou score / meilleur score) puis moyenne | 25 % | Compte la distance au meilleur, pas le rang. Tâche à score borné et serré (T1 mIoU) : peu de gain relatif. T2 et T3 (étalement fort) : un score nettement au-dessus de la médiane pèse plus. Un zéro vaut -1 écart-type environ. |
| H3 | Rang du rang / produit des rangs / moyenne géométrique | 10 % | Récompense un seul rang 1-3 (produit : 1×20×20 = 400 contre 10×10×10 = 1000). Les pics comptent plus que les creux, mais un rang ~50 reste catastrophique. |
| H4 | Moyenne pondérée (poids égaux ou par difficulté) | 10 % | Comme H1/H2 ; on ne connaît pas les poids : ne rien parier dessus. |
| H5 | Rang par tâche avec départage (ex. meilleur pire rang), ou « top-k par tâche » | 10 % | Le pire rang compte : équilibre. |
| H6 | Rang combiné + note du jury pour les 6 finalistes | 5 % ajout | Le classement du jury après les pitchs (13:00-14:30) change l'ordre des 6 ; préparer le pitch dès 22:00. |

CLAUDE.md dit « le classement final combine nos rangs sur les trois tâches » : cela favorise H1/H3/H5 par rapport à H2, mais c'est l'interprétation d'Adem, pas un texte des organisateurs.

### 1.1 Résultats de simulation (49 adversaires + nous ; 2500 tirages ; bruit public-privé inclus)
Champs [SIM, hypothèses en 2.4] : T1 ~ N(0,55 ; 0,10) ; T2 80 % N(0,25 ; 0,15) + 20 % N(0,60 ; 0,10) ; T3 25 % de zéros (porte ou abandon) + N(0,30 ; 0,14). ρ = corrélation de force latente entre les tâches (une bonne équipe est bonne partout). Colonnes : rang combiné moyen / P(top 6) / P(top 3).

| Profil (rangs par tâche T1/T2/T3) | Somme, ρ=0 | Somme, ρ=0,4 | Produit, ρ=0 | Produit, ρ=0,4 |
|---|---|---|---|---|
| 5/5/5 | 1,3 / 1,00 / 0,99 | 2,4 / 0,99 / 0,82 | 2,2 / 0,99 / 0,86 | 3,7 / 0,92 / 0,51 |
| 10/10/10 | 3,1 / 0,95 / 0,66 | 5,9 / 0,64 / 0,16 | 7,4 / 0,44 / 0,11 | 9,3 / 0,18 / 0,02 |
| 15/15/15 | 7,6 / 0,43 / 0,11 | 11,4 / 0,08 / 0 | 15,4 / 0,02 / 0 | 15,9 / 0,01 / 0 |
| pics 1/20/20 | 5,9 / 0,66 / 0,23 | 9,6 / 0,17 / 0,02 | 4,7 / 0,79 / 0,36 | 7,0 / 0,48 / 0,06 |
| pics 3/12/25 | 5,3 / 0,72 / 0,27 | 8,9 / 0,20 / 0,02 | 7,0 / 0,49 / 0,13 | 9,1 / 0,19 / 0,02 |
| T3 = 0 ; 5/5 ailleurs | 10,9 / 0,12 / 0,01 | 14,5 / 0,01 / 0 | 8,2 / 0,36 / 0,11 | 9,9 / 0,17 / 0,02 |

Lecture :
- Sous la somme, un profil 10/10/10 bat un profil 1/20/20 (3,1 contre 5,9 ; 5,9 contre 9,6 avec corrélation). Sous le produit, c'est l'inverse (4,7 contre 7,4). Les deux conventions s'accordent sur un point : **un T3 à zéro détruit un 5/5** (rang 11-14 au lieu de 1-2).
- Le calcul suppose des rangs adverses indépendants entre tâches (ρ=0) ou modérément corrélés (0,4). Dans la réalité, ρ est probablement 0,2-0,5 (les meilleures équipes ML sont fortes partout) : lire la colonne ρ=0,4 comme la plus prudente. Les équipes vraiment fortes seront moins fragiles que le modèle ne le suppose.
- Ne pas lire ces nombres comme des probabilités de victoire : le champ (loi des scores adverses) est inventé.

### 1.2 Règle d'allocation qui en découle (robuste aux six hypothèses)
1. **Plancher** : viser sur chaque tâche au moins le niveau « baseline solide » (rang ≤ 25) avant de chercher un pic. Valeur marginale maximale (zéro → baseline = +8 à +14 places).
2. **Rendements décroissants** (par tâche, [HYP]) : la 1re heure donne un score valide ; les 4-6 premières heures donnent l'essentiel de la méthode ; chaque heure supplémentaire donne +0,5 à 1 point de mIoU sur T1, quelques % sur T3, rien de sûr sur T2 (limité par le budget irréversible). Quand le gain horaire estimé passe sous ~0,3 point de score ou 1 rang par 2 h, basculer l'heure vers la tâche voisine.
3. **Pic sur la tâche où l'on a le plus d'avance relative** une fois le plancher atteint : probablement T1 (kit complet, validation locale sur 510 images, données synthétiques illimitées, effet de l'effort le plus continu) ou T3 si le pseudo-gold fonctionne. T2 est un coup de dés plafonné.
4. **Pas de poursuite d'un rang 1 improbable** : un rang 1 n'a de valeur privilégiée que sous H3 (produit), hypothèse à ~10 %.

## 2. Public contre privé : modèle d'incertitude

### 2.1 T1 (mIoU, 1020 images test, public 30 % ≈ 306, privé ≈ 714)
- Écart-type du mIoU par image : 0,20-0,30 (distribution bimodale : images presque parfaites et images presque ratées) [HYP]. Erreur-type du public : 0,25/√306 = ±0,014 ; du privé : ±0,009 ; écart public-privé d'une même équipe : sd ≈ 0,017.
- Mais ce bruit est **partagé** entre équipes (mêmes images) : il déplace tout le monde de la même façon. Le bruit qui reclasse est celui de la différence entre deux méthodes sur les mêmes images (sd par image de la différence ≈ 0,10 → ±0,006 à 0,010). Deux équipes à moins de ~0,02 de mIoU sont indiscernables.
- Simulation avec bruit indépendant (borne haute) : un rang médian bouge en moyenne de ±3 places, la frontière du top 6 de ±1,8 [SIM].
- Équilibre par méthode : le split 30/70 est un hachage de l'identifiant ; le public contient ~102 images par watermark ± 8. Si une méthode (p. ex. StegaStamp) est notre point faible, un public à 110 images StegaStamp sous-estime le privé de ~0,005-0,01. À corriger : toujours rapporter le score local **par méthode**.
- Éditions rares (outpainting extrême, masque vide) : quelques images à IoU 0,5 → 1 de différence font bouger le total de 0,002 chacune. Ignorer.

### 2.2 T2 (MCC sur 528 paires, public = « une partie des paires »)
- Taille du public non indiquée. Si ~30 % des paires (≈ 160) : le nombre de paires « même modèle » est petit (en supposant 33 endpoints répartis en ~12-15 modèles de 1 à 5 endpoints : ~25-40 paires positives sur 528, donc ~8-12 dans le public) [HYP]. Un MCC sur ~10 positives a un écart-type de 0,15-0,25.
- Les erreurs sont structurelles : mal classer UN endpoint touche 32 paires d'un coup. Le public et le privé partagent les mêmes endpoints : un endpoint mal placé coûte aux deux. Ce n'est pas un bruit d'échantillon indépendant mais une erreur de décision ; le public est donc un test **assez fidèle** pour les erreurs de regroupement, mais peu fidèle si le sous-ensemble de paires du public est biaisé vers des paires faciles.
- Simulation (bruit de score sd 0,06 à 0,10) : rang médian ±5-7 places, top 6 ±2,6-3,9 [SIM]. La plus grande incertitude du portefeuille.
- Mise en garde des organisateurs, en majuscules : « DO NOT RELY ON THE PUBLIC SCORE » ; le public et le final « peuvent différer beaucoup » [SOURCE : Model_Equality_Testing.txt l.177-182].

### 2.3 T3 (forget × retain, public 30 % des données de scoring cachées)
- Tailles des jeux cachés (copies modifiées des 3500 images, jeu de contrôle, jeu de validation) non données. Probablement 1000-5000 images chacun : bruit modéré (sd ≈ 0,02-0,04 sur le produit) → ±3 places au milieu [SIM].
- Hachage déterministe, une copie suit son image source [SOURCE : Unlearning.txt l.186-187] : pas de fuite entre public et privé, donc surajustement possible sur le public sans que le privé suive.
- Risque dominant : **la porte à 70 %** sur un jeu caché, sans labels pour nous. Si le public affiche un score > 0, la porte est passée sur la part publique ; une marge de sécurité de 3-5 points reste nécessaire pour le privé.
- Le score est relatif à orig : un réglage « à la limite » (forget élevé, retain proche du seuil) peut sembler bon au public et tomber à 0 au privé si la précision privée est 1-2 points plus basse. Choisir le réglage le plus robuste, pas le plus haut.

### 2.4 Hypothèses de la simulation (à garder en tête)
49 adversaires ; distributions de scores inventées (section 1.1) ; bruit privé-public gaussien indépendant entre équipes (surestime le bruit de rang sur T1/T3) ; corrélation entre tâches ρ ∈ {0 ; 0,4}. Aucune donnée réelle d'équipes. Les fichiers sim.py et sim2.py sont dans le dossier de travail pour rejouer avec d'autres paramètres.

## 3. Stratégie de soumission

### 3.1 Contraintes connues et inconnues
- T2 : 1 soumission / 30 min, même après erreur (≈ 40 possibles d'ici 12:00) [SOURCE]. Pas de limite écrite pour T1 et T3 : à vérifier (une soumission de 20 Mo ou de 350-450 Mo peut être lente ; T3 : state_dict float32 ≈ 344 Mo pour ViT-B/16 à tête 100 classes, limite 450 Mo).
- Le classement montre le meilleur par équipe ; la soumission « finale » n'est pas décrite. À demander : « le classement privé utilise-t-il la meilleure soumission publique, la dernière, ou une sélection ? ». En attendant, **supposer le pire** (meilleur public), donc ne jamais soumettre de candidat dont on ne voudrait pas comme version finale.

### 3.2 Combien et quand
| Tâche | Soumissions visées | Calendrier | Règle |
|---|---|---|---|
| T1 | 5-8 | S0 (valide, ~18:30) ; S1 méthode B (~22:00) ; S2 modèle appris (~03:00) ; S3 amélioration (~07:00) ; S4 candidat final vérifié (≤ 10:30) ; 1-2 de réserve | Soumettre seulement si le CV local bat le meilleur précédent de ≥ 0,01 (≈ 1,5 erreur-type paire). |
| T2 | 8-12 | S0 « tout différent » ou famille seule (~18:30) ; S1 après premier clustering ; puis une par changement d'hypothèse majeur ; dernières à 09:00 et 11:15 | Utiliser 30 min d'intervalle comme budget de **tests de structure** (p. ex. fusionner/séparer un groupe) mais jamais comme gradient fin : chaque soumission révèle un score public sur ~160 paires, donc peu d'information. |
| T3 | 4-6 | S0 filet rang 4 (~19:00, score non nul) ; S1 plan A (~23:00) ; S2 plan B/pseudo-gold (~04:00) ; S3 candidat final (≤ 10:00) | Ne pas balayer d'hyperparamètres sur le public. Réglage fixé par le mini-scoreur local + critère de précision proxy. |

### 3.3 Choix de la soumission finale (sans surajuster)
1. Premier critère : validation **locale** sur données jamais utilisées pour régler (T1 : validation croisée 5 plis du train, stratifiée par méthode ; T2 : refs ref-01..09 ; T3 : mini-scoreur + proxy).
2. Deuxième : le public sert (a) à détecter un bug de format ou de pipeline (score public ≪ score local de plus de 0,05 en T1 → alarme) et (b) de test de cohérence (classement des candidats : doit être le même que le local, à ±0,01).
3. Si deux candidats sont à moins de 1 erreur-type, prendre **le plus simple et le plus robuste** (moins d'hyperparamètres, plus de marge sur la porte T3, moins de dépendance à un seuil).
4. Règle anti-sur-ajustement : au plus 1 changement de seuil motivé par le public ; tout le reste fixé en amont.
5. Fiabilité du pipeline (le moment de la soumission est l'évaluation) : un script `validate_submission.py` par tâche, exécuté automatiquement avant chaque envoi : T1 : clés `image_names` + `masks`, forme [1020,128,128], dtype, valeurs 0/1, noms uniques et identiques à la liste du test, taille < 20 Mo ; T2 : 33 lignes, colonnes `endpoint,family,model`, tous les ep-01..33 présents, famille cohérente (même model ⇒ même family) ; T3 : `load_state_dict(strict=True)` OK, tous les tenseurs float32, taille < 450 Mo, accuracy proxy ≥ orig − 3 points, accord argmax ≥ 95 %. Un humain différent de l'auteur lance le script et la soumission (relecture par un second regard, y compris pour la dernière).
6. **Soumissions de dernière minute interdites après 11:15** (marge de 45 min pour un échec et la file d'attente du serveur). Avant 11:15, la version finale doit déjà avoir été acceptée par le serveur au moins une fois ; la « dernière heure » ne sert qu'à un éventuel candidat amélioré déjà prévalidé.

## 4. Scores atteignables et classement probable (~50 équipes)
Aucune mesure : fourchettes tirées des trois rapports de l'équipe et de mon jugement. Les classements supposent que les adversaires sont des équipes du niveau d'un hackathon ML européen de ce type [HYP] ; un champ plus fort décale tout de 3 à 8 rangs.

| Tâche | Bas (15 %) | Médian | Haut (15 %) | Rang bas / médian / haut | Comment lire |
|---|---|---|---|---|---|
| T1 mIoU | 0,45 (constante ou fenêtre glissante mal réglée) | 0,65 (B + post-traitement rectangle) | 0,80 (C appris, par méthode) | ~32 / ~14 / ~4 | Intervalle 80 % : 0,52-0,76. Constante tout-0 ≈ (1-f)/2 : à mesurer en 5 min. |
| T2 score | 0,20 (famille juste, modèles mal séparés) | 0,45-0,50 (MCC modèle ~0,4-0,5, famille ~0,9) | 0,75 | ~22 / ~12 / ~3 | Intervalle 80 % : 0,25-0,70. Dépend du nombre d'endpoints « variantes » indiscernables (LoRA 0,3, fp8). Rang très sensible. |
| T3 score | 0 (porte ratée) | 0,35 | 0,55 | ~40 / ~14 / ~5 | forget ∈ [0,1], retain ≈ 0,9 : 0,35 suppose forget ≈ 0,4. Probabilité de zéro si on avance sans scoreur local : 10-15 % ; avec protocole de la section 6 de t3-*.md : 3-5 %. |

- Classement combiné attendu (profil médian 14/12/14) : rang ~8-12 sous la somme, ~12-15 sous le produit [SIM, ρ=0,4]. Si le profil haut se réalise sur deux tâches sur trois : top 6 plausible.
- P(top 6) ≈ 15-30 %, P(top 3) ≈ 5-10 % [SIM + HYP]. Incertitude principale : le niveau des adversaires, qui reste invisible jusqu'au classement public (disponible à tout moment sur http://34.147.229.55/leaderboard_page) : **consulter le classement public après chaque soumission** pour estimer la densité du champ (≥ 3 points d'écart entre rangs voisins : champ dispersé ; < 0,01 : champ dense, le rang dépend du bruit).
- Rendements marginaux par tâche (estimés en rangs par heure de travail, avec champ supposé) : T1 : 1 point de mIoU ≈ 2-3 rangs au milieu (champ dense) ; T3 : 0,01 de score ≈ 1-1,5 rang ; T2 : très variable, un seul endpoint bien classé = 0,02-0,04 de MCC modèle = ~1 rang.

## 5. Répartition de l'équipe, GPU et agents

Hypothèse de départ (à ajuster selon les compétences et les préférences ; aucune info personnelle fiable dans equipe/*.md qui sont vides) :
- **Titouan** (chef Jülich destors1) : ops (nœud GPU, jobs, stockage partagé, validate_submission, planning) + **T3**. Raison : T3 est rapide à faire tourner (20 s à 1 min par époque), donc compatible avec un rôle de coordination.
- **Adem** (banni1) : **T1**, la tâche la plus lourde (méthode B puis C, génération synthétique, 12-16 h).
- **Yassir** (boudda1) : **T2** (prompts, refs, clustering, budget) jusqu'à ~00:00, puis renfort sur T1 ou T3 selon la situation (T2 coûte peu en GPU et en heures une fois le budget dépensé).
- Chaque humain pilote 1 agent principal + sous-agents de relecture ; un « agent QA » (n'importe qui) exécute validate_submission.py et vérifie que le fichier annoncé = le fichier envoyé (hash).

GPU : 1 nœud = 4 A100 par job ; 40 nœuds partagés entre toutes les équipes, donc file d'attente possible. Plan : 1 job long (tmux/salloc) avec `CUDA_VISIBLE_DEVICES` découpé : T1 = 2 GPU (génération synthétique + entraînement), T3 = 1 GPU, T2/scratch local = 1 GPU. Ne pas bloquer plus d'un nœud sans nécessité. Soumettre un premier `sbatch` court avant 17:00 pour mesurer la file. Si la file dépasse 45 min : faire les essais légers sur le nœud de login (RTX8000, 2 GPU) en attendant [SOURCE : slides, 12 login nodes avec 2×Quadro RTX8000].

## 6. Critères d'arrêt (« si pas X à l'heure H, on bascule »)

| Heure | Test | Si échec | Action |
|---|---|---|---|
| 18:30 | Les 3 pipelines de soumission ont produit une soumission valide acceptée par le serveur | Un pipeline reste invalide | Personne ne code de méthode avant ; tout le monde aide à débloquer (c'est le risque n°1). |
| 20:30 | T1 : B (fenêtre glissante + bits connus) bat la meilleure constante de ≥ 0,10 en CV sur le train | < 0,05 de gain | Basculer directement vers C (segmentation apprise sur cartes du décodeur) ; abandonner D. |
| 20:30 | T3 : mini-scoreur opérationnel, soumission filet acceptée, score public > 0 | Score 0 ou porte | Revoir en priorité (la porte) ; arrêter plan B jusqu'à ce que ce soit réglé. |
| 21:00 | T2 : sur les ref, une statistique sépare les 5 « S » (mêmes modèles déguisés) de T1..T4 avec un seuil unique | Aucun test ne sépare | Réduire l'objectif : famille (10 %) + regroupement grossier ; ne dépenser que ~40 réponses par endpoint ; garder 60 en réserve. |
| 23:00 | Au moins 2 tâches sur 3 avec un score public > baseline (T1 > 0,55 ; T3 > 0,15 ; T2 > 0,25) | Deux tâches sous baseline | Reporter les effectifs vers la tâche la plus basse ; geler la plus haute. |
| 02:00 | T1 : C bat B de ≥ 0,03 sur la validation locale | Pas de gain | Geler B + post-traitement ; réaffecter Adem au pitch et à T2/T3. |
| 03:00 | T3 : plan A atteint score public ≥ 0,25 avec marge de porte ≥ 3 points | < 0,15 | Passer au plan B (pseudo-gold) ; si < 0,10 à 06:00, geler le filet rang 4 et passer en vérification seulement. |
| 06:00 | Chaque tâche a un candidat final identifié (même imparfait) | Candidat manquant | Pas de nouvelle idée après 06:00 sauf correction d'un défaut identifié. |
| 08:30 | Gel des méthodes : plus d'entraînement long | – | À partir de là : réglage de seuil (T1), choix de la version (T3), derniers prompts (T2), pitch. |
| 10:30 | Dernières soumissions en cours | Échec de format | Rejouer la dernière version valide déjà acceptée. |
| 11:15 | Aucune nouvelle soumission | – | Archivage : poids, CSV, .pt, logs (hors dépôt Git, sur scratch). |

## 7. Préparation du pitch (6 min, 13:00, 6 meilleures équipes)

Le jury (équipe technique de CISPA) regarde ce qui est généralisable et compris, pas seulement le score ; et peut compter dans le classement (H6, source web non confirmée).

Structure (6 min = 4 blocs de ~1 min 20 + 40 s d'ouverture et 40 s de clôture) :
1. **0:00-0:40** : une phrase par tâche + rang public + message clé : « une pipeline, trois tâches, validée contre le surajustement du classement public ».
2. **0:40-2:00 T1** : schéma « décodeur connu → carte locale de bits corrects → masque rectangle » ; image avant/après avec masque vérité terrain contre masque prédit ; courbe mIoU par méthode.
3. **2:00-3:20 T3** : schéma « orig → cible non-membre → distillation → gold » ; courbe oubli/rétention selon l'intensité ; point de la porte 70 %.
4. **3:20-4:40 T2** : schéma des deux étages (famille puis variantes), matrice de distance des 33 endpoints avec dendrogramme, budget de 100 réponses dépensé en tranches.
5. **4:40-5:30** : ce qui généralise : validation locale, soumissions limitées, public contre privé (montrer le score public et notre estimation de l'écart).
6. **5:30-6:00** : ce qui échoue (une limite honnête par tâche) et ce qu'on ferait avec 24 h de plus.
Démonstration : une seule, courte (≤ 45 s) : charger une image de test, afficher le masque T1 superposé ; sinon des captures pré-générées (le Wi-Fi ou JURECA peut tomber).

À capturer pendant la nuit (dossier partagé `pitch/` avec un fichier par capture, nommé par heure) :
- T1 : tableau mIoU par méthode (PixelSeal/TrustMark/StegaStamp) et par type d'édition si connu ; 6-8 triplets (image, masque vérité terrain, masque prédit) dont 2 réussites, 2 échecs et 2 cas limites (outpainting, transplantation) ; carte de bit accuracy locale ; ablation B contre C ; courbe de seuil.
- T3 : courbe oubli/rétention contre intensité ; histogrammes de perte sur images à oublier (orig, sub, cible) pour illustrer « imiter gold » ; accord argmax avec orig ; tableau des plans (rang 4, A, B) avec score public.
- T2 : matrice de distances des 33 endpoints ; résultats sur les ref (quelle statistique sépare S de T1..T4) ; courbe MCC contre nombre de réponses (sous-échantillonnage des ref) ; tableau du budget dépensé par endpoint.
- Transversal : chronologie des soumissions et des scores publics (graphique), journal des décisions « on bascule » de la section 6, estimation de l'écart public-privé, schéma d'architecture (jobs, pipelines de validation).
- Capturer **toutes les 2-3 h** pendant les points de synchronisation (la capture finale à 11:30 ne suffit pas).

## 8. Calendrier heure par heure (samedi 16:00 → dimanche 12:00)

Repères organisateurs [SOURCE : slides] : dîner livré 17:00, petit-déjeuner 08:00, code fermé 12:00, pitch 13:00. Sommeil par roulement : chacun 3,5-4 h entre minuit et 08:00, jamais deux dormeurs sur la même tâche critique ; au moins 2 personnes éveillées à tout moment jusqu'à 06:00.

| Heure | Titouan (T3 + ops) | Adem (T1) | Yassir (T2) | Points de sync / décisions |
|---|---|---|---|---|
| 16:00-17:00 | Connexion JURECA, job GPU de test, dépôt partagé, validate_submission.py (3 versions) | Kit T1, lire train, mesurer fraction f et mIoU des constantes | Lire énoncé T2 ; interroger les 9 ref (gratuit) | 16:30 : poser sur le bus #2 les rôles et le plan GPU |
| 17:00-18:00 | Mini-scoreur T3 ; chargement de orig | Soumission S0 T1 (constante) ; décodeurs sur 10 images | 60 prompts candidats ; ref x 10 répétitions | |
| 18:00-19:00 | Filet T3 (recalibrage seul), soumission S0 | Fenêtre glissante PixelSeal/TrustMark | Statistique de séparation S contre T sur ref | **18:30 sync** : 3 soumissions valides ? Sinon tout le monde débloque. |
| 19:00-20:00 | Plan A T3 (distillation + ancrage), balayage local | Fenêtre StegaStamp, seuil + post-traitement rectangle | Choisir 12-20 prompts gagnants ; liste relue à deux (Titouan) | |
| 20:00-21:00 | Plan A suite | Évaluation CV par méthode ; décision B ou C | **Tranche 1 de requêtes ep-** : 20 réponses par endpoint (famille + 1ers clusters) | **20:30 critères** T1 et T3 ; 21:00 critère T2 |
| 21:00-22:00 | Soumission S1 T3 (plan A) | Soumission S1 T1 (B) ; lancement génération synthétique (1-2 GPU) | Clustering famille, soumission T2 S1 | **22:00 sync** (15 min) : états, rebasculements, pitch : désigner le présentateur |
| 22:00-23:00 | Plan B (pseudo-gold) lancé sur 1-2 GPU | Entraînement C (premier essai) | Tranche 2 : 30 réponses supplémentaires ciblées | **23:00 critère** « 2 tâches > baseline » |
| 23:00-00:00 | Plan B suite ; ops (vérifier file GPU) | Entraînement C suite | Analyse, soumission T2 S2 | |
| 00:00-01:00 | **Dort** (00:00-04:00) | C : évaluation ; réglage de l'architecture | Réponses, clustering fusion ; tranche 3 (20 réponses) | **01:00 sync** : Adem et Yassir seuls ; messages sur le bus |
| 01:00-02:00 | (dort) | **Dort 02:00-06:00** à partir de 02:00 (après critère 02:00) | Pipeline T2 final, soumission S3 | **02:00 critère T1** |
| 02:00-03:00 | (dort) | (dort) | Yassir tient seul T1/T2 en surveillance : jobs en cours, pas de nouvelle expérience | Un seul humain éveillé : pas de changement de pipeline |
| 03:00-04:00 | (dort jusqu'à 04:00) | (dort) | **Yassir dort 03:30-07:30** après relais | **03:00 critère T3** (par Yassir en lecture seule) |
| 04:00-05:00 | Reprise T3 : plan B, soumission S2 T3 | (dort) | (dort) | **04:00 sync** Titouan : lire les journaux et issues |
| 05:00-06:00 | T3 : réglage de robustesse (marge de porte) | (dort) | (dort) | |
| 06:00-07:00 | Soumission S3 T3 ; captures pitch T3 | Reprise : C, soumission S2 T1 | (dort jusqu'à 07:30) | **06:00 critère** : un candidat final par tâche |
| 07:00-08:00 | Vérification T3 : strict=True, float32, taille | T1 : seuils, post-traitement | Reprise 07:30 : dernières tranches T2 (réserve de ~30 réponses sur endpoints ambigus) | **07:30 sync** complète (tous éveillés) |
| 08:00-09:00 | Petit-déjeuner ; captures pitch | T1 candidat final, S3 | T2 S4 | **08:30 gel des méthodes** |
| 09:00-10:00 | Rédaction pitch (structure section 7) ; T3 candidat final | Captures T1 (triplets) | Captures T2 ; T2 S5 | |
| 10:00-11:00 | Soumission finale T3 (acceptée), relecture par Adem | Soumission finale T1, relecture par Yassir | Soumission finale T2, relecture par Titouan | **10:30** : 3 versions finales acceptées par le serveur |
| 11:00-11:15 | Hash des fichiers, archivage hors Git | | | **11:15 fin des soumissions** |
| 11:15-12:00 | Répétition du pitch (2 passages chronométrés) ; vérifier que les captures s'ouvrent hors ligne | | | 12:00 : code fermé |

(Sommeil : Titouan 00:00-04:00, Adem 02:00-06:00, Yassir 03:30-07:30. Si un humain préfère dormir à un autre moment, conserver la règle « le dormeur a passé la main avec un état écrit sur le bus avant de dormir ».)

## 9. Tableau de décision

| Question | Choix par défaut | Quand changer |
|---|---|---|
| Quelle agrégation supposer ? | H1 (somme des rangs) + garde-fou H3 : plancher partout, puis pic | Si les organisateurs précisent. |
| Quelle tâche creuser après le plancher ? | T1 (rendement continu) si B ≥ 0,60 à 20:30 ; sinon T3 | Si T3 > 0,35 au public avant T1 > 0,60. |
| Donner un renfort humain à T2 ? | Non (limité par budget irréversible) ; Yassir passe en renfort après 00:00 | Si les ref ne séparent pas les « S » : réduire T2 et libérer Yassir tôt. |
| Combien de requêtes ep- à dépenser avant 00:00 ? | ≤ 50 par endpoint | Si les critères T2 21:00 échouent : seulement 20. |
| Soumissions publiques par tâche ? | 4-8 selon section 3.2 | Si le « meilleur public » est le critère final : réduire à 4. |
| Soumission finale | Meilleure validation locale parmi les candidats acceptés par le serveur | Jamais le meilleur public si l'écart local < 1 erreur-type. |
| Qui présente ? | Celui qui a le plus de contexte sur la tâche la plus forte ; 2 voix (1 pour T1+T3, 1 pour T2 + conclusions) | Désigner à 22:00. |
| Dormir ? | Oui, 3,5-4 h chacun, jamais simultanément | Seulement sauter le sommeil de la personne qui a un job critique à surveiller, pas les trois. |
| Si un pipeline de soumission échoue à 10:30 | Re-soumettre la dernière version acceptée | Ne jamais tester un nouveau format dans la dernière heure. |

## 10. Risques transversaux
- **Zéro silencieux** (porte T3, noms d'images faux T1, endpoint manquant T2) : exécuter validate_submission.py + vérifier que le serveur répond « success » ; le score public affiché doit être cohérent avec le local.
- **Fatigue** : 20 h sans pause entraînent des erreurs de format entre 04:00 et 08:00 ; relecture par une 2e personne obligatoire pour chaque soumission après 02:00.
- **GPU partagés** : prévoir que la file soit lente après minuit ; checkpoints toutes les 10 min ; ne pas perdre le nœud sur une erreur de script (sauvegarde sur /p/scratch).
- **Budget T2 irréversible** : aucune requête ep- sans plan posté ; garder une réserve de 20-30 réponses par endpoint jusqu'à 07:30.
- **Dérive des rapports** : les trois rapports t1/t2/t3 contiennent des fourchettes [HYP] jamais mesurées ; toute décision sur un chiffre non mesuré doit être revue dès qu'une mesure locale existe.
- **Interdit** : toute collaboration avec une autre équipe (disqualification, [SOURCE : CLAUDE.md]) ; ne pas discuter de méthodes sur des canaux publics.

## Sources
- Dépôt : CLAUDE.md, reports/sources/{slides.txt, Watermark_Localization.txt, Model_Equality_Testing.txt, Unlearning.txt, GPU-usage-paris.txt}, reports/t1-*, t2-*, t3-*, BRIEF-3-taches.md.
- Page du championnat : https://cispa.de/en/cispa-hackathon-championship-paris (non lue en entier ; résultat de recherche seulement).
- Édition précédente (classement sur score combiné + présentation) : https://eng.pw.edu.pl/News/WUT-representatives-on-the-podium-at-the-CISPA-hackathon (résumé de recherche, à confirmer).
- Simulation : sim.py, sim2.py dans /private/tmp/claude-501/-Users-titouandestors-Hack-European-Championship/9a07d52b-6d63-4c22-8aa1-0954384631b9/scratchpad/research/.
