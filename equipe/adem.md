# Adem (`adembnn`)

Fiche personnelle : seul Adem (ou son agent) l'édite. Les autres agents la lisent pour savoir sur quoi il travaille.

- **Track(s) en charge :** T1 et T3 (envois compris). T2 : soutien à Titouan, jamais d'appel API ni d'envoi.
- **Machine et agent :** Claude Code sur PC Windows, une session orchestratrice et des sous-sessions.
- **Compte Jülich :** banni1 (plus aucun job ; règle : au plus un job à un GPU, lancé par l'orchestrateur).

## Passation de l'orchestrateur (21/11, 21 h 05)

Écrite pour qu'une session neuve reprenne sans l'historique. À lire avec `CLAUDE.md` (section Décisions) et les issues #3, #4, #5.

### Scores réels

| Tâche | Notre meilleur | Tête du tableau | Fichier | Ce que vaut un gain pour le classement général |
|---|---|---|---|---|
| T1 | **0,8033** (candidat 35, soumission 2550, envoyé par l'agent de Yassir vers 3 h ; validation croisée 0,8138, +0,025 prévu contre le 27, +0,010 réel) ; avant : 0,7935 (candidat 27, soumis par l'agent de Yassir vers 1 h 10 : sortie intermédiaire de t1-assemblage avec la carte StegaStamp pleine résolution) ; avant : 0,7695 (soumission 1984, 23 h 45 ; `submission_candidat_22.pt`, sha 48b0f3b9, sur h200, h200b et `E:/EAC-lab/sauvegarde-t1/`), 1er | Murmell, LesRooters, Incoming PhD 0,78 (3 h 08) | réserve 0,7514 (prédit +0,028, réel +0,031) | base de comparaison de tout nouveau candidat |
| T2 | **0,60** (Titouan, 0 h 40) ; avant 0,40 (envoi 1882 ; critique `reports/critique/t2-tour-severe.md`) | cinq équipes à 1,00 (Murmell, Fable 4.2, Azerit, 3V1B, Unblo), Eugenia 0,85 (3 h 08) | — | **le plus gros levier** : sans T2 l'équipe n'est pas classée ; 0,70 ≈ +0,3 au général |
| T3 | **0,3942** (soumission 2531, 2 h 36 ; `SOUPs1.pt` : même recette que le 0,3888, graine 1 ; ensuite, sans gain : moyenne de 4 graines 0,3838, part 60 % 0,3648, 65 % 0,3743, 80 % 0,3657, le sommet est à 70 %) ; avant : 0,3888 (soumission 2344, 1 h 46 ; `PG_RDt24r_70_l1016c.pt` : images ET classe fausse choisies par les modèles redistillés de t3-redistill, 70 %, loss 10-16 ; étapes : classe redistill 0,3273 (60 %) → 0,3502 (70 %) → loss 10-16 0,3653 → 80 % 0,3713 → rang redistill 0,3888 ; sondes de porte tenues à 70 et 80 % ; axe rétention locale réfuté : `RDL_s3k_l6` 0,3578) ; avant : 0,3502 (`PG_RDt24_70_l915c.pt` : classe fausse tirée des modèles redistillés de t3-redistill, cible `cache/tgt_RDt24_70.pt` ; même recette à 60 % = 0,3273) ; avant : 0,3179 (`PG_T85_f80_l814c.pt` ; 70 % = 0,3159, palier de la part ; `PG_T85_f70_l814c.pt`, sauvegardé dans `E:/EAC-lab/sauvegarde-t3/`). Série à loss 8-14, part d'images basculées seule changée : 20 % 0,1487 / 30 % 0,1962 / 40 % 0,2400 / 50 % 0,2779 / 60 % 0,3009 / 70 % 0,3159. Sondes de porte `G10_f60` 0,189 et `G10_f70` 0,2025 (> 0 : au moins 10 points de marge). `PGho_f60` 0,2701 (pire). Piste « 9 classes entières » réfutée (`reports/critique/t3-tour-classes-entieres.md`) | 0,44 (Murmell), 0,42 (Skesiea) (3 h 08) | `h200:~/eac/t3/t3-methodes/candidates/` | suite : sonde `G10_f80` puis `f80` ; forme des cibles à m et s égaux (masse fausse répartie, images non basculées plus étalées, copies) chez t3-methodes |

À 21 h 50 : grille (part × loss) et variantes `PGr_` (rotations) sur `h200`, variantes rétention `PGl6_` (`--lam 6`), `PGs3k_` (3000 pas) et parts 50-60 % sur `h200c` (`~/eac/t3/t3-methodes`, lancées par `vague3c.sh`), candidats écrits par `code/finish.sh`. Avant, à 21 h 10 : deux entraînements lancés par l'orchestrateur sur `h200e` (`~/eac/t3/t3-methodes/logs/PG_T85_f30_l814c.log` et `…f40_l814c.log`, mêmes réglages à 30 % et 40 % d'images) ; le fichier candidat s'écrit ensuite par `code/dash.py --save` (voir `code/finish.sh`). Session neuve `t3-redistill` (id `local_b4394f32-4858-48d7-8fc2-2374103c08d1`) lancée sur la piste 3 ci-dessous. Ids des autres sessions : liste « linked » de l'application.

### Ce qui est établi (mesuré, vérifié par une session séparée)

- **T1.** La réserve (`index % 5 == 0`) prédit le public à ±0,005 près, cinq fois de suite. Ce qui a payé : U-Net sur exemples fabriqués + vraies images ; recette « toile agrandie puis rognée » (StegaStamp +0,02) ; **géométrie sur les images partielles** : masques SAM pour les objets, boîte trouvée par le décodeur pour les rectangles (TrustMark 0,597 → 0,678). Réserve actuelle : StegaStamp 0,843, TrustMark 0,678 (pleines 0,616), PixelSeal 0,648. Écartés par la mesure : filtre adapté, cartes du décodeur, occlusion, inférence zoomée, Lovász, tige stéganalyse (pas de gain à 5000 pas), modèle par population.
- **T3.** Modèle du vérificateur, qui a prédit quatre envois avant leur résultat : le gold (modèle réentraîné sans les images) a une précision de 0,5 à 0,7 sur les images à oublier, avec des erreurs très confiantes (loss moyenne 2,6 à 6, écart-type 5 à 8). Notre facteur limitant est le score de loss (0,3 pour notre meilleur), puis le score de prédiction (0,4). La porte des 70 % a au moins 10 points de marge (`G10_PG_T70` = 0,0208). Réfutés par un envoi réel : adoucir toutes les images (`TS_tau4` = 0,0057), « 9 classes oubliées en entier » (`Z9_b0` = 0,0), négation d'un vecteur de tâche (0,0219). Recette qui marche : erreurs confiantes sur 20 % des images, reste adouci, ancre sur mélanges (famille `PG` de t3-methodes).
- **T2.** Le 0,842 de la méthode d'Adem est un score sur mondes simulés ; la même méthode fait 0,111 si les endpoints échantillonnent. Rien n'a touché le réel. Titouan a annoncé une collecte de 25 réponses par endpoint à 20 h 16, sans résultat publié.

### Ce qui peut faire rebondir (par ordre de valeur)

1. **T2 : obtenir un envoi.** Décision qui revient à Adem et Titouan. t2-critique a posté sur l'issue #4 la lecture du tableau (0,70 = familles justes + un seul type d'erreur) et une règle de décision sans seuil.
2. **T3 : prédire QUELLES images le gold rate et VERS quelle classe**, au lieu de les tirer au hasard. Le vérificateur chiffre ce choix à un facteur 2 à 3. Trois sélecteurs en cours d'entraînement (`PS30_rand`, `PS30_oof`, `PS30_err`) ; cibles hors échantillon de t3-poids (`oof.py`).
3. **T3 : réinitialiser et redistiller** (piste neuve, session `t3-redistill`) : remettre à zéro les derniers blocs et la tête, puis les réapprendre par distillation d'`orig` sur des images que le modèle n'a jamais vues (images générées du même domaine, mélanges). Les erreurs sur les images mémorisées apparaissent alors d'elles-mêmes, avec leur classe et leur confiance naturelles.
4. **T1 : pleines TrustMark et PixelSeal** (0,62 à 0,65) par entraînement long sur exemples fabriqués seuls (à 5000 pas il égale déjà les modèles avec vraies images ; verdict à 10 000 et 20 000 pas).

### Sessions gardées (9) et arrêtées (4)

| Session | Mission unique | Machine |
|---|---|---|
| t1-assemblage | fabriquer le candidat T1 cumulé (SAM + boîtes sur les trois filigranes) | h200b |
| t1-pixelseal | masques SAM train + test des trois filigranes, règle d'union | h200d |
| t1-reel | 3 entraînements longs au plus (TrustMark réf., TrustMark pleine résolution, PixelSeal) | h200b, h200d |
| t1-verificateur | feu vert écrit avant tout envoi T1 (+0,02 de réserve, IC95 hors de 0) | — |
| t3-methodes | candidats T3 (erreurs confiantes, sélecteurs, 30 à 40 %) | h200, h200e |
| t3-poids | sélecteurs seulement : sonde hors échantillon, classe adversariale | h200c |
| t3-nonmembres | réservoir d'images générées jamais vues, étiquetées par `orig` | h200b, h200c |
| t1-residus | carte locale de « re-tamponnage » (seule mission) | h200b |
| t3-verificateur | prédiction chiffrée et verdict de porte avant chaque envoi T3 | — |
| t2-critique | appliquer la règle à 0,70 aux vraies réponses de Titouan dès qu'elles existent | — |

Arrêtées : t3-simulateur (mission rendue à 21 h 35), t3-sondes (son calage est fait par le vérificateur, ses sondes ne décident plus rien), t2-methode (méthode non validée sur le réel, rendue sur la PR #27), et les variantes NEG / NPO de t3-poids.

### Règles en vigueur (voir `CLAUDE.md`)

- T1 : aucun envoi sans +0,02 de moyenne de réserve et feu vert écrit du vérificateur.
- T3 : annoncer chaque envoi sur l'issue #5 ; 10 min entre deux envois, partagés avec l'agent de Yassir ; envois détachés depuis `h200:~/eac/t3/orchestrateur/` (`nohup bash sub.sh <fichier> > sub_<nom>.log`).
- T2 : aucun appel API, aucun envoi.
- Calcul : six H200 (hôtes SSH `h200`, `h200b`, `h200c`, `h200d`, `h200e` par rebond sur `h200`, et `h200f` = nœud 6, 204.12.188.1, autre réseau), toutes à remplir ; suivi https://h200.ami.exchange/ (en retard de quelques minutes) ; VM préemptibles (n4 et n5 éteintes de 22 h 25 à 22 h 45, disques conservés). Jülich : rien.
- Clé du leaderboard : `h200:~/eac/.env`, jamais lue ni affichée.
- Scripts de l'orchestrateur : `E:\EAC-lab\remote\`. Dans une commande SSH, ne jamais écrire un `pkill -f` ou `pgrep -f` dont le motif apparaît dans la commande elle-même : il tue sa propre session. Passer par un fichier de script.

- **En ce moment (22/11, 3 h 10) :** 3e au général (0,83 ; Murmell 0,97, Fable 4.2 0,87). L'orchestratrice (`local_15adbc8e…`) ne garde que la file d'envoi T1 / T3 ; une session relais (`local_f823412d-fdc1-4f13-8c33-371df79bc478`) reçoit les sous-sessions T1, tient les commits, le bus #2, le tri de la file T3 et le suivi T2. Elle n'envoie rien au leaderboard.
- **Ce que les autres doivent savoir :** annoncez vos envois T3 sur l'issue #5 avant de les faire ; les prédictions T1 partagées sont dans `h200b:/tmp/eac-t1-partage/`.
