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
| T1 | **0,7384** (soumission 1540), 1er | Eugenia 0,73, Fable 4.2 0,72 | `submission_candidat_5.pt` (h200, h200b, `E:\EAC-lab\sauvegarde-t1\`) | presque rien tant qu'on est 1er (score ramené à 1) |
| T2 | **aucun envoi** | Fable 4.2 0,75 ; quatre équipes à 0,70 | — | **le plus gros levier** : sans T2 l'équipe n'est pas classée ; 0,70 ≈ +0,3 au général |
| T3 | **0,0735** (soumission 1445) | 0,27 | `PG_T85_f20_l48c.pt` (`h200:~/eac/t3/t3-methodes/candidates/`) | 0,15 ≈ +0,09 ; 0,27 ≈ +0,24 |

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
| t3-simulateur | monde A (golds entraînés de zéro) : quel sélecteur prédit le mieux les erreurs du gold | h200e |
| t3-verificateur | prédiction chiffrée et verdict de porte avant chaque envoi T3 | — |
| t2-critique | appliquer la règle à 0,70 aux vraies réponses de Titouan dès qu'elles existent | — |

Arrêtées : t1-residus (filtre adapté mort, boîtes reprises par t1-assemblage), t3-sondes (son calage est fait par le vérificateur, ses sondes ne décident plus rien), t2-methode (méthode non validée sur le réel, rendue sur la PR #27), et les variantes NEG / NPO de t3-poids.

### Règles en vigueur (voir `CLAUDE.md`)

- T1 : aucun envoi sans +0,02 de moyenne de réserve et feu vert écrit du vérificateur.
- T3 : annoncer chaque envoi sur l'issue #5 ; 10 min entre deux envois, partagés avec l'agent de Yassir ; envois détachés depuis `h200:~/eac/t3/orchestrateur/` (`nohup bash sub.sh <fichier> > sub_<nom>.log`).
- T2 : aucun appel API, aucun envoi.
- Calcul : cinq H200 (hôtes SSH `h200`, `h200b`, `h200c`, `h200d`, `h200e` ; les deux dernières par rebond sur `h200`), toutes à remplir. Jülich : rien.
- Clé du leaderboard : `h200:~/eac/.env`, jamais lue ni affichée.
- Scripts de l'orchestrateur : `E:\EAC-lab\remote\`. Dans une commande SSH, ne jamais écrire un `pkill -f` ou `pgrep -f` dont le motif apparaît dans la commande elle-même : il tue sa propre session. Passer par un fichier de script.

- **En ce moment :** voir la passation ci-dessus.
- **Ce que les autres doivent savoir :** annoncez vos envois T3 sur l'issue #5 avant de les faire ; les prédictions T1 partagées sont dans `h200b:/tmp/eac-t1-partage/`.
