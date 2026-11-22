# T3 Unlearning : rapport de la session t3-methodes

État au 22/11, 2 h 00. Décision débloquée : quelle méthode et quel fichier soumettre sur le track 3.
Les chiffres viennent de commandes lancées sur la H200 (mode évaluation, fp16, comme le serveur). Tableau complet : `reports/t3-tableau.md`. Code : `tracks/t3-unlearning/t3-methodes/`. Fichiers : H200 `~/eac/t3/t3-methodes/candidates/`.

## En bref
1. **Score** : meilleur de l'équipe à 1 h 40 le 22/11 : **0,3713**. Progression : 0,0241 → 0,0735 → 0,1487 → 0,1962 → 0,2779 → 0,3009 → 0,3273 → 0,3676 → 0,3713.
2. **Ce qui paie, mesuré au serveur** : (a) des erreurs confiantes (loss 8 à 14) sur les seules images à oublier ; (b) une part d'images basculées élevée, jusqu'au palier de 70 à 80 % ; (c) la classe fausse choisie par des modèles réappris sans l'image (t3-redistill) plutôt que par mes vues dégradées : +0,026 à 60 %.
3. **Ce qui ne paie pas** : adoucir toutes les images (0,0241), un adoucissement global (0,0057), l'ancre sur vues très dégradées (0,0916 contre 0,1487), mes réglages `PGh` / `PGho` (0,2701 contre 0,3009), étaler davantage les images non basculées (0,3037 contre 0,3009 : neutre).
4. **Ce que j'ai tenté pour la rétention, et qui a perdu au serveur** : `PGh` / `PGho` (0,2701 contre 0,3009) puis la recette « RDL », entraînement plus long et ancre plus forte (0,3578 contre 0,3653 à loss égale). Diagnostic : sur les images basculées rien ne change ; ces recettes laissent les images NON basculées plus proches d'orig, donc plus sûres d'elles, alors que le gold y est moins sûr qu'orig. Rester collé à orig sur les images du jeu n'est pas ce que le serveur récompense.
5. **Outil de décision, avec sa limite** : la réserve de 698 images et la dérive sur mélanges mesurent « proche d'orig ». Elles ont bien prédit la porte à 70 % et les effondrements de classe, mais elles se sont trompées trois fois sur le score (`PGho`, `RDL`, et `PGs70` qu'elles donnaient perdant). À garder comme garde-fou de porte, pas comme critère de choix.

## 1. Relecture du code du premier agent (`scripts-t3v1/`)
Ce qui est faux ou fragile, par ordre de gravité :
- **Le gold supposé a un écart-type de loss trop bas.** `targets.py` pose qu'une image mal classée a une loss de 3,0 ± 1,3 et en tire, pour un gold à 85 % de précision, moyenne 0,55 et écart-type 1,17. Mesure : quand orig se trompe (sur des versions dégradées des images), sa loss moyenne est de 3,95 et, à 85 % de précision, moyenne 0,63 et écart-type 1,76. Un gold entraîné de la même façon se trompe sans doute avec la même assurance. Aucun des 14 candidats ne dépasse 0,96 d'écart-type.
- **Deux cibles incompatibles.** `make_q` vise une distribution à température 1 et une autre à température 2 qui ne peuvent pas sortir du même modèle (la seconde ajoute 15 % d'une autre distribution). Les deux pertes se combattent : c'est une raison pour laquelle les candidats « sous-tiennent » leur cible (loss 0,22 à 0,38 pour 0,55 visé).
- **L'image exacte n'est jamais montrée à l'entraînement**, seulement des vues recadrées et recolorées. Or les images exactes pèsent la moitié de la note d'oubli.
- **La mesure de difficulté mélange trois signaux dont un est indépendant des deux autres** (corrélation mesurée par l'agent lui-même). Le classement final est donc en bonne partie du bruit, et rien ne dit qu'il prédit les erreurs du gold. Deux de ces signaux ne connaissent que les 45 classes présentes : ils ne peuvent pas proposer une classe voisine parmi les 55 absentes.
- **L'ancre et l'oubli se contredisent en partie.** L'ancre demande de garder la réponse d'orig sur des collages qui contiennent 40 à 80 % d'une image à oublier, pendant que l'oubli demande de changer la réponse sur des recadrages de la même image.
- **La dérive n'était mesurée que hors domaine** (CIFAR, ImageNet), où orig hésite déjà : ces chiffres ne disent rien du jeu de contrôle caché, fait d'images du même jeu. D'où la mesure sur « réserve » ci-dessous.
- **Les contrôles tournent en rond** : « KL cible propre » et « score de loss simulé » comparent le candidat à l'hypothèse qui a servi à l'entraîner.
- **Copies de contrôle trop dures** : la moyenne des 4 types est dominée par « couleur forte », où orig lui-même se trompe 48 fois sur 100. Le serveur parle de copies « légèrement » modifiées.
- La perte « de moments » estime un écart-type sur 64 images par lot : très bruité, et redondant avec les cibles par image.

## 2. Mesure nouvelle : la réserve
Sans images du jeu en dehors des 3 500, personne ne pouvait mesurer la rétention. Je retire 698 images à oublier (20 % de chaque classe) de l'entraînement : ce sont des images du même jeu, des mêmes classes, qu'orig connaît par cœur, comme les images d'entraînement que le gold garde. J'applique dessus la formule de rétention du serveur. Limite : elle ne couvre que les 45 classes présentes, et le fichier final est entraîné sur les 3 500 images.
| Recette | Classes changées (oubli) | Rétention sur la réserve |
|---|---|---|
| Sans protection | 12,1 % | 0,52 |
| Ancres croisées (mon idée : oublier une moitié en protégeant l'autre, puis l'inverse, et additionner) | 5,4 % | 0,49 |
| Ancre « mélanges » du premier agent, forte (λ=10), 600 pas | 0,7 % | 0,91 |
| Pseudo-gold + ancre mélanges légère (λ=3), 1 200 pas | 13,7 % | **0,90** |
Lecture : les bascules de classe restent sur les images visées (0 à 0,4 % de bascules dans la réserve), mais l'adoucissement de la confiance se propage à toute la classe. L'ancre « mélanges » freine cette propagation ; l'idée des ancres croisées ne survit pas à l'addition des deux déplacements de poids.

## 3. Ce que les scores réels ont appris (et où je me suis trompé)
| Fichier | Loss moyenne / écart-type | Bascules | Score réel |
|---|---|---|---|
| `S0_tau3` (adoucissement seul, 1er agent) | 0,012 / 0,040 | 0 % | 0,0010 |
| candidat de Yassir (tête × 0,35) | très adouci | 0 % | 0,0039 |
| `TS_tau4` (tête ÷ 4, global) | 0,401 / 0,229 | 0 % | 0,0057 |
| `S1_A85_l30` (1er agent) | 0,087 / 0,638 | 1,9 % | 0,0114 |
| `NEGce_all_a2` (t3-poids) | 0,229 / 1,489 | 4,1 % | 0,0219 |
| `PG_T85` (moi) | 0,301 / 0,802 | 13,0 % | 0,0241 |
| `S1_r2_A85_l10_lr1e-4` (1er agent) | 0,376 / 0,961 | 9,3 % | 0,0273 |
| `PG_T70` (moi) | 0,435 / 0,752 | 19,5 % | 0,0323 |
| `PG_T85_f20_l48c` (moi) | 1,066 / 2,236 | 19,8 % | 0,0735 |
| `PD_f20_l814` (moi, ancre sur vues dégradées) | 1,377 / 3,170 | 19,3 % | 0,0916 |
| `PG_T85_f20_l814c` (moi) | 1,824 / 3,867 | 19,9 % | 0,1487 |
| `PG_T85_f20_l1218c` (moi) | 2,312 / 4,974 | 19,9 % | 0,1707 |
| `PG_T85_f30_l814c` (ma recette) | 2,564 / 4,293 | 29,6 % | 0,1962 |
| `PG_T85_f30_l1218c` (ma recette) | cible 4,5 / 6,9 | 30 % | 0,1998 |
| `PG_T85_f50_l814c` (ma recette) | 50 % × 8-14 | 50 % | 0,2779 |
| `PG_T85_f60_l814c` (ma recette) | 60 % × 8-14 | 60 % | **0,3009** |
| `Z9_b0` (têtes de 9 classes à zéro) | 3,21 / 4,76 | 33,4 % | 0,0 |
- **Mon erreur** : j'avais déduit de 0,0114 que la note de prédiction était le verrou et qu'il suffisait d'étaler toutes les images à température 2. `PG_T85` l'a fait et n'a rien gagné. Le verrou est la note de loss : le gold a une loss bien plus haute que tout ce que nous visions.
- **Ce qui tient** : pas de saturation jusqu'à 0,435 de loss moyenne ; l'incertitude appliquée à tout le monde est punie ; `Z9_b0` (tout faux sur 9 classes, partout) vaut 0, par la porte ou par excès.
- **Estimations du gold venues des autres sessions** : précision 0,62 à 0,80, loss 0,9 à 2,1, écart-type 2,1 à 3,5 (images régénérées, t3-nonmembres) ; ou, si 9 classes ont été retirées en entier, loss 3,7 et écart-type 5,4 (t3-simulateur). Mes vues dégradées donnent le même ordre de grandeur quand orig tombe à 68 % : 1,58 / 2,91.

## 4. La méthode, version actuelle
Idée de départ : le gold n'a jamais vu ces images. Orig, devant une version assez dégradée d'une image, ne peut plus s'appuyer sur sa mémoire. Ses réponses sur ces versions servent de modèle du gold, sans donnée externe.
- **Vues dégradées** (`pg_views.py`) : flou, recadrage serré, masquage de carrés, bruit, couleur, à 5 forces : 100 réponses d'orig par image. Aucune famille ne renvoie vers une classe fourre-tout.
- **Cible** (`mk_tgt.py`) : pour chaque image, la moyenne de 20 réponses, lissée par celle de sa classe. Puis une part choisie d'images, les plus instables sous dégradation, bascule vers sa classe voisine la plus votée, avec une loss imposée de 4 à 8 (orig lui-même a une loss de 4 à 7 quand il se trompe).
- **Entraînement** (`train.py`) : 4 derniers blocs, sur l'image exacte et des vues légères. Un terme colle la loss de chaque image à sa loss cible : sans lui les bascules restaient molles (loss 1,4 à 2,3) et l'écart-type plafonnait à 0,8.
- **Protections** : ancre « mélanges » (collages d'images à oublier, garder la réponse d'orig) et, depuis 20 h, ancre « vues très dégradées » (garder la réponse d'orig là où il se comporte déjà comme devant une image jamais vue). La seconde protège la précision sur images jamais vues, donc la porte à 70 %.
- **Abandonné** : ancres croisées (rétention 0,49) ; toutes les images rendues incertaines sans bascule (rétention 0,40) ; mon simulateur hors ligne comme guide de réglage (son faux gold est trop facile).

## 5. État des fichiers et suite
| Fichier | Réglage | Rétention sur la réserve | État |
|---|---|---|---|
| `PG_T85_f60_l814c` | 60 % × loss 8-14, recette simple | en cours de mesure | 0,3009 |
| `PG_T85_f50_l814c` | 50 % | 0,630 | 0,2779 |
| `PGho_f50_l814c` | 50 %, cumul des deux réglages | 0,758 | livré, non recommandé depuis le 0,2701 |
| `PGho_f60_l814c` | 60 % | 0,670 | **0,2701** : sous la recette simple |
| `PGho_f70_l814c` | 70 % | 0,568 | livré, à faire précéder d'une sonde de porte |
| `PGrd1_f30_l814c` | 30 %, images et classe choisies par t3-redistill | 0,841 (mon choix : 0,795) | livré |
| `PGrd1h_f60_l814c` | 60 %, idem + images non basculées intactes | en cours | en entraînement |
| `PGs70_f60_l814c`, `PGs48_f60_l814c` | 60 %, images non basculées plus étalées (probabilité de la bonne classe à T=2 : 0,64 puis 0,44, contre 0,83) | en cours | en entraînement |
Non proposé : `PGx` (ancre sur images générées par un modèle externe). Il n'apporte rien de plus que `PGh`, et l'orchestrateur a tranché : pas de risque de disqualification.

## Mesuré / supposé
- Mesuré : les tableaux, la réserve, le profil de loss d'orig sur vues dégradées, les contrôles de fichier, les scores réels.
- Supposé : le gold (précision, classes voisines choisies, hypothèse des 9 classes) ; que le jeu de contrôle caché ressemble à la réserve ou à ses copies floues.
- Non mesuré : la dérive sur les 55 classes absentes des images à oublier ; la précision réelle sur la validation cachée.
