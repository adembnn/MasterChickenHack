# Track 3 : tour de critique sur le plateau à 0,39 (session `t3-verificateur`, 22/11, 3 h 25)

Décision débloquée : sur quel facteur du score dépenser les créneaux d'envoi restants, et quelles expériences lancer (5 au plus).
Scripts et sorties : `h200:~/eac/t3/t3-verificateur/` (`fitk.py`, `pg.py`, `pg2.py`, `look.py`), copie dans `E:\EAC-lab\t3\t3-verificateur\`.
Tout chiffre vient d'une sortie de commande ; ce qui est supposé est dit.

## 1. Réponse courte
- Le score est le produit de trois facteurs : score de loss × score de prédiction × rétention.
- **Le score de loss est à 0,95 environ : il ne reste que 5 % au plus.** Les parts 60 à 80 %, les plages de loss et les graines ne bougeront plus rien.
- **Le score de prédiction est le facteur le plus loin de son maximum : entre 0,41 et 0,51** selon la rétention réelle, que nos sondes ne séparent pas.
- Une seule expérience non encore tentée a un gain chiffré localement au-dessus du bruit : **couvrir la bande d'images incertaines** au lieu de les basculer
  à fond (§4, E1). Gain local + 5,5 à + 7,2 % dans huit configurations sur huit ; gain réel attendu 0 à + 0,02 (voir la correction en E1).
- Je ne vois aucune expérience qui promette + 0,05 à elle seule. Rattraper 0,44 demande E1 et une meilleure source (E2, E3) ensemble.

## 2. Décomposition du meilleur fichier (0,3888 et 0,3942, même recette, deux graines)
| Facteur | Valeur | D'où vient le chiffre | Gain possible |
|---|---|---|---|
| Score de loss, images exactes | 0,96 | gold ajusté sur 5 paires notées qui ne diffèrent que par la loss visée (`fitk.py`) : m_gold 5,0 à 5,2, s_gold 4,6 à 4,8 ; fichier 4,85 / 4,57 | + 4 % au plus |
| Score de loss, copies altérées | 0,93 à 0,97 | supposé : dépend du gold sur les copies, non mesuré ; `RV2` (en file) le dira | + 2,5 % à − 3 % |
| Rétention | 0,8 à 1,0 | non identifiée. Les sondes `G10` donnent seulement le rapport K / (1 + KL) ≈ 4 (K = coût d'une image de contrôle faussée) ; mes mélanges donnent KL = 0,24 | + 0 à + 25 %, inconnu |
| Score de prédiction | 0,41 à 0,51 | par division : 0,39 / (0,95 × rétention) | le seul grand réservoir |

Ce qui a déjà payé va dans ce sens : classe fausse redistillée + 9 %, images choisies par redistill + 5 %. Ce sont deux gains de score de prédiction.
Le rapport sonde / fichier reste entre 0,63 et 0,65 de 20 % à 80 % de bascules : la KL du contrôle réel ne monte presque pas avec les bascules.
C'est un indice, pas une preuve, que la rétention est déjà haute et que le score de prédiction est plus près de 0,41 que de 0,51.

## 3. Mesures nouvelles de ce tour
### 3.1 Un gold de substitution, et ce qu'il vaut
`pg.py` : je note nos fichiers contre un gold fait des redistillations hors pli de `t3-redistill` (4 modèles à 6 blocs, 2 à 8 blocs, précision 0,41 à 0,45),
logits multipliés par un facteur calé pour que leur loss moyenne vaille 5,1. Images exactes seulement, sans la rétention.
- Corrélation de rang avec le score réel : **0,93 sur 28 fichiers notés**, 0,75 sur les 18 fichiers au-dessus de 0,30.
- **Il est biaisé en faveur de tout ce qui vient des redistillations.** Rapport réel / local : 1,06 à 1,09 pour la classe votée, 0,91 à 0,94 pour la
  classe redistillée, 0,67 à 0,74 pour le rang redistill. Il annonce + 35 % pour le rang redistill, le serveur a donné + 6 %.
- Il ne reproduit pas le maximum à 70 % : il préfère 60 à 65 %. Le vrai gold se trompe donc sur plus d'images que ces modèles.
- Son écart-type de loss une fois calé vaut 6,7 contre 4,7 pour le gold ajusté : inutilisable pour le score de loss, que je calcule sur le gold ajusté.
- **Usage retenu : comparer deux façons de construire la cible à source égale. Pas pour comparer deux sources, ni pour prédire un score.**

### 3.2 Où part le score de prédiction (`pg2.py`, base `PG_RDt24r_70_l1016c`, gold de substitution)
| Groupe d'images | Part | KL du fichier | KL d'`orig` | Part de la KL totale |
|---|---|---|---|---|
| Bascule voulue, atteinte, classe = 1re classe fausse du gold | 44,7 % | 1,02 | 4,35 | 42 % |
| Bascule voulue, atteinte, autre classe | 21,5 % | 1,87 | 4,11 | 37 % |
| Bascule voulue, non atteinte | 3,8 % | 1,22 | 4,21 | 4 % |
| Image laissée juste | 30,0 % | 0,61 | 1,20 | 17 % |
Par confiance du gold de substitution dans la vraie classe (probabilité à température 2) :
- 0,30 à 0,70 : 22 % des images, 69 % basculées par le fichier, KL 1,55 (**32 % de la KL totale**).
- 0,70 à 0,95 : 18 % des images, 38 % basculées, KL 1,25, soit pire qu'`orig` (0,66) sur ces images.
Lecture : deux pertes dominent. (1) Une bascule sur trois va vers une autre classe que celle du gold. (2) Les images où le gold hésite sont traitées en tout
ou rien : basculées à fond ou laissées justes.

### 3.3 La cible n'est pas le problème d'entraînement
La cible `tgt_RDt24r_70_l1016` a m 9,28 / s 6,27 et un score de prédiction local de 0,50 ; le fichier entraîné a 4,85 / 4,57 et 0,67. Le fichier est
meilleur que sa cible : l'ancre l'adoucit. Réentraîner plus longtemps ou plus de blocs pour « mieux atteindre la cible » n'a pas de raison de payer
(`PGB6` : 0,658 local, m 5,85 / s 5,17 au-dessus du gold).

### 3.4 Ce que sont les images (`look.py`, planches regardées)
Les images sont générées. Les classes vont par voisines presque identiques : classes 6 et 7 = deux sortes de lanternes, classes 30, 31, 33 = barques sur l'eau.
Les confusions des modèles profonds sont entre voisines (7 → 6 : 46 images, 6 → 7 : 30 ; 30 ↔ 31 ; 19 ↔ 21 ; 43 ↔ 42). Les images « étiquette 7, modèles 6 »
ressemblent à l'œil aux images de la classe 6 (lanterne suspendue dehors) plus qu'à celles de la classe 7 (lanterne posée, intérieur).
Sur les 6 modèles profonds : 32 % des images où au moins 5 donnent la classe d'`orig`, 32 % où au moins 5 donnent une même autre classe, **36 % sans
majorité de 5**. C'est cette dernière part que la recette actuelle traite en tout ou rien.

### 3.5 Bascules dures contre bande couverte, à source égale (`pg2.py`, section d)
Sorties synthétiques construites sur un jeu de modèles, notées sur un jeu disjoint (2 modèles à 6 blocs + 2 à 8 blocs) ; score de loss sur le gold ajusté.
« Trois paliers » = 45 % du haut du rang basculés à loss 10, les 20 % suivants à probabilité de la vraie classe 0,2 (température 2), le reste inchangé.
| Source du rang et de la masse fausse | Bascules dures, meilleur réglage | Trois paliers | Gain |
|---|---|---|---|
| 1 modèle à 4 blocs | 50 % à 9,5 : 0,949 × 0,600 = 0,569 | 0,994 × 0,613 = 0,609 | + 7,0 % |
| 3 modèles à 4 blocs | 0,949 × 0,666 = 0,632 | 0,994 × 0,679 = 0,675 | + 6,8 % |
| 8 modèles à 4 blocs | 0,949 × 0,707 = 0,671 | 0,993 × 0,725 = 0,719 | + 7,2 % |
| 8 × 4 blocs + 2 × 6 blocs | 60 % à 8,5 : 0,962 × 0,833 = 0,801 | 0,988 × 0,858 = 0,847 | + 5,7 % |
Même tableau avec un facteur fixe de 2,5 au lieu du calage : + 5,8 %, + 6,0 %, + 5,6 %, + 5,5 %. Le gain vient pour moitié du score de loss (m et s
touchent le gold ensemble) et pour moitié du score de prédiction.
Limite : c'est un monde de substitution ; le gain réel peut être plus petit ou nul. Mon idée initiale d'une bande à 0,45 n'a rien donné : la bonne
valeur locale est 0,2.

## 4. Expériences, par ordre de gain attendu
**E1. Bande couverte (hors des pistes listées).** Même rang et même masse fausse que la base. 45 à 50 % du haut : bascule avec une loss visée plus haute
(fichier à ≈ 10 sur ces images, contre 7,3). Rangs 50 à 70 % : sortie partagée, probabilité de la vraie classe à température 2 entre 0,2 et 0,35, le reste
sur les classes fausses de la source. Rangs 70 à 100 % : comme aujourd'hui. Régler pour un fichier à m 4,9 à 5,1 et s 4,6 à 4,7.
Deux fichiers : bande 50-70 % et bande 55-80 % (le vrai gold se trompe plus que le gold de substitution). Contrôle avant envoi : `pg2.py` doit donner
un produit local au-dessus de 0,646 (la base). Lecture réelle : ≥ 0,400, on élargit la bande ; ≤ 0,385, piste fermée. Attendu : 0,390 à 0,410.
Correction (3 h 30, relevée par `t3-redistill`) : dans mon essai la masse fausse est concentrée, donc la bande à 0,2 y porte une loss de 2,4 ; avec la masse
étalée de la recette elle porte 0,7. La moitié « score de loss » du gain local ne se transpose pas : compter + 2 à + 4 % en local pour les fichiers `PG_RDb*`.

**E2. Rang de consensus, fichier prêt.** `PG_RDt24rc_70_l1117c` (rang 4 + 6 + 8 blocs, conforme, porte go) a le meilleur score de prédiction local de tous
les fichiers déposés : 0,723 contre 0,674 pour la base. Le gold de substitution surévalue ce genre de gain d'un facteur 5 environ : attendu réel + 1 à
+ 1,5 % (0,393 à 0,400). C'est le prochain créneau utile parmi les fichiers existants.

**E3. Masse fausse moyennée sur tous les modèles.** 37 % de la KL locale vient des bascules vers une autre classe que celle du gold. `PG_RDp_70_l1016c`
(8 partitions, en file) en est le premier test : local 0,685 contre 0,674. S'il fait ≥ 0,395, refaire avec la moyenne des 14 modèles (4, 6 et 8 blocs).
Attendu : + 0,005 à + 0,01. À combiner avec E1 dans le même fichier dès que E1 est lu.

**E4. Copies altérées.** Rien à lancer : `PG_RDt24rRV2_70_l8515c` est en file. ≥ 0,399 : garder le réglage pour E1 ; ≤ 0,384 : fermé.

**E5. Mesurer la rétention avant d'y toucher.** Aucune sonde par échange de classes ne sépare le niveau de KL du coût par image. Je n'ai pas de
protocole qui la mesure en un envoi ; une ancre plus forte a déjà détruit les bascules (`lam` 10 : m = 1,6). Je déconseille d'y mettre des créneaux
tant que E1 n'est pas lu.

## 5. À ne plus envoyer
- Parts 68 et 72 % (72 % = 0,3849, attendu 0,390 par la parabole, dans le bruit), parts 50, 55 %, nouvelles graines, moyennes de poids.
- `PGB6_RDt24r_70` sous cette forme (prédit 0,345 à 0,380), `PG_RDt24_85_*` et `_90_*` (autre recette).
- Tout fichier au-dessus de 70 % de bascules dures avec le rang redistill.

## 6. Limites de ce rapport
- Le gold de substitution n'est validé que comme classement grossier ; ses gains chiffrés ne sont pas des gains réels.
- La rétention réelle et le gold sur les copies altérées restent inconnus.
- Le bruit entre deux graines repose sur une paire (0,3888 / 0,3942).
