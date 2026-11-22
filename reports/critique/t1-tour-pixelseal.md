# T1 · tour de critique centré sur PixelSeal (vérificateur, 22/11, 4 h 30)

Décision débloquée : quelles expériences T1 lancer d'ici 12 h, PixelSeal d'abord, sans refaire ce qui est déjà mesuré.
Base : candidat 35 (public 0,8033). Chiffres ci-dessous : 170 images par filigrane, prédictions hors-pli (140 hors réserve + 30 réserve pour
PixelSeal), masques du candidat 35 en validation croisée complète (`candidat_35_cv_*.npz`) et cartes `ps_f_x1+f_s1+f_deca`,
`tm_f_x1+f_c5+f_r50`, `ss_f_x1+f_c4` de t1-reel. Scripts : `E:\EAC-lab\t1s\t1-verificateur\{pstour,psphys,chroma,canal0,cocomatch,dupmask}.py`,
sorties sur `h200b:~/eac/t1/t1-verificateur/`. Les gains « espérés » sont mes estimations, pas des mesures ; les plafonds sont mesurés.

Termes : *zone rare* = la plus petite classe de l'image (trou propre d'une pleine, zone filigranée d'une partielle) ; *IoU rare* = recouvrement
intersection / union sur cette zone ; *AUC* = probabilité que la carte range une case de la zone rare avant une autre case (0,5 = hasard) ;
*touchée* = la case la plus sûre de la carte tombe dans la vraie zone ; *case* = cellule 4×4 px du masque 128×128.

## 1. Où part le 0,30 qui manque à PixelSeal (candidat 35 : 0,699 en validation croisée, réserve 0,672)

| Tranche | n | mIoU candidat | mIoU carte au seuil 0 | IoU rare | AUC | touchée | Part de la perte |
|---|---|---|---|---|---|---|---|
| Pleines | 85 | 0,695 | 0,695 | 0,450 | 0,933 | 0,79 | 51 % |
| Partielles | 85 | 0,703 | 0,662 | 0,448 | 0,874 | 0,64 | 49 % |
| Pleines, trou < 1 % | 12 | 0,505 | 0,513 | 0,035 | 0,916 | 0,17 | 12 % |
| Pleines, trou 1-3 % | 16 | 0,655 | 0,647 | 0,338 | 0,899 | 0,75 | 11 % |
| Pleines, trou 3-6 % | 15 | 0,669 | 0,646 | 0,376 | 0,909 | 0,87 | 10 % |
| Pleines, trou 6-15 % | 23 | 0,769 | 0,786 | 0,589 | 0,962 | 0,91 | 10 % |
| Pleines, trou > 15 % | 19 | 0,779 | 0,777 | 0,698 | 0,955 | 1,00 | 8 % |
| Partielles, zone < 1 % | 17 | 0,499 | 0,514 | 0,036 | 0,733 | 0,24 | 17 % |
| Partielles, zone 1-3 % | 19 | 0,569 | 0,618 | 0,172 | 0,805 | 0,42 | 16 % |
| Partielles, zone 3-6 % | 10 | 0,716 | 0,693 | 0,475 | 0,891 | 0,70 | 6 % |
| Partielles, zone 6-15 % | 20 | 0,830 | 0,681 | 0,696 | 0,953 | 0,80 | 7 % |
| Partielles, zone > 15 % | 19 | 0,878 | 0,801 | 0,819 | 0,975 | 1,00 | 5 % |
| Rectangles, petit côté < 64 px | 30 | 0,530 | 0,530 | 0,103 | 0,734 | 0,30 | **28 %** |
| Rectangles, petit côté 64-128 px | 10 | 0,772 | 0,720 | 0,575 | 0,962 | 0,80 | 4 % |
| Rectangles, petit côté ≥ 128 px | 3 | 0,948 | 0,752 | 0,923 | 0,933 | 1,00 | 0 % |
| Objets partiels, 1 morceau | 25 | 0,820 | 0,718 | 0,675 | 0,940 | 0,76 | 9 % |
| Objets partiels, plusieurs morceaux | 17 | 0,751 | 0,760 | 0,567 | 0,961 | 0,88 | 8 % |
| Pleines, famille A (taches lisses) | 45 | 0,647 | 0,641 | 0,359 | 0,908 | 0,76 | **31 %** |
| Pleines, famille B (silhouette) | 38 | 0,745 | 0,757 | 0,548 | 0,960 | 0,82 | 19 % |
| Mal classées pleine / partielle | 4 | 0,896 | 0,914 | | | | 1 % |

Lecture :
- **Les zones rares de moins de 3 % (64 images, 38 %) portent 55 % de la perte** ; parfaites, elles mettraient PixelSeal à 0,865.
- **Les petits rectangles (30 images) portent 28 % de la perte** à eux seuls, avec une IoU rare de 0,10.
- L'erreur de population (pleine / partielle) ne coûte rien : 4 images, 0,002.
- **Bords** : 32 % des cases fausses sont à 2 cases ou moins (8 px) de la vraie frontière, 50 % à 4 cases ou moins. Corriger toutes
  celles à 2 cases ou moins donnerait 0,785 (+0,086) ; toutes les autres seules, 0,839.
- **Types d'erreur** : dans les pleines, 25 % du vrai trou est manqué et 25 % du trou prédit est faux (dont 44 % loin du vrai trou) ; dans les
  partielles, 38 % des images n'ont aucun recouvrement entre zone prédite et vraie zone (14 % des pleines).
- L'assemblage dégrade la carte brute sur trois tranches : partielles 1-3 % (0,569 contre 0,618), pleines 6-15 % (0,769 contre 0,786),
  objets en plusieurs morceaux (0,751 contre 0,760).

Plafonds mesurés (en connaissant la réponse), PixelSeal : meilleur seuil par image 0,725 (+0,027) ; taille de la zone connue 0,714 (+0,015) ;
meilleure union de masques SAM pour les objets partiels 0,971 contre 0,792 (**+0,045** sur PixelSeal) ; SAM ne sert ni aux rectangles (0,53) ni
aux trous des pleines (0,54 à 0,62). Les rectangles ne sont pas la boîte englobante d'un objet SAM (IoU médiane 0,14).

## 2. Les deux autres filigranes, même mesure

| | PixelSeal | TrustMark | StegaStamp |
|---|---|---|---|
| Candidat 35, validation croisée (réserve) | 0,699 (0,672) | 0,809 (0,781) | 0,934 (0,923) |
| Pleines / partielles | 0,695 / 0,703 | 0,747 / 0,871 | 0,937 / 0,930 |
| Pleines : carte brute au seuil 0 | 0,695 | 0,756 | 0,945 |
| Pleines famille A : mIoU, part de la perte | 0,647, 31 % | 0,707, 42 % | 0,921, 32 % |
| Objets partiels : candidat → plafond SAM (gain sur le filigrane) | 0,792 → 0,971 (+0,045) | 0,813 → 0,951 (+0,035) | 0,930 → 0,942 (+0,006) |
| Objets en plusieurs morceaux | 0,751 | 0,740 | 0,886 |
| Rectangles, petit côté < 64 px | 0,530 (30 images) | 0,840 (14) | 0,907 (27) |
| Cases fausses à 2 cases ou moins de la frontière | 32 % | 40 % | 76 % |
| Score si ces cases étaient corrigées | 0,785 (+0,086) | 0,883 (+0,074) | 0,982 (+0,048) |
| Meilleur seuil par image | +0,027 | 0,000 | +0,002 |

Deux constats valent pour les trois : la précision du contour (en moyenne +0,07 par filigrane au plafond) et, pour les pleines, un assemblage
qui fait un peu moins bien que la carte brute (TrustMark −0,009, StegaStamp −0,008 ; StegaStamp trou < 1 % : 0,879 contre 0,911).

## 3. Quatre tests courts faits pendant ce tour

| Test | Résultat | Conséquence |
|---|---|---|
| **Signal dans un rectangle PixelSeal selon sa taille** (`psphys.py` : vrai encodeur, mêmes boîtes que les 43 vrais rectangles) | Petit côté < 32 px : le décodeur relit 0,53 des bits sur le réel et 0,52 sur le fabriqué, mais **43 % des pixels sont modifiés, écart RMS 0,94 niveau** (image entière : 62 %, 1,14). 32-64 px : 0,68 / 0,73, 45 % | Le filigrane n'a pas disparu des petits rectangles : il est illisible pour le décodeur, pas absent. La recette fabriquée reproduit le réel taille par taille |
| **Canal 0 « détection » du détecteur PixelSeal** (`canal0.py`, entrées 256 / 512 / 1024) | AUC pleines 0,52 / 0,56 / 0,53, partielles 0,48 / 0,49 / 0,51 ; valeur moyenne identique pleines et partielles | Canal non entraîné : piste fermée |
| **Contraste luminance / chrominance** (`chroma.py` : le résidu PixelSeal est gris, il ajoute de l'énergie haute fréquence à (R+V+B)/3 et aucune à R−V, V−B) | AUC 0,566 (témoin sur images sans PixelSeal : 0,49) ; petits rectangles 0,52 | Signal réel mais faible devant le réseau (0,87 à 0,93) : inutile seul |
| **Image d'origine dans COCO val2017** (`cocomatch.py`, vignettes 24×24, recadrage centré ou non) ; **masques répétés** (`dupmask.py`) | Plus proche voisin : similarité médiane 0,63, aucune image ≥ 0,95 (2 sur 1020 au test) ; aucun jumeau train / test ni train / train ; aucun masque répété (0 paire ≥ 0,9) | Pas de val2017 ; `train2017` (118 000 images, 18 Go) non testé, voir X1 |

Recette de fabrication des rectangles de t1-reel (`reel7.py` lignes 101-111, arguments de `g_x1` PixelSeal) : aire log-uniforme 0,3 à 45 %,
`psmall = 0`. Écart avec le réel : 70 % des vrais rectangles font moins de 3 % de l'image (46 % en fabriqué) et 23 % sont des bandes de rapport
≥ 8,7 (12×116, 132×12, 148×12, 12×172, 32×456… ; 11 % en fabriqué).

## 4. Inventaire : ce qui n'a jamais été essayé

Inventaire complet (rapports `t1-*`, issue #3 jusqu'à 3 h 15, PR 34 et 35) fait par un agent séparé ; je ne reprends ici que les pistes
sans mesure. « Fermée » = mesurée pendant ce tour.

**PixelSeal**
| # | Piste | Proposée par | État |
|---|---|---|---|
| 1 | Canal 0 du détecteur | `t1-tour1.md` §2 | fermée (§3) |
| 2 | Carte JND (amplitude attendue du résidu) en entrée ou pour masquer la perte ; chrominance témoin | `t1-pixelseal-tour1.md` §d | chrominance seule : fermée (§3) ; JND en entrée ou en masque de perte : jamais essayée (négatif sur TrustMark) |
| 3 | Détecteur vraiment local (rognages 96-160 px, sans contexte global) | `t1-pixelseal-tour1.md` n° 4 | jamais essayée |
| 4 | Test physique des petites zones sur fabriqué (AUC et IoU par taille) | `t1-pixelseal-tour2.md` §a | jamais essayée ; §3 en fait la moitié (le signal existe) |
| 5 | Seuils en quantile ou température (réglés sur un modèle, appliqués à une moyenne) | `t1-pixelseal-tour2.md` §d1 | jamais essayée |
| 6 | Décodage masqué des candidats SAM (extérieur flouté) | `t1-pixelseal-tour2.md` §c | jamais essayée (le sens inverse : AUC 0,66 / 0,80) |
| 7 | Spécialiste pleines : trous alignés sur le contenu, perte masquée où le résidu est nul | `t1-pixelseal-tour2.md` n° 4 | jamais essayée sur PixelSeal |
| 8 | Pseudo-étiquetage du test | PR 35 | jamais essayée sur PixelSeal (négatif sur TrustMark) |
| 9 | Coller `embed(x, message)` dans une région candidate et lire la marge | PR 35 | jamais essayée |
| 10 | Tête résiduelle sur les features du décodeur à 512 px, tronc figé | Yassir | jamais essayée |
| 11 | Paires contrefactuelles sur PixelSeal | Yassir | lancée sur TrustMark seulement |
| 12 | Mélange appris des cartes sur les mêmes plis | Yassir, 3 h 09 | jamais essayée |
| 13 | Sur-tirage des petits rectangles et des bandes fines à l'entraînement | ce tour | jamais essayée (`psmall = 0`) |

Laissé en plan par la session t1-pixelseal (archivée) : test physique des petites zones ; sélection SAM étalonnée (« il reste de la marge dans
le choix des masques », plafond 0,98 contre 0,87) ; spécialiste pleines ; correction des seuils modèle seul / moyenne.

**TrustMark, StegaStamp, commun** : détecteur de boîte livré par Adobe (`loadBBoxDetector`) ; TruFor / Noiseprint++ en canal d'appoint ;
`wmforger` pour estimer le résidu ; exemples à zones générées (LaMa, diffusion) ; réseau unique pour les trois filigranes ; décision par
composantes (gradient boosting) ; rectangle optimal par Dinkelbach ; recadrage à l'inférence par la marge de toile ; force réduite pour
StegaStamp ; `padcal.py stegastamp` (prêt, non exécuté).

## 5. Expériences proposées

Classées par gain espéré sur le score T1 global (chaque filigrane pèse un tiers). La décision se lit sur la validation croisée des 510 images.

**PixelSeal (et retombées sur les deux autres)**
| # | Expérience | Pourquoi (mesure) | Coût | Chiffre qui décide | Gain espéré, global |
|---|---|---|---|---|---|
| P1 | **Choix des masques SAM pour les objets partiels, appris en commun sur les trois filigranes** : un score par masque (logit moyen, part de cases > 0, aire, contraste avec l'anneau voisin), classifieur entraîné sur les 124 objets partiels des trois filigranes en 5 plis ; chaque masque décidé seul, pas d'union gloutonne | Plafond SAM : PixelSeal 0,792 → 0,971, TrustMark 0,813 → 0,951 ; objets en plusieurs morceaux à 0,75 / 0,74 / 0,89 | **Mesurée à 5 h 30 par t1-assemblage, négative** (`rank3.py`, chiffres non recalculés par moi) : sur les objets hors réserve, union de masques jugés seuls 0,858 / 0,807 / 0,933 contre 0,910 / 0,813 / 0,946 pour le vote actuel à un masque (cartes du candidat 36) ; l'union détruit les rectangles | fermée ; le reste à gagner est dans le choix du masque TrustMark (0,813 contre un plafond de 0,932) | 0 |
| P2 | **Supervision sous la case** : sortie et perte à 512 px avec les masques PNG 512, puis règle officielle (case filigranée si ≥ 50 % de ses 16 pixels) ; perte doublée dans la bande de 8 px autour de la frontière. À sauter si c'est déjà le cas | 32 % / 40 % / 76 % des cases fausses sont à 2 cases ou moins de la frontière ; plafond +0,086 / +0,074 / +0,048 | 1 affinage par filigrane depuis `g_x1` | part des cases fausses à ≤ 2 cases ; mIoU hors-pli +0,01 par filigrane | +0,008 |
| P3 | **Petits rectangles PixelSeal** : (a) 15 min d'inférence : IoU rare de `f_x1` sur des rectangles FABRIQUÉS de petit côté 12-32, 32-64, 64-128 px ; (b) si (a) < 0,3 : affinage avec `psmall` 0,6 et bandes de rapport 8 à 16 tirées une fois sur quatre | 30 images, 28 % de la perte, IoU rare 0,10 alors que 43 % des pixels y sont modifiés ; la fabrication sous-représente ces tailles | (a) inférence ; (b) 1 affinage | IoU rare des vrais rectangles < 64 px, hors-pli : 0,10 → ≥ 0,25 | +0,006 |
| P4 | **Assemblage : garder la carte brute quand elle fait mieux**, selon l'aire PRÉDITE de la zone rare. **Mesurée à 5 h, nulle** (`hyb.py`, candidat 36, seuils d'aire choisis sans le pli de l'image) : PixelSeal 0,0000 (aucun seuil n'aide, même sans validation croisée), TrustMark −0,0019 [−0,005 ; +0,001], StegaStamp −0,0011 [−0,008 ; +0,003] | L'écart vu par tranche d'aire VRAIE ne se retrouve pas par aire prédite | fait | fermée | 0 |
| P5 | **Détecteur de présence local sur vignettes** (tronc sans réduction de résolution dans les premières couches, vignettes 48-96 px, filigrané contre propre, y compris résidu de petit rectangle), passé en fenêtre glissante, ajouté comme source | Trous < 1 % : AUC 0,92 mais touchée 0,17 ; petits rectangles : AUC 0,73. Ne lancer que si `g_st` (passe-haut en première couche, en cours) ne bouge pas ces tranches | 1 entraînement court | AUC des zones < 3 % hors-pli : pleines ≥ 0,95, partielles ≥ 0,85 | +0,004 |

**Hors des réponses évidentes**
| # | Expérience | Pourquoi | Coût | Ce qui décide |
|---|---|---|---|---|
| X1 | **Retrouver l'image d'origine** : relancer `cocomatch.py` contre COCO `train2017` et `unlabeled2017`. Si l'origine est trouvée, la différence image − origine donne le résidu exact : les pixels propres sont identiques à l'origine au niveau près, ce qui vérifie aussi la chaîne de redimensionnement | Les contenus (bus, zèbre, cerf-volant, horloge) ressemblent à COCO ; val2017 est exclu (§3). Si ça marche, les trois filigranes montent d'un coup | Téléchargement de 18 Go (accord d'Adem requis : téléchargement et lecture du règlement sur les données externes), puis 20 min | part des images du train avec un voisin ≥ 0,95 |
| X2 | **Trois sondes publiques par filigrane** : candidat 35 où l'on remet un seul filigrane du candidat 27, à tour de rôle | Le public ne donne qu'un total ; l'écart prévu / réel du 35 (+0,025 / +0,010) vient d'un filigrane qu'on ne connaît pas | 3 créneaux d'envoi, aucun calcul | l'écart public de chaque sonde contre 0,8033 |

**TrustMark**
| # | Expérience | Pourquoi | Chiffre qui décide |
|---|---|---|---|
| T1 | **Second étage entraîné sur fabriqué** : un petit U-Net reçoit l'image et la carte du premier étage et rend le masque (les exemples fabriqués sont illimités) ; affinage 5 plis | Pleines = 66 % de la perte TrustMark, famille A 42 % (IoU du trou 0,46), 40 % des cases fausses près du bord ; le correcteur de Yassir (appris sur 30 images) avait un plafond de +0,112 | IoU du trou des pleines hors-pli 0,535 → ≥ 0,58 |
| T2 | P1 et P2 appliqués à TrustMark | Objets partiels 0,813 → plafond 0,951 ; bords +0,074 au plafond | voir P1, P2 |

**StegaStamp (en dernier)** : 76 % des cases fausses sont à 2 cases ou moins de la frontière et le score monterait à 0,982 si elles étaient
justes : seule P2 compte. Ensuite les objets en plusieurs morceaux (22 images à 0,886, 22 % de la perte), couverts par P1.

Ce que je ne recommande pas de relancer : décodeur en fenêtre glissante (0,47 à 0,51), filtre adapté (AUC 0,50), carte par position du
décodeur (AUC 0,54 à 0,71), ensemble de plis (aucun gain), seuil par image (+0,027 au plafond), SAM sur les trous des pleines (0,54 à 0,62).

### État des expériences à 5 h 40 (chiffres des sessions qui les ont faites, non recalculés par moi sauf mention)
- **P1** : fermée, négative (t1-assemblage). **P4** : fermée, nulle (ma mesure, `hyb.py`).
- **P3a** (t1-reel, `p3a.py`) : sur des rectangles FABRIQUÉS, les 5 modèles `f_x1` PixelSeal font 0,095 d'IoU de la zone pour un petit côté de 10-32 px
  (80 % jamais touchés), 0,254 pour 32-64 px, 0,500 pour 64-128 px, 0,789 au-delà. Le réseau ne voit donc pas non plus les petits rectangles
  fabriqués : ce n'est pas un écart entre fabriqué et réel. **P3b** lancée (`f_x1r`, 60 % de petits rectangles dont un quart de bandes).
- **P2** : la sortie est déjà à 512 px et la perte directe à 512 a déjà été mesurée sans gain ; seule la perte doublée près de la frontière
  est nouvelle, lancée sur TrustMark (`f_r50e`). **T1** et **P5** : pas encore lancées.
- **X1** : **mesurée à 6 h, négative, fermée** (`cocomatch2.py`, 118 287 images de COCO `train2017`, quatre cadrages par image). Plus proche
  voisin : similarité médiane 0,70 (train) et 0,69 (test) ; ≥ 0,95 pour 1 image sur 510 et 8 sur 1020 ; pleines ≥ 0,90 : 1,2 %. Pour ces
  quelques cas le deuxième voisin est aussi proche que le premier (0,951 contre 0,951 ; 0,984 contre 0,981) et la planche des 12 meilleures
  paires (`cocoview.jpg`) montre des photos différentes de même composition (ciel, piste, pelouse). Les images du hackathon ne viennent pas
  de COCO `train2017` ni `val2017` ; plusieurs ont l'allure d'images générées. Aucun fichier de test n'a été produit. **X2** : accordée,
  fichiers demandés à t1-assemblage.

## 6. Règle de feu vert après le candidat 35

Prévision contre public, écart apparié contre l'envoi précédent : 22 : +0,032 / +0,031 ; 27 : +0,018 / +0,024 ; 30 : +0,003 / −0,0004 ;
35 : +0,025 / +0,010. Le bruit d'un écart lu sur 306 images vaut environ 0,005 : le 35 est à 3 erreurs types sous sa prévision.
En niveau, la validation croisée reste juste (0,8138 contre 0,8033 public, bruit du public ±0,011 en écart-type).
**Avis : la règle ne change pas** (≥ +0,02 sur les 510 images en validation croisée, IC95 hors de 0, signe par filigrane). J'annonce
désormais le gain public attendu entre 0,4 et 1 fois l'écart mesuré. Les sondes X2 diront quel filigrane ne transfère pas.

### Complément de 8 h : pistes mesurées après le candidat 38 (public 0,8230)
- **Tache autour du maximum pour les partielles sans recouvrement** (`vide.py`, candidat 38, règle choisie sans le pli de l'image) : négatif.
  PixelSeal −0,0015 [−0,004 ; 0,000], TrustMark −0,0009, StegaStamp −0,0050. Constat utile : 24 partielles PixelSeal sur 85 n'ont aucun
  recouvrement entre zone prédite et vraie zone (score 0,479, contre 0,872 pour les autres) ; 7 pour TrustMark, 3 pour StegaStamp.
- **Mélange appris des 5 cartes PixelSeal** (piste n° 12, `melange.py`, régression logistique par case ou arbres boostés, 5 plis par image) :
  nul. Au mieux +0,0020 [−0,006 ; +0,010] contre la composition choisie à la main ; la moyenne simple des 5 fait −0,0098.
- **Boîte par décodeur à seuil abaissé** (`boitez.py`) : z = (bits justes − 0,5) × 32. Bruit : z maximal de 600 boîtes hors zone, médiane 3,0,
  maximum 4,2. À la vraie boîte (±4 px) : rectangles de 32-64 px, z ≥ 6 pour 7 sur 12 (z ≥ 12,8 pour 2) ; 64-128 px, z ≥ 8 pour 10 sur 10 ;
  < 32 px, aucun. Le seuil actuel de l'assemblage est z ≥ 12,76. Piste donnée à t1-assemblage : recherche fine amorcée par la carte, seuil
  choisi en validation croisée entre 5 et 13. Gain espéré +0,02 sur PixelSeal (estimation).
- **Nouvelle carte PixelSeal** `ps_pop_x2+x2b+x1+dx__st+x2` (modèle à filtres passe-haut appris) : +0,0183 [+0,007 ; +0,030] au seuil 0,
  petits rectangles 0,094 → 0,244 d'IoU de zone ; c'est le seul gain PixelSeal mesuré de la nuit côté cartes.
