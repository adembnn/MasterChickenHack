# T1 · vérification indépendante (session `t1-verificateur`)

Passage 1, 21/11 ~20 h. Décision débloquée : sur quels chiffres s'appuyer pour choisir la prochaine soumission T1, et quels candidats refuser.
Code : `E:\EAC-lab\t1s\t1-verificateur\` (`verif.py`, `decomp.py`, `ens.py`, `dups.py`, `jpegtest.py`, `xmeth.py`, `padtest.py`), copies sur Jülich
`adem/t1-verificateur/` et sur `h200b:~/eac/t1/t1-verificateur/`. Aucun code des fabricantes n'est importé.

Termes (une fois) : **mIoU** = score officiel, par image la moyenne de deux IoU (intersection sur union), une pour la classe « propre », une pour la
classe « filigranée » ; **réserve** = images du train d'index global `% 5 == 0` (30 PixelSeal, 39 TrustMark, 33 StegaStamp), jamais apprises ;
**hors-pli** = prédiction d'une image par un modèle qui ne l'a pas vue (validation croisée) ; **IC95** = intervalle à 95 % obtenu en retirant
10 000 fois les images au hasard avec remise (rééchantillonnage) ; **écart apparié** = différence de deux méthodes calculée image par image
sur les mêmes images, dont l'intervalle est bien plus étroit que celui de chaque score ; **classe minoritaire** = le trou propre d'une image
pleine, ou la zone filigranée d'une image partielle.

## 0. Mon évaluateur
Score réécrit depuis la définition de `kit/wmloc/utils.py`, contrôlé contre une version naïve par ensembles (6 cas, dont tout-0 et tout-1).
Vérité : `masks_128.npz` comparé aux PNG 512 que je rééchantillonne moi-même (règle ≥ 50 % par case 4×4) : 0 image différente sur 510.

## 1. Base soumise (soumission 1227, 0,697 public) : REPRODUITE
| Filigrane | Annoncé | Recalculé (réserve) | IC95 | pleines / partielles | Hors-pli (n) | Verdict |
|---|---|---|---|---|---|---|
| StegaStamp, `mixr` 5 plis moyennés | 0,820 | **0,8197** (n=33) | [0,760 ; 0,873] | 0,822 / 0,817 | 0,804 (137) [0,777 ; 0,830] | conforme |
| TrustMark, `mixr` 5 plis moyennés | 0,597 | **0,5968** (n=39) | [0,548 ; 0,646] | 0,611 / 0,585 | 0,629 (131) [0,603 ; 0,656] | conforme |
| PixelSeal, `run_D_res512` | 0,622 | **0,6218** (n=30) | [0,567 ; 0,679] | 0,655 / 0,565 | pli tenu à l'écart 0,631 (36) | conforme |
| Moyenne des trois | 0,679 | **0,6794** | [0,648 ; 0,711] | | | conforme |

- Plis : la réserve de chaque `res.npz` est exactement l'ensemble `index % 5 == 0` ; les 5 lots hors-pli sont disjoints et couvrent les 136 autres
  images ; les rôles de `run_D` (0 réserve, 1 pli, 2 appris) sont exacts. Les images apprises de `run_D` valent 0,864 : ne jamais les compter.
- Fichier `submission_v4.pt` : 1020 masques booléens, identiques aux prédictions test seuillées à 0 que je recalcule (0 image différente).
- Relecture de `reel.py` : fonds, banque de formes et exemples « alignés » excluent la réserve et le pli tenu à l'écart ; nombre de pas fixe
  (2500), seuil fixe à 0, pas de choix de la meilleure époque. Pas de fuite trouvée.
- Doublons : vignettes 24×24 des 1530 images, corrélation maximale 0,92, aucune paire au-dessus de 0,95, aucune paire réserve / hors réserve
  au-dessus de 0,9, 1530 messages distincts. Pas de source partagée entre réserve, train et test.

## 2. Biais trouvés
1. **Candidat 3 de `t1-assemblage` : NO-GO.** Annoncé 0,633 sur la réserve ; recalculé depuis `controle_train_3.pt` : **0,6377** (PixelSeal 0,576,
   TrustMark 0,573, StegaStamp 0,764 contre 0,756 annoncé). Écart apparié contre la base : −0,046 / −0,024 / −0,055, les trois significatifs.
   Cause : `assemble.py` choisit la source par validation croisée sur les images hors réserve, or les sources `seg_v2*` ont été APPRISES sur ces
   images (hors réserve 0,87 à 0,90 ; réserve 0,56 à 0,75). L'assembleur préfère donc `seg_v2unet_stegastamp` (réserve 0,748) à `reel/mixr` (0,820).
   Correction : n'admettre au choix que des prédictions réellement hors-pli (`oof.npz` de t1-reel, rôle ≠ 2 de t1-pixelseal), ou choisir sur la réserve.
2. **Le choix de `run_D_res512` pour PixelSeal s'est fait sur la réserve.** Réserve : D 0,622, A 0,604, C 0,598. Pli tenu à l'écart (36 images) : A 0,658,
   C 0,656, D 0,631. Sur les 66 images non apprises : A 0,634, C 0,629, D 0,627, écarts non significatifs. « L'entrée en 512 fait gagner 0,02 »
   n'est pas établi. Le niveau du modèle (≈ 0,63) est bon ; la base n'est pas remise en cause.
3. **`run_F_base` (entraîné sur pleines) et `run_P_base` (sur partielles)** ne valent que sur leur population : F pleines 0,644 (D : 0,655), P partielles
   0,598 sur la réserve et 0,674 sur le pli (D : 0,565 et 0,630). Seul ou mal aiguillé, chacun est significativement sous D (−0,06 à −0,08).
   L'aiguillage par le drapeau pleine/partielle (bits justes de l'image entière) doit être noté tel qu'il sera au test.

## 3. Ce qui est significatif et ce qui ne l'est pas
- Un score seul sur la réserve d'un filigrane a un IC95 de ±0,05 à ±0,06 ; la moyenne des trois ±0,03.
- **Le score public porte sur 306 images : son écart-type vaut 0,011** (écart-type par image 0,188). 0,697 contre 0,69 et 0,68 n'est donc pas
  une avance établie, et la note finale (714 images cachées, écart-type 0,007) peut réordonner ces trois équipes.
- Un écart apparié est bien plus précis (±0,01 à ±0,02 sur la réserve, ±0,01 hors-pli). Variantes de t1-reel, hors-pli contre `mixr` :
  `mix` −0,000 (TM) et +0,006 (SS), non significatif ; `flag` −0,026 (TM), significatif ; `noaug` −0,023 (TM) et −0,032 (SS), significatif ;
  `syncoco` −0,115 et −0,189. Les chiffres annoncés par t1-reel pour `mix`, `flag`, `noaug` se reproduisent (0,830/0,589 ; 0,817/0,577 ; 0,807/0,592).
- Règle proposée : comparer toute nouveauté à la base par écart apparié sur réserve + hors-pli, et ne retenir que si l'IC95 exclut 0.

## 4. Cohérence train / test sans vérité (base 1227)
| Filigrane | Part de cases filigranées, train non appris / test | Images majoritairement filigranées | Prédictions tout-0 | Prédictions tout-1 |
|---|---|---|---|---|
| PixelSeal | 0,524 / 0,497 | 0,545 / 0,503 | 0,045 / 0,026 | 0,015 / 0,006 |
| TrustMark | 0,493 / 0,484 | 0,494 / 0,488 | 0,147 / 0,171 | 0,076 / 0,076 |
| StegaStamp | 0,485 / 0,486 | 0,500 / 0,500 | 0,082 / 0,088 | 0,006 / 0,009 |
Distributions d'aires et nombres de composantes proches entre train et test : aucun signe de rupture de généralisation. Seul écart : les prédictions
StegaStamp des images majoritairement filigranées sont moins remplies au test (10e centile 0,715 contre 0,807), à surveiller.
Fait utile : **aucune des 510 vérités n'est tout-0 ni tout-1**. Une prédiction à une seule classe plafonne à ~0,47 ; TrustMark en rend 25 % au test.

## 5. Où est l'erreur restante (images non apprises, base 1227)
Le score d'une image vaut environ 0,5 + 0,5 × IoU de la classe minoritaire (la classe majoritaire est à 0,92-0,98).
| Filigrane | Type | n | mIoU | IoU classe minoritaire | Classe minoritaire absente de la prédiction | Manque (points du filigrane) |
|---|---|---|---|---|---|---|
| PixelSeal | pleine | 36 | 0,644 | 0,35 | 3 % | 0,194 |
| PixelSeal | rectangle | 11 | 0,542 | 0,13 | 18 % | 0,076 |
| PixelSeal | objet | 19 | 0,643 | 0,34 | 5 % | 0,103 |
| TrustMark | pleine | 85 | 0,604 | 0,30 | 15 % | 0,198 |
| TrustMark | rectangle | 43 | 0,643 | 0,34 | 37 % | 0,090 |
| TrustMark | objet | 42 | 0,636 | 0,35 | 21 % | 0,090 |
| StegaStamp | pleine | 85 | 0,835 | 0,70 | 1 % | 0,082 |
| StegaStamp | rectangle | 43 | 0,773 | 0,56 | 21 % | 0,058 |
| StegaStamp | objet | 42 | 0,786 | 0,60 | 12 % | 0,053 |
Lecture : le plus gros gisement est le **trou propre des images pleines TrustMark et PixelSeal** (0,20 point chacun, soit 0,13 sur la moyenne des trois),
puis les rectangles TrustMark (37 % non trouvés). StegaStamp n'a plus que 0,19 à gagner au total. L'erreur est bimodale : 36 % (TrustMark) et 44 %
(PixelSeal) des images ont la classe minoritaire ratée (IoU < 0,1) et portent 50 à 60 % du manque.

## 6. Petits gains gratuits mesurés (moyenne de logits entre runs déjà déposés)
- PixelSeal A + D contre D : +0,013 sur 66 images non apprises, IC95 [+0,002 ; +0,026] (réserve seule −0,001, pli +0,025).
- TrustMark `mixr` + `mix` : +0,007 hors-pli [+0,001 ; +0,014], réserve −0,003. StegaStamp `mixr` + `mix` : +0,007 réserve et hors-pli.
Gain attendu ≤ +0,01 sur la moyenne : à prendre à l'assemblage, pas une priorité.

## 7. Nouveaux dépôts vérifiés (non encore annoncés sur l'issue)
- **`p_mixr` de t1-reel (recette « toile agrandie », 4 plis finis sur 5)** : StegaStamp réserve **0,8431** contre 0,8197, écart apparié +0,023
  IC95 [+0,005 ; +0,047], et hors-pli +0,037 [+0,019 ; +0,056] : gain établi. TrustMark réserve 0,5978 (+0,001, non significatif), hors-pli +0,018
  [+0,001 ; +0,037] : gain faible. PixelSeal par ce U-Net : 0,510, très sous `run_D`. La découverte de la toile se confirme surtout sur StegaStamp.
- **PixelSeal, aiguillage par population** (drapeau = bits justes de l'image entière > 0,75, exact à 96 %, donc tel qu'au test) : `run_D` pour les
  pleines + `run_P_base` pour les partielles = 0,645 contre 0,627 sur 66 images non apprises, écart +0,018 IC95 [+0,007 ; +0,032], positif sur la
  réserve (+0,012) comme sur le pli (+0,023). Avec (A + D)/2 pour les pleines : +0,027 [+0,011 ; +0,044] (une variante parmi quatre essayées).

## 8. Regard neuf : tests courts faits par moi, puis pistes proposées
Faits lus dans la consigne officielle (`reports/sources/Watermark_Localization.txt`), non cités jusqu'ici : (a) le filigrane est posé « sur un
sous-rectangle de l'image d'origine » : c'est l'origine de la toile plus grande que l'image ; (b) les zones propres sont du contenu généré ou greffé
(outpainting, retrait ou ajout d'objet, greffe d'objet), pas des pixels d'origine.

Tests négatifs (pistes à ne pas ouvrir) :
- **Trace d'édition commune aux trois filigranes** (`xmeth.py` : segmenteur appris sur les vraies images de deux filigranes, noté sur le troisième) :
  TrustMark jamais vu 0,31, PixelSeal jamais vu 0,27, sous la constante (0,45) ; AUC par image médiane 0,61 et 0,55. Le réseau lit le filigrane,
  pas une trace de retouche : mettre en commun les vraies images des trois filigranes n'apporterait rien. (StegaStamp non mesuré : carte saturée.)
- **Grille JPEG 8×8** (`jpegtest.py`) : aucun pic de phase (1,006) : les sources ne gardent pas de grille JPEG, pas de détecteur par bloc.
- **Estimateur du cadre par image avec la marge du décodeur, TrustMark** (`padtest.py`, `padcal.py`) : sur des images fabriquées dont le bord de toile
  est connu, l'argmax de la marge retrouve le bord dans 27 à 34 % des cas (hasard : 25 %), erreur moyenne 8 à 10 px. Le décodeur TrustMark tolère trop
  les décalages : on ne peut pas dire par cette voie si le rognage est centré. Sur les vraies pleines la marge passe de 7,74 à 8,07 au mieux.
  À noter : les vraies pleines TrustMark ont une marge de 7,4 quand mes fabriquées (toile +8 ou +16 px, fonds réels, sans trou) ont 9,2 : il reste un
  écart réel / fabriqué pour TrustMark, cohérent avec le gain faible de `p_mixr`. Test pour t1-reel : fabriquer avec les vrais trous et comparer la marge.
- **Seuil par image** : même le meilleur seuil choisi image par image avec la vérité ne donne que 0,665 (TrustMark) et 0,657 (PixelSeal) contre
  0,622 et 0,627 : le post-traitement ne peut pas combler l'écart, c'est la carte qu'il faut améliorer (borne approximative pour les très petites zones).

Fait mesuré qui oriente (`area.py`, images non apprises) : **l'erreur vient des petites zones minoritaires.**
| Aire de la classe minoritaire | TrustMark : n, IoU min, absente | PixelSeal : n, IoU min, absente | StegaStamp : n, IoU min, absente |
|---|---|---|---|
| < 1 % de l'image | 23 ; 0,02 ; 74 % | 10 ; 0,01 ; 20 % | 24 ; 0,46 ; 29 % |
| 1 à 3 % | 30 ; 0,14 ; 40 % | 16 ; 0,16 ; 6 % | 35 ; 0,43 ; 11 % |
| 3 à 8 % | 48 ; 0,29 ; 15 % | 11 ; 0,24 ; 0 % | 52 ; 0,68 ; 6 % |
| 8 à 20 % | 44 ; 0,48 ; 5 % | 20 ; 0,53 ; 5 % | 45 ; 0,78 ; 2 % |
| > 20 % | 25 ; 0,61 ; 0 % | 9 ; 0,51 ; 0 % | 14 ; 0,86 ; 0 % |
Les zones de moins de 3 % (un carré de moins de 90 px de côté) font 31 % des images TrustMark et 39 % des images PixelSeal, et portent 39 % et 48 %
du manque. Chaque image pèse autant, quelle que soit la taille de sa zone.

Pistes proposées, que personne ne suit, avec le test qui tranche :
1. **Inférence zoomée pour les petites zones (TrustMark et PixelSeal d'abord).** Raison physique : dans un rectangle d'outpainting, tout le motif
   (calculé en 256 px) est comprimé dans le rectangle ; le décodeur TrustMark relit pourtant ces petites boîtes à 0,94 parce qu'il agrandit le rognage
   à 245 px. Le U-Net, lui, voit le motif à une échelle 2 à 8 fois plus fine que dans les pleines et sa première couche (pas de 4) l'efface.
   MESURÉ au passage 2 (§ 12) : vrai pour les rectangles (+0,06 à +0,11), faux pour les objets et les pleines (−0,17 et −0,55) ; plafond +0,017
   sur TrustMark avec un tri rectangle / objet parfait. Piste déclassée.
2. **Aiguillage PixelSeal D + P** (mesuré, § 7) : dépassé par la fusion v1 de t1-pixelseal, envoyée dans v5. **StegaStamp `p_mixr`** : envoyé dans v6.
3. **Accepter ou refuser par écart apparié** (`verif.py`), pas par comparaison de deux scores de réserve : précision ±0,01 à ±0,02 au lieu de ±0,05.
4. **Estimateur de cadre pour StegaStamp** (son décodeur ne répond qu'à ±4 px, donc l'étalonnage peut réussir là où TrustMark échoue) :
   `padcal.py stegastamp 24` est prêt, non exécuté faute de carte. S'il retrouve le bord, on saura si le rognage réel est centré.

## 9. Verdicts go / no-go
| Candidat | Réserve recalculée | Écart apparié contre la base | Verdict |
|---|---|---|---|
| Soumission 1227 (base) | 0,679, IC95 [0,648 ; 0,711] | référence | déjà soumise, chiffres conformes |
| Candidat 3 de t1-assemblage | 0,638 | −0,024 à −0,055 par filigrane, significatif | **NO-GO** |
| StegaStamp → `p_mixr` (5 plis à finir), reste inchangé | StegaStamp 0,843 (4 plis) | +0,023 [+0,005 ; +0,047] | **GO** dès le 5e pli, après re-vérification |
| PixelSeal → D pleines + P partielles | 0,634 réserve, 0,645 sur 66 | +0,018 [+0,007 ; +0,032] | **GO** si `pred_test` de `run_P` existe et si la part d'images classées partielles au test (≈ 49 %) est respectée |
| TrustMark → `p_mixr` | 0,598 | +0,001 (réserve), +0,018 (hors-pli) | neutre : sans risque, sans gain établi |
Risques communs : la réserve PixelSeal n'a que 30 images ; le public (306 images) a un écart-type de 0,011, donc un gain attendu de +0,014 peut ne
pas se voir au classement public tout en étant réel.

Non vérifié dans ce passage : les fichiers `pred_test` des runs A, F, P de t1-pixelseal ; le 5e pli de `p_mixr` ; `p_syn` ; la recherche de boîte de
t1-residus (aucune prédiction déposée) ; le test de transfert pour StegaStamp et l'étalonnage du cadre StegaStamp (carte `h200b` saturée à 141 Go).

---

# Passage 2, 21/11 ~21 h : règle « envoi solide », meilleur candidat hors ligne, zoom, TrustMark réserve contre validation croisée

Règle d'Adem (19 h 46) : un fichier T1 ne part que si la moyenne de réserve des trois filigranes gagne au moins +0,02 sur le dernier envoi, avec un
écart apparié dont l'IC95 exclut 0, et mon feu vert écrit. Jülich refuse ma connexion depuis ~20 h 40 (code MFA) : ce passage est calculé sur
`h200b` (`~/eac/t1/t1-verificateur/` : `cmpnpz.py`, `tmcv.py`, `zoom.py`, `zoomeval.py`). Non relus pour cette raison : le fichier `submission_v6.pt`
lui-même et les 5 plis TrustMark `p_mixr` de Jülich.

## 10. Dernier envoi (soumission 1364, v6, 0,7059 public) : composants reproduits
| Filigrane | Composant | Réserve recalculée | IC95 | 170 images (réserve + hors réserve) |
|---|---|---|---|---|
| StegaStamp | `p_mixr`, 5 plis (`t1-reel/pred_train_stegastamp.npz`) | 0,8417 | [0,785 ; 0,891] | 0,8343 |
| TrustMark | `mixr`, 5 plis | 0,5968 | [0,548 ; 0,646] | 0,6218 |
| PixelSeal | fusion v1 (`t1-pixelseal/pred_train_pixelseal.npz`) | 0,6362 | [0,581 ; 0,695] | 0,6580 |
| Moyenne | | **0,6916** | | 0,7047 |

La fusion PixelSeal v1 contre `run_D` vaut +0,015 sur la réserve, IC95 [−0,009 ; +0,043] : gain non établi à lui seul (le public a donné +0,004).

## 11. Meilleur candidat hors ligne par filigrane
| Filigrane | Meilleur dépôt vérifié | Réserve | Écart apparié contre v6, réserve | Écart apparié, 170 images | État |
|---|---|---|---|---|---|
| StegaStamp | `p_mixr` (déjà dans v6) | 0,8417 | 0 | 0 | vote de formes sur `mixr` = `p_mixr` seul (−0,005, non significatif) : à refaire SUR `p_mixr` |
| TrustMark | vote de formes SAM, modèle commun (`rangP-reel-mixr`, t1-assemblage) | **0,6223** | **+0,026 [+0,005 ; +0,055]** | +0,024 [+0,013 ; +0,037] | gain établi |
| PixelSeal | fusion v1 (déjà dans v6) | 0,6362 | 0 | 0 | `pred_train_G` : −0,015, écarté |
| **Moyenne des trois** | | **0,7001** | **+0,0085 [+0,0015 ; +0,018]** | +0,0081 [+0,004 ; +0,012] | **sous le seuil de +0,02 : pas de feu vert** |

Vote de formes (`rank.py`) relu : le modèle qui choisit un masque SAM est appris en 4 plis hors réserve, la réserve est prédite par un modèle qui
ne l'a jamais vue, le seuil `tau` est réglé hors réserve, la carte de départ est hors-pli. Pas de fuite trouvée. Statistiques train / test des
prédictions proches (prédictions tout-0 : 0,6 % des deux côtés, contre 15 à 17 % avant le vote).
Écarté : `p_dec` TrustMark (tronc du décodeur) = 0,549 sur la réserve, −0,047 [−0,089 ; −0,012] contre `mixr`, −0,030 hors-pli : significativement pire.
Il manque +0,035 (somme des trois filigranes) pour atteindre le seuil. Sources plausibles, par rendement attendu : (a) le vote de formes refait sur
`p_mixr` pour StegaStamp (sur `mixr` il valait +0,017 en réserve et +0,027 hors-pli) et pour TrustMark ; (b) un vote de formes pour les objets
PixelSeal (`rank.py` ne traite que TrustMark et StegaStamp) ; (c) la boîte du décodeur sur les rectangles TrustMark (aucun gain mesuré à ce jour).

## 12. Inférence zoomée : mesurée, gain limité aux rectangles, piste déclassée
Tuiles de 256 px (pas 128) agrandies à 512 = Z2, tuiles de 128 px (pas 64) = Z4, logits remis sur la grille, recouvrements moyennés. Modèles existants,
sans réentraînement ; hors-pli = modèle qui n'a pas vu l'image, réserve = moyenne des modèles. Contrôle : mes logits normaux égalent ceux sauvés (écart max 0,007).
| TrustMark, écart apparié contre l'inférence normale | `p_mixr` U-Net (2 plis, 91 images) | `p_dec` (5 plis, 170 images) |
|---|---|---|
| Rectangles, moyenne(normal, Z2, Z4) | **+0,061 [+0,017 ; +0,106]** (n=25) | **+0,114 [+0,071 ; +0,162]** (n=43) |
| Objets, même fusion | −0,170 [−0,242 ; −0,104] | −0,158 [−0,207 ; −0,111] |
| Images classées pleines, Z2 seul | −0,54 | −0,56 |
| Partielles de zone < 3 %, max(normal, Z2) | rectangles < 3 % : −0,004 | −0,012 [−0,036 ; +0,016] |
| Toutes, zoom sur les images classées partielles, max(normal, Z2) | +0,002 [−0,009 ; +0,012] | +0,006 [−0,005 ; +0,018] |
| Toutes, zoom seulement sur les VRAIS rectangles (tri parfait) | +0,017 [+0,004 ; +0,031] | non calculé |

Lecture : l'hypothèse était à moitié juste. Le zoom aide les rectangles (motif entier comprimé dans la boîte), quelle que soit leur taille, mais il
casse les objets et les pleines, dont le motif est à l'échelle de l'image. Ce n'est donc pas un remède aux petites zones en général. Avec un tri
rectangle / objet parfait le gain plafonne à +0,017 sur TrustMark, soit +0,006 sur la moyenne ; la recherche de boîte par le décodeur vise les mêmes
images avec un plafond plus haut (0,90 quand la boîte est rendue). Je ne pousse pas le zoom plus loin ; il reste un appoint possible sur les images
où la recherche de boîte dit « rectangle » sans rendre de boîte fiable.

## 13. TrustMark : la validation croisée monte, la réserve non. C'est du bruit.
- 4 plis `p_mixr` de Jülich (§ 7) : gain en réserve +0,001 [−0,014 ; +0,015], gain hors-pli +0,018 [+0,001 ; +0,037]. La différence des deux gains
  vaut 0,017 pour un écart-type d'environ 0,012 : 1,5 écart-type, compatible avec le bruit.
- 2 plis `p_mixr` réentraînés sur `h200b` (mesure indépendante) : réserve +0,012 [−0,003 ; +0,027], hors-pli +0,022 [−0,001 ; +0,048], différence
  +0,009 [−0,018 ; +0,039]. Ici la réserve monte aussi.
- La réserve est prédite par une moyenne de modèles et le hors-pli par un seul : cet effet est nul sur le gain (−0,000 [−0,007 ; +0,005]).
- La réserve TrustMark est un peu plus dure que le reste : 38 % de zones minoritaires de moins de 3 % contre 29 %, niveau de la base −0,038
  [−0,092 ; +0,017] (non significatif). Repondérée à la composition des 170 images, elle vaudrait 0,602 au lieu de 0,597.

Conclusion : 39 images ne distinguent pas +0,00 de +0,02. Le gain réel de `p_mixr` sur TrustMark est d'environ +0,01 à +0,02.

Critère d'envoi proposé (à faire valider par Adem, je ne modifie pas `CLAUDE.md`) : garder le seuil de +0,02 sur la moyenne des trois filigranes, mais
le mesurer par **écart apparié sur les 510 images (réserve + hors-pli)**, la réserve servant de garde-fou (son IC95 doit contenir l'estimation des 510).
Précision : ±0,004 sur la moyenne au lieu de ±0,008. Condition : hors réserve, un composant n'est compté que s'il y est réellement hors-pli et si
aucun seuil n'a été réglé sur ces images ; sinon il est jugé sur la réserve seule (cas de la fusion PixelSeal v1, réglée sur 36 images).

---

# Passage 3, 21/11 ~21 h 45 : candidat 5 de t1-assemblage, FEU VERT

Fichier : `h200b:~/eac/t1/t1-assemblage/submission_candidat_5.pt`, sha256 `5319e7f6ee6a93877880a7878418777ee3ca48432c7e8c89ac4d111f4a0275d4`. Script : `cand5b.py`.

## 14. Recalcul (réserve, recette réglée hors réserve) contre v6
| Filigrane | Candidat 5 | pleines / partielles | v6 | Écart apparié | IC95 |
|---|---|---|---|---|---|
| PixelSeal (n=30) | 0,6483 | 0,647 / 0,650 | 0,6362 | +0,012 | [−0,011 ; +0,045] |
| TrustMark (n=39) | **0,6778** | 0,616 / 0,731 | 0,5968 | **+0,081** | [+0,009 ; +0,153] |
| StegaStamp (n=33) | 0,8429 | 0,817 / 0,875 | 0,8417 | +0,001 | [−0,011 ; +0,016] |
| **Moyenne des trois** | **0,7230** | | 0,6916 | **+0,0314** | **[+0,005 ; +0,058]** |

Annoncé par t1-assemblage : 0,6483 / 0,6778 / 0,8429, reproduit à la 4e décimale. Probabilité que le gain réel dépasse +0,02 : 0,80.
Le gain vient des partielles TrustMark : +0,146 [+0,014 ; +0,267] sur 21 images (13 mieux, 7 pires), par le vote de formes SAM (`rangP-reel-mixr`)
puis la boîte du décodeur (confiance `z0` = 5). Hors réserve (recettes réglées dessus, donc un peu optimiste) TrustMark gagne +0,082 : cohérent.

## 15. Recherche de fuite : aucune
- `assemble_5.json` : seuils, nettoyage et `z0` réglés hors réserve, sources choisies par validation croisée hors réserve.
- Le « seuil re-réglé sur les 170 » ne modifie que deux seuils du fichier final (TrustMark pleines −0,074 → −0,042 ; StegaStamp pleines −0,403 → −0,031).
  Le fichier de contrôle utilise ces seuils finaux (réserve 0,7271, optimiste). J'ai reconstruit les masques honnêtes : contrôle pour les populations
  inchangées, et (logit centré-réduit > seuil honnête) pour les deux autres ; ma reconstruction retrouve le contrôle à 100 % avec les seuils finaux.
- `bs_sh.py` (boîte du décodeur) : la recherche n'utilise que l'image et le message ; la vérité ne sert qu'aux diagnostics du train.
- Avec le seuil honnête les pleines StegaStamp perdent −0,017 (17 images sur 18) ; le seuil final revient près de celui de v6 et corrige ce point.

## 16. Fichier de test et cohérence train / test
1020 noms distincts égaux à ceux du test, forme [1020, 128, 128], uint8, valeurs 0/1. Part filigranée : PixelSeal 0,503 (v6 0,502), TrustMark 0,510
(0,484), StegaStamp 0,488 (0,489). Cases changées contre v6 reconstruite : 2,8 % / 5,3 % / 0,7 % ; images changées de plus de 5 % : 14 % / 30 % / 2 % au
test, 16 % / 32 % / 3 % au train. Boîte rendue : TrustMark 52 % des partielles au train et 54 % au test, PixelSeal 16 % et 21 %. Au train, 93 % des
boîtes TrustMark rendues sont de vrais rectangles (mIoU 0,84). Plus aucune prédiction tout-0. Limite : v6 est reconstruite depuis ses composants,
le `.pt` v6 étant sur Jülich.

## 17. Verdict
**FEU VERT** : +0,0314 sur la moyenne de réserve (seuil +0,02), IC95 de l'écart apparié [+0,005 ; +0,058] hors de 0, pas de fuite, fichier conforme.
Réserves : intervalle large (le gain repose sur 21 partielles TrustMark de la réserve) ; PixelSeal et StegaStamp ne sont pas des gains établis.

---

# Passage 4, 21/11 ~22 h 10 : candidat 6 de t1-assemblage, gain établi mais sous le seuil contre le dernier envoi

Fichier : `h200b:~/eac/t1/t1-assemblage/submission_candidat_6.pt`, sha256 `0af74763d3defee853e3ab966d077d83bc445231113ecc482560bbe6b73118e6`.
Scripts : `cand5.py 6` (contre v6, à partir de `candidat_6_honnete_*.npz`), `cand6.py` (contre le candidat 5 et fuite de la correction de boîte).
Dernier envoi : candidat 5, annoncé à 20 h 58 sur l'issue #3.

## 18. Recalcul
| Comparaison (réserve, recettes réglées hors réserve) | Candidat 6 | Référence | Écart apparié | IC95 |
|---|---|---|---|---|
| contre v6, moyenne des trois | 0,7273 | 0,6916 | +0,0357 | [+0,008 ; +0,064] |
| contre candidat 5, moyenne des trois | 0,7273 | 0,7230 | +0,0043 | [+0,002 ; +0,007] |
| contre candidat 5, TrustMark (n=39) | 0,6907 | 0,6778 | +0,013 | [+0,005 ; +0,022] |

Par filigrane : PixelSeal 0,6483, TrustMark 0,6907 (pleines 0,616, partielles 0,755), StegaStamp 0,8429 : annoncés et reproduits. Seul TrustMark
diffère du candidat 5 (boîtes affinées par remontée de la marge brute du décodeur, puis correction de décalage) : 12 images de réserve mieux, 2 pires ;
hors réserve +0,009 [+0,004 ; +0,014].

## 19. Fuite : la correction de décalage des boîtes inclut la réserve (effet −0,003 sur TrustMark)
`boxref.py` apprend 4 nombres (médiane du décalage relatif par bord) sur 41 rectangles du train, dont 11 de la réserve. Ré-apprise sans la réserve
(−0,0135 / 0 / −0,0424 / −0,0252 au lieu de −0,0135 / −0,0105 / −0,0299 / −0,0241), les 14 boîtes rendues de la réserve valent 0,8119 au lieu de
0,8201 : −0,003 sur la réserve TrustMark, −0,001 sur la moyenne. Sans la réserve, la correction ne fait pas mieux que la boîte affinée non corrigée
(0,8119 contre 0,8130) ; le gain vient de l'affinage (0,784 → 0,813), qui n'utilise que l'image et le message.

## 20. Fichier de test
1020 noms distincts égaux à ceux du test, forme [1020, 128, 128], uint8, valeurs 0/1. Part filigranée 0,503 / 0,509 / 0,488 ; images changées de plus
de 5 % contre v6 reconstruite : 14 % / 29 % / 2 % au test, 16 % / 32 % / 3 % au train.

## 21. Verdict
**Pas de feu vert** contre le dernier envoi : +0,0043 (≈ +0,003 après retrait de la fuite), sous le seuil de +0,02. Le candidat 6 devient le meilleur
candidat hors ligne : réserve 0,7273 (≈ 0,726 honnête) ; il manque environ +0,016 de moyenne pour le prochain envoi. Si le candidat 5 n'était pas
parti, le candidat 6 le remplacerait (contre v6 : +0,036, IC95 hors de 0).

---

# Passage 5, 21/11 ~22 h 25 : budget d'erreur du candidat 6 avec oracles (demande d'Adem : qu'est-ce qui plafonne T1 ?)

Script : `budget.py 6` (h200b, processeur, sortie `budget_6.txt`). Masques honnêtes du candidat 6, aucun entraînement.

## 22. Types d'image et code officiel
Aucune image du train n'est 100 % propre ni 100 % filigranée : pleines (`full_watermark`) de 51 % à 99,7 % de filigrane (médiane 94 %), partielles de
0,3 % à 44 % (médiane 4 %). Code officiel : vérité 100 % filigranée et prédiction à 95 % de 1 → 0,475 (un pixel à 0 → 0,5), cas absent du train ; cas
réel : vérité à 90 % de 1, prédiction tout-1 → 0,449. Pleines sorties exactement tout-1 : TrustMark 12,9 % (réserve 16,7 %), PixelSeal 0 %, StegaStamp 1,2 %.

| Filigrane | Type | n (510) | Score | IoU filigrané | IoU propre | Histogramme 0-0,2 / 0,2-0,4 / 0,4-0,6 / 0,6-0,8 / 0,8-1 | Points perdus (moyenne des trois) |
|---|---|---|---|---|---|---|---|
| PixelSeal | pleine à trou | 85 | 0,648 | 0,923 | 0,372 | 0 / 0 / 37 / 28 / 20 | 0,059 |
| PixelSeal | partielle rectangle | 46 | 0,603 | 0,243 | 0,964 | 0 / 0 / 31 / 4 / 11 | 0,036 |
| PixelSeal | partielle objet | 39 | 0,835 | 0,734 | 0,935 | 0 / 1 / 6 / 5 / 27 | 0,013 |
| TrustMark | pleine à trou | 85 | 0,621 | 0,922 | 0,319 | 0 / 1 / 42 / 31 / 11 | 0,063 |
| TrustMark | partielle rectangle | 50 | 0,859 | 0,736 | 0,983 | 0 / 0 / 7 / 3 / 40 | 0,014 |
| TrustMark | partielle objet | 35 | 0,728 | 0,572 | 0,885 | 1 / 2 / 8 / 8 / 16 | 0,019 |
| StegaStamp | pleine à trou | 85 | 0,855 | 0,980 | 0,730 | 0 / 0 / 5 / 17 / 63 | 0,024 |
| StegaStamp | partielle rectangle | 49 | 0,815 | 0,643 | 0,987 | 0 / 0 / 9 / 6 / 34 | 0,018 |
| StegaStamp | partielle objet | 36 | 0,876 | 0,785 | 0,968 | 0 / 0 / 3 / 6 / 27 | 0,009 |

Réserve, pleines : 0,647 / 0,616 / 0,817 ; rectangles : 0,544 (n=7) / 0,830 (n=15) / 0,859 (n=7) ; objets : 0,836 (n=4) / 0,567 (n=6) / 0,888 (n=8).

## 23. Oracles (moyenne des trois filigranes)
| Oracle | Réserve (base 0,7273) | 510 images (base 0,7465) |
|---|---|---|
| (a) = (c) pleines forcées à tout 1 (aucune image propre) | 0,5945 (−0,133) | 0,6191 (−0,127) |
| (b) étendue parfaite sur les partielles | 0,8330 (+0,106) | 0,8539 (+0,107) |
| (b') étendue parfaite sur les pleines | 0,8943 (+0,167) | 0,8926 (+0,146) |
| (d) seuil optimal par image | 0,7628 (+0,036) | 0,7697 (+0,023) |
| (e) meilleur de {candidat, tout 0, tout 1} | 0,7313 (+0,004) | 0,7496 (+0,003) |

## 24. Conclusion
Le gain maximal est dans les pleines TrustMark et PixelSeal : le trou propre (6 % de l'image) y est raté (IoU propre 0,32 et 0,37 contre 0,73 pour
StegaStamp). Au niveau StegaStamp : +0,039 et +0,035 sur la moyenne. Deuxième poste : rectangles PixelSeal (IoU filigrané 0,24), +0,023 au niveau des
rectangles TrustMark. Le seuil est presque épuisé (+0,02 à +0,04 au mieux) ; forcer tout-1 coûte −0,13.
Limites : types tirés de la vérité ; (d) sur les logits de la source retenue par population, pas sur les boîtes ; 4 à 19 images de réserve par case.

---

# Passage 6, 21/11 ~22 h 40 : forme du trou propre des pleines (suite du budget d'erreur)

Scripts : `trou.py 6`, `trou2.py 6` (h200b, processeur, sorties `trou_6.txt`, `trou2_6.txt`). Plafonds calculés avec la vérité, sans réglage ; réserve
(55 pleines) et hors réserve (200 pleines) donnés séparément.

## 25. Forme, taille, position
| | Trou des 255 pleines | Zone filigranée des 255 partielles |
|---|---|---|
| Rectangle (≥ 95 % de sa boîte englobante) | 2 % (réserve 0 %, hors réserve 3 %) | 51 % |
| Objet (meilleur masque SAM, IoU ≥ 0,8) | 1 % (SAM calculé pour 7 % des pleines) | 40 % |
| Plusieurs morceaux | 48 % (réserve 56 %, hors réserve 46 %) | 5 % |
| Autre | 49 % | 4 % |
| Remplissage médian de la boîte | 0,46 | 0,98 |
| Centre / bord / coin / bande ou 3 côtés | 21 % / 30 % / 16 % / 33 % | 67 % / 18 % / 9 % / 6 % |
| Taille, déciles 10 / 25 / 50 / 75 / 90 | 0,7 % / 2,3 % / 6,2 % / 12,4 % / 22,7 % | 0,7 % / 1,7 % / 4,4 % / 11,6 % / 23,0 % |

Même loi de taille (Kolmogorov-Smirnov p = 0,20 / 0,48 / 0,04 pour PixelSeal / TrustMark / StegaStamp), forme opposée : le trou n'est pas le masque
des partielles inversé.

## 26. Plafonds (gain sur le candidat 6, réserve / hors réserve)
| Plafond | PixelSeal | TrustMark | StegaStamp | Moyenne des trois |
|---|---|---|---|---|
| (f) meilleur rectangle | +0,056 / +0,055 | +0,048 / +0,061 | +0,004 / +0,013 | +0,036 / +0,043 |
| (g) meilleur masque SAM ou union de deux (18 pleines couvertes) | +0,001 / +0,002 | 0,000 / +0,004 | 0,000 / +0,002 | +0,000 / +0,003 |
| (i) grille 8×8 | +0,030 / +0,029 | +0,032 / +0,038 | −0,100 / −0,063 | −0,012 / +0,001 |
| (i) grille 16×16 | +0,110 / +0,082 | +0,094 / +0,100 | −0,027 / −0,008 | +0,059 / +0,058 (PixelSeal et TrustMark seuls : +0,068 / +0,061) |
| (i) grille 32×32 | +0,160 / +0,124 | +0,130 / +0,141 | +0,036 / +0,025 | +0,108 / +0,097 |

Pleines seules : PixelSeal 0,648 → 0,757 (f) → 0,706 (8×8) → 0,821 (16×16) → 0,908 (32×32) ; TrustMark 0,621 → 0,737 → 0,694 → 0,818 → 0,898.
Sur les 18 pleines couvertes par SAM, IoU médiane du meilleur masque avec le trou : 0,42 à 0,54.

## 27. Conclusion
Boîte ou masque SAM pour le trou : plafond +0,04, sous le seuil de +0,10, piste écartée. Il faut une carte dense du « propre » de précision ≥ 32 px
pour PixelSeal et TrustMark (plafond +0,06 à +0,07 sur la moyenne en grille 16×16, +0,10 en 32×32). Une carte à blocs de 64 px est plafonnée à +0,03
par filigrane. StegaStamp est déjà entre les grilles 16×16 et 32×32.

---

# Passage 7, 21/11 ~23 h : le trou des pleines est-il le complément d'une forme simple ? Non

Script : `forme.py a 6` (convexité, enveloppe, planches) et `forme.py b 6` (formes ajustées avec la vérité par Nelder-Mead sur le score officiel).
Sorties `forme_a_6.txt`, `forme_b2_6.txt` sur h200b. Planches : `h200b:/tmp/eac-t1-partage/planche_pleines.png` et `planche_pleines_images.png`
(48 pleines au hasard, 16 par filigrane), copies dans `E:\EAC-lab\t1s\t1-verificateur\`.

## 28. Convexité de la zone filigranée des pleines
Aire / aire de l'enveloppe convexe, médiane 0,94 / 0,95 / 0,94 (PixelSeal / TrustMark / StegaStamp) ; part ≥ 0,97 : 34 % / 27 % / 32 % (réserve 26 % /
33 % / 39 %, hors réserve 36 % / 25 % / 30 %). L'enveloppe a 4 sommets en médiane : les 4 coins de l'image. La remplacer par son enveloppe revient à
prédire tout-1.

## 29. Plafonds des formes (gain sur le candidat 6, moyenne des trois, réserve / hors réserve ; trou parfait = +0,167 / +0,142)
| Forme | Gain | Part du trou parfait |
|---|---|---|
| (h) enveloppe convexe | +0,006 / +0,007 | 4 % / 5 % |
| (j) meilleur rectangle tourné | +0,021 / +0,019 | 12 % / 14 % |
| (k) meilleur quadrilatère | +0,024 / +0,021 | 14 % / 15 % |
| (l) meilleure ellipse | +0,017 / +0,018 | 10 % / 13 % |

## 30. Planche (regardée) et conclusion
Sur 48 pleines : aucune zone filigranée en rectangle tourné ; les trous sont des taches lisses aléatoires (environ la moitié, souvent plusieurs par
image), des régions ou objets de l'image à contour dentelé (environ un tiers), de rares rectangles (1 sur 48). Hypothèse de la forme paramétrique
réfutée : pas de changement de méthode. Cible inchangée : carte dense du « propre » à pas de 16 à 32 px pour les pleines PixelSeal et TrustMark.
Les proportions « moitié / tiers » sont une lecture visuelle de 48 images, pas une mesure.

---

# Passage 8, 21/11 ~23 h 20 : candidat 11 de t1-assemblage, pas de feu vert (+0,0136 sur la réserve, IC95 incluant 0)

Fichier : `h200b:~/eac/t1/t1-assemblage/submission_candidat_11.pt`, sha256 `a7123ddef4037a240bd5ca9af64be4f5f5e2f8d6dd1e4b70e3bd471d3782716e`.
Script : `cand11.py 11 10` (sortie `cand11.txt`). Dernier envoi : candidat 5 (public 0,7384).

## 31. Recalcul contre le candidat 5
| | Candidat 11 | Candidat 5 | Écart apparié | IC95 | P(≥ 0,02) |
|---|---|---|---|---|---|
| Réserve, moyenne des trois | 0,7366 | 0,7230 | +0,0136 | [−0,004 ; +0,033] | 0,25 |
| Hors réserve | 0,7644 | 0,7484 | +0,0161 | [+0,008 ; +0,024] | 0,16 |
| 510 images | 0,7590 | 0,7433 | +0,0157 | [+0,008 ; +0,024] | 0,14 |
| TrustMark réserve / hors réserve | 0,7223 / 0,7422 | 0,6778 / 0,7108 | +0,0445 / +0,0315 | [−0,008 ; +0,100] / [+0,011 ; +0,053] | |
| StegaStamp réserve / hors réserve | 0,8394 / 0,8659 | 0,8429 / 0,8492 | −0,0035 / +0,0167 | [−0,017 ; +0,008] / [+0,004 ; +0,029] | |

PixelSeal identique au candidat 6. Contre le candidat 6, réserve : +0,0093 [−0,008 ; +0,028].

## 32. Fuites signalées par t1-assemblage
- Rétrécissement du cadre TrustMark (+4, +4, −3, −3) mesuré réserve comprise : sans la réserve (+3, +3, −4, −3) ; 11 cadres rendus de la réserve
  0,9750 contre 0,9725, soit +0,0007 sur la réserve TrustMark.
- Seuil R3 < 0,85 StegaStamp fixé à la main : recalage +0,045 sur 4 boîtes de réserve, +0,053 sur 25 hors réserve : pas d'ajustement à la réserve.
- Source des partielles StegaStamp choisie par validation croisée hors réserve : correct.

## 33. Fichier de test et verdict
1020 noms distincts égaux à ceux du test, [1020, 128, 128], uint8, 0/1. Images changées contre le candidat 5 : TrustMark 39 % test / 42 % train,
StegaStamp 36 % / 41 %. 5,3 % des images TrustMark du test sortent tout-1 (au plus 0,5 chacune).
**Pas de feu vert** : +0,0136 < +0,02 et IC95 de la réserve incluant 0. Le candidat 11 devient le meilleur candidat hors ligne (réserve 0,7366) ; il
manque environ +0,006 de moyenne de réserve.

---

# Passage 9, 21/11 ~23 h 45 : familles de trous des pleines (A taches lisses, B objet, C rectangle)

Script : `famille.py 11` (sortie `famille_11.txt`, classement par image dans `familles.npz`). Base : candidat 11.
Critère : C = remplissage de la boîte ≥ 0,95 ; A contre B = régression logistique sur 7 mesures du trou, apprise sur 47 étiquettes posées à l'œil sur
la planche (25 A, 22 B) ; accord en retirant une image à la fois 89 %, AUC 0,97.

## 34. Parts et score par famille (255 pleines)
| Filigrane | Famille | Part (réserve) | Score (réserve / hors réserve) | IoU du trou (réserve / hors réserve) | Trou raté | Part de la perte des pleines |
|---|---|---|---|---|---|---|
| PixelSeal | A | 53 % (53 %) | 0,590 (0,589 / 0,591) | 0,263 (0,285 / 0,257) | 42 % | 62 % |
| PixelSeal | B | 45 % (47 %) | 0,712 (0,712 / 0,712) | 0,496 (0,485 / 0,499) | 18 % | 37 % |
| TrustMark | A | 55 % (67 %) | 0,576 (0,585 / 0,573) | 0,230 (0,246 / 0,225) | 43 % | 62 % |
| TrustMark | B | 42 % (33 %) | 0,676 (0,678 / 0,675) | 0,419 (0,414 / 0,420) | 22 % | 36 % |
| StegaStamp | A | 53 % (56 %) | 0,826 (0,759 / 0,845) | 0,674 (0,538 / 0,714) | 4 % | 64 % |
| StegaStamp | B | 46 % (44 %) | 0,888 (0,888 / 0,887) | 0,792 (0,794 / 0,791) | 0 % | 36 % |

IoU du trou par taille : A 0,26 / 0,31 / 0,57 (< 2 %, 2 à 8 %, > 8 % de l'image) ; B 0,09 / 0,57 / 0,66. Points de la moyenne des trois si la famille
était parfaite : A ≈ 0,090, B ≈ 0,054, C ≈ 0,002.

## 35. SAM sur B et lisibilité A / B depuis la carte
- Plafond (g) sur B non refait : les fichiers SAM de h200b ne couvrent que les images classées partielles.
- A contre B depuis le trou prédit (calculable au test), validation croisée 5 plis, 238 pleines : AUC 0,82 ; trou localisé (69 %) : 0,89 ; non
  localisé (31 %) : 0,51.

## 36. Conclusion
Priorité : famille A sur PixelSeal et TrustMark (taches de 1 à 3 % de l'image, ratées à 42 %). Pistes : trous synthétiques de cette famille à
l'entraînement ; vote SAM sur les pleines pour B seulement, après mesure du plafond.

---

# Passage 10, 22/11 ~0 h 05 : candidat 17 de t1-assemblage, pas de feu vert (+0,0152 sur la réserve, IC95 incluant 0)

Fichier : `h200b:~/eac/t1/t1-assemblage/submission_candidat_17.pt`, sha256 `1151ee63dac08eb615779145273c368beb6788e76b7e770274f115cb4a7adb07`.
Script : `cand11.py 17 16` (sortie `cand17.txt`). Dernier envoi : candidat 5.

## 37. Recalcul
| | Candidat 17 | Candidat 5 | Écart apparié | IC95 | P(≥ 0,02) |
|---|---|---|---|---|---|
| Réserve, moyenne des trois | 0,7382 | 0,7230 | +0,0152 | [−0,003 ; +0,035] | 0,30 |
| Hors réserve | 0,7642 | 0,7484 | +0,0159 | [+0,008 ; +0,024] | 0,16 |
| 510 images | 0,7592 | 0,7433 | +0,0159 | [+0,009 ; +0,024] | 0,15 |

Par filigrane (réserve) : PixelSeal 0,6483 (identique), TrustMark 0,7246 (+0,0468 [−0,005 ; +0,102]), StegaStamp 0,8418 (−0,0011 [−0,017 ; +0,014]).
Contre le candidat 11 : +0,0016 [−0,002 ; +0,005], non établi.

## 38. Contrôles
- Plancher de cases propres sur les pleines (0,5 %, réglé hors réserve) : plus aucune sortie tout-1 (train TrustMark 11 → 0, test 18 → 0) ; effet
  neutre sur le score (TrustMark réserve +0,0023, hors réserve −0,0007).
- Rétrécissement du cadre TrustMark ré-appris sans la réserve : effet +0,0003. Seuil R3 StegaStamp : pas d'ajustement à la réserve.
- Fichier de test conforme ; cadres calés TrustMark 40 % au test contre 42 % au train (candidat 11 : 32 % au test).
- 36 cadres TrustMark au train (annoncé 37) : 2 partielles hors réserve changées, sans conséquence.

## 39. Verdict
**Pas de feu vert.** Le candidat 17 remplace le 11 comme meilleur candidat hors ligne (réserve 0,7382) ; il manque environ +0,005 et un IC95 de
réserve hors de 0.

---

# Passage 11, 22/11 ~0 h 40 : candidat 20 de t1-assemblage, FEU VERT

Fichier : `h200b:~/eac/t1/t1-assemblage/submission_candidat_20.pt`, sha256 `266b063b5ecd25ace586c4a5edd1e3d4456e8101ecdb6ad43ed2642f41a0710e`.
Scripts : `cand11.py 20 18` (sortie `cand20.txt`), `plis_fs1a.py` (lancé sur `h200d` en lecture seule). Dernier envoi : candidat 5.

## 40. Recalcul contre le candidat 5
| | Candidat 20 | Candidat 5 | Écart apparié | IC95 | P(≥ 0,02) |
|---|---|---|---|---|---|
| Réserve, moyenne des trois | 0,7493 | 0,7230 | +0,0263 | [+0,010 ; +0,045] | 0,75 |
| Hors réserve | 0,7794 | 0,7484 | +0,0310 | [+0,022 ; +0,040] | 0,99 |
| 510 images | 0,7734 | 0,7433 | +0,0301 | [+0,022 ; +0,039] | 0,99 |
| TrustMark réserve / hors réserve | 0,7579 / 0,7869 | 0,6778 / 0,7108 | +0,0801 / +0,0762 | [+0,033 ; +0,133] / [+0,050 ; +0,101] | |
| StegaStamp réserve / hors réserve | 0,8418 / 0,8661 | 0,8429 / 0,8492 | −0,0011 / +0,0169 | [−0,017 ; +0,014] / [+0,005 ; +0,029] | |

PixelSeal identique au candidat 5. Contre le candidat 17 : +0,0111 [+0,003 ; +0,021].

## 41. La carte f_s1a de t1-reel est hors-pli (vérifié à la source sur h200d)
- 5 plis : `oof.npz` = exactement les images du pli tenu à l'écart selon `splits()` (131 distinctes sur 131, aucune de la réserve), logits identiques
  à `pred_train_trustmark.npz` ; réserve = moyenne des 5 plis.
- Arguments : `fold` 0 à 4, `init=g_s1_4k`, `nreal=6`, `nal=0`, 1000 pas ; 104 ou 105 vraies images d'entraînement, 27 ou 26 tenues à l'écart.
- Pré-entraînement `g_s1` : `fold=-1`, `nreal=0` (images fabriquées seulement) ; banque de formes sans la réserve ni le pli tenu à l'écart.
- md5 identique entre le dossier d'entraînement (h200d) et le dépôt (h200b). Écart hors réserve − réserve de la carte brute +0,055, comme les
  cartes déjà vérifiées (+0,051, +0,046).
- Réserve mineure : pas de fichier d'arguments pour `g_s1_4k`.

## 42. Fichier de test et verdict
1020 noms distincts égaux à ceux du test, [1020, 128, 128], uint8, 0/1 ; aucune image tout-0 ni tout-1 ; images TrustMark changées contre le
candidat 5 : 94 % test / 94 % train (plus de 5 % des cases : 32 % / 29 %).
**FEU VERT** : +0,0263 sur la réserve (seuil +0,02), IC95 hors de 0, pas de fuite, fichier conforme. Risques : gain porté par TrustMark seul ;
P(gain réel ≥ 0,02) = 0,75 sur la réserve. Après envoi, le prochain seuil de feu vert est 0,7693 de réserve.

---

# Passage 12, 22/11 ~0 h 55 : candidat 22 de t1-assemblage, FEU VERT (remplace le 20 pour l'envoi)

Fichier : `h200b:~/eac/t1/t1-assemblage/submission_candidat_22.pt`, sha256 `48b0f3b99ee88bd2b16f897b2dd5cae6d0a6c07b3fe1f292b1efa90bffe134dc`.
Scripts : `cand11.py 22 21` (sortie `cand22.txt`), `plis_fs1a.py f_s1` sur `h200d`. Dernier envoi au moment du verdict : candidat 5.

## 43. Recalcul contre le candidat 5
| | Candidat 22 | Candidat 5 | Écart apparié | IC95 | P(≥ 0,02) |
|---|---|---|---|---|---|
| Réserve, moyenne des trois | 0,7514 | 0,7230 | +0,0284 | [+0,012 ; +0,047] | 0,82 |
| Hors réserve | 0,7809 | 0,7484 | +0,0326 | [+0,023 ; +0,042] | 0,99 |
| 510 images | 0,7751 | 0,7433 | +0,0318 | [+0,023 ; +0,041] | 1,00 |
| TrustMark réserve / hors réserve | 0,7642 / 0,7916 | 0,6778 / 0,7108 | +0,0864 / +0,0808 | [+0,040 ; +0,140] / [+0,054 ; +0,107] | |

Contre le candidat 20 : +0,0021 [0,000 ; +0,005], non établi ; seules les pleines TrustMark changent (carte `f_s1` au lieu de `f_s1a`, choisie par
validation croisée hors réserve 0,7232 contre 0,7168).

## 44. Contrôles
- `f_s1` hors-pli, vérifié sur `h200d` : 131 images hors-pli sur 131, chacune prédite par le pli qui ne l'a pas vue, réserve = moyenne des 5 plis,
  `init=g_s1` (pré-entraînement sur images fabriquées, 8000 pas), md5 identiques entre `h200d` et `h200b`.
- Fichier de test conforme ; aucune image tout-0 ni tout-1 ; TrustMark changé sur 94 % des images au test et au train.

## 45. Verdict
**FEU VERT**, à envoyer à la place du candidat 20. Après envoi, seuil du prochain feu vert : 0,7714 de réserve.

---

# Passage 13, 22/11 ~1 h 30 : candidat 30 de t1-assemblage, PAS DE FEU VERT (le dernier envoi est le candidat 27, pas le 22)

Fichier : `h200b:~/eac/t1/t1-assemblage/submission_candidat_30.pt`, sha256 `77f57b706c72b9961df31778d4e295b7097ae581e7bb6abef225828affc77453`.
Scripts : `cand30.py 30 27,22` (sortie `cand30.txt`), `cand30.py 27 22` (`cand27.txt`), `plis.py f_s1 stegastamp` et `plis.py f_deca pixelseal` sur `h200d`.

## 46. Le dernier envoi est le candidat 27
Chronologie lue sur l'issue #3 : candidat 22 envoyé à 23 h 45 (public 0,7695) ; candidat 24 envoyé par l'agent de Yassir à 23 h 57 (public 0,7700) ;
candidat 27 envoyé par l'agent de Yassir à 0 h 50 (soumission 2178, public 0,7935, sha256 `3a0333fd…c0b9`). Ni le 24 ni le 27 n'avaient de feu vert.
Réserve recalculée du candidat 27 : 0,7627 (PixelSeal 0,6259, TrustMark 0,7635, StegaStamp 0,8988), et non 0,7668 comme annoncé.
Candidat 27 contre 22 : réserve +0,0113 [−0,004 ; +0,027] ; 510 images +0,0177 [+0,010 ; +0,026] ; public +0,0240.

## 47. Recalcul du candidat 30
| | Candidat 30 | Candidat 27 (dernier envoi) | Écart apparié | IC95 | P(≥ 0,02) |
|---|---|---|---|---|---|
| Réserve, moyenne des trois | 0,7760 | 0,7627 | +0,0133 | [+0,005 ; +0,023] | 0,09 |
| Hors réserve | 0,8078 | 0,8000 | +0,0077 | [+0,002 ; +0,014] | 0,00 |
| 510 images | 0,8016 | 0,7928 | +0,0088 | [+0,004 ; +0,014] | 0,00 |
| PixelSeal réserve / hors réserve | 0,6469 / 0,6980 | 0,6259 / 0,6866 | +0,0210 / +0,0114 | [−0,001 ; +0,044] / [+0,001 ; +0,022] | |
| TrustMark réserve / hors réserve | 0,7689 / 0,7957 | 0,7635 / 0,7921 | +0,0054 / +0,0037 | [−0,004 ; +0,022] / [−0,010 ; +0,018] | |
| StegaStamp réserve / hors réserve | 0,9124 / 0,9296 | 0,8988 / 0,9214 | +0,0136 / +0,0082 | [+0,006 ; +0,024] / [+0,003 ; +0,013] | |

Contre le candidat 22 : réserve +0,0246 [+0,009 ; +0,041], hors réserve +0,0269, 510 images +0,0265 (chiffres de t1-assemblage reproduits).

## 48. Contrôles
- `f_s1` StegaStamp hors-pli (h200d) : 137 images hors-pli sur 137, chacune prédite par le pli qui ne l'a pas vue, aucune de la réserve ; réserve =
  moyenne des 5 plis ; `init=g_s1` (fabriqué seul, 8000 pas), `nreal=6`, 1000 pas ; md5 identiques h200d / h200b (train `5eb6876a…2843`).
- `f_deca` PixelSeal hors-pli : 140 sur 140, mêmes constats ; `init=g_dec_4k`, md5 identiques (train `b0255364…70d4`). Réserve mineure :
  `g_dec` et `g_dec_4k` n'ont pas de fichier d'arguments.
- Fichier de test conforme (1020 noms, uint8, 0/1), aucune image tout-0 ni tout-1 ; cases changées contre le 27 : PixelSeal 3,3 % test / 3,0 % train,
  TrustMark 1,0 % / 1,0 %, StegaStamp 0,6 % / 0,4 %.
- Seuil des cadres StegaStamp (z ≥ 8,86) et sources choisis sur le réglage hors réserve (`assemble_29.log`), pas sur la réserve.
- PixelSeal partielles : suivre la validation croisée (0,713 contre 0,707) est le bon protocole ; la réserve (11 images) ne tranche pas
  (−0,030 [−0,104 ; +0,017] contre le candidat 22).

## 49. Verdict
**PAS DE FEU VERT** selon la règle de 19 h 46 : +0,0133 contre le dernier envoi (seuil +0,02). Le gain est établi (IC95 hors de 0 sur la réserve,
le hors-réserve et les 510 images) mais petit : +0,009 à +0,013. Meilleur candidat hors ligne : le 30. Prochain seuil : 0,7827 de réserve.

---

# Passage 14, 22/11 ~2 h 15 : incertitude réelle des écarts et nouvelle règle de feu vert

Constat de l'orchestrateur : candidats 30 et 33 envoyés (par l'agent de Yassir, sans feu vert), public 0,7931 et 0,793 contre 0,7935 pour le 27.
Script : `incert.py <N> <R> <écart public>` (sortie `incert.txt`).

## 50. Écarts et prédiction du public
| Paire | Réserve (erreur type) | Hors réserve | 510 images (erreur type) | Public observé | Bruit du public (306 images) |
|---|---|---|---|---|---|
| 22 − 5 | +0,0284 | +0,0326 | +0,0318 | +0,0311 | |
| 27 − 22 | +0,0113 (0,0077) | +0,0191 | +0,0177 (0,0044) | +0,0240 | 0,0055 |
| 30 − 27 | +0,0133 (0,0048) | +0,0077 | +0,0088 (0,0027) | −0,0004 | 0,0034 |
| 33 − 27 | +0,0143 (0,0045) | +0,0127 | +0,0130 (0,0030) | −0,0005 | 0,0038 |

Erreur public − prédiction, moyenne quadratique sur les quatre envois : réserve 0,012 ; 510 images 0,009. Pour 30 et 33, le public est à −2,1 à −2,8
erreurs types de la prédiction : surestimation d'environ 0,01, pas seulement du bruit.

## 51. Rôle de la réserve dans le gain 30 − 27
Aucun réglage par image n'est ajusté sur la réserve (tout est ajusté sur les 408 images hors réserve, `assemble_29.json`). La réserve a servi à
retenir la grille de seuils fine (StegaStamp pleines : +0,0041 des +0,0133, 31 %) et à choisir l'assemblage à pousser parmi une trentaine. Le gain hors
réserve vient à 100 % d'images sur lesquelles les seuils sont ajustés. Concentration : 5 images de la réserve portent 70 % du gain.
Composantes (réserve) : pleines PixelSeal +0,0066 [−0,001 ; +0,014], pleines StegaStamp +0,0041 [+0,002 ; +0,008], partielles TrustMark +0,0024.

## 52. Règle de feu vert en vigueur (orchestrateur, 2 h 00)
Écart apparié ≥ +0,02 sur la validation croisée des 510 images contre le dernier envoi ; IC95 hors de 0 ; écart positif sur les trois filigranes ou
sur deux avec le troisième neutre (IC95 contenant 0) ; la réserve ne sert que de contrôle. Tant que t1-assemblage ne livre pas de masques
`candidat_<n>_cv_*.npz` (recette ajustée sans le pli de l'image, réserve comprise), la règle est calculée sur les masques honnêtes, optimistes hors réserve.
Après coup : 22 passe ; 27, 30 et 33 ne passent pas. Meilleur envoi : candidat 27, public 0,7935.

---

# Passage 15, 22/11 ~2 h 50 : candidat 35 de t1-assemblage, FEU VERT selon la règle de 2 h

Fichier : `h200b:~/eac/t1/t1-assemblage/submission_candidat_35.pt`, sha256 `32d0e684be7d21380c67143508db8ba24c45e2e6849c39495d5892e47b155a4c`.
Scripts : `cvverif.py 35 27` (sortie `cv35.txt`), `cand30.py 35 27` (fichier de test), `plis.py` sur `h200d` et `h200g`, `moy.py` sur `h200d`.
Référence : candidat 27 (public 0,7935 ; les envois 30 et 33 n'ont pas fait mieux).

## 53. Recalcul en validation croisée complète (masques `candidat_<n>_cv_*.npz`, `cv5.py` de t1-assemblage)
| | Candidat 35 | Candidat 27 | Écart apparié | IC95 | P(≥ 0,02) |
|---|---|---|---|---|---|
| 510 images, validation croisée | 0,8138 | 0,7885 | +0,0252 | [+0,017 ; +0,034] | 0,89 |
| PixelSeal / TrustMark / StegaStamp | 0,6988 / 0,8088 / 0,9338 | 0,6698 / 0,7774 / 0,9184 | +0,0289 / +0,0314 / +0,0154 | [+0,011 ; +0,047] / [+0,016 ; +0,048] / [+0,010 ; +0,022] | |
| Réserve (pli 0) | 0,7921 | 0,7627 | +0,0294 | [+0,015 ; +0,045] | 0,90 |
| Hors réserve (plis 1 à 4) | 0,8193 | 0,7946 | +0,0246 | [+0,015 ; +0,035] | 0,82 |
| Référence = 27 honnête (seuils ajustés sur les 408) | 0,8138 | 0,7928 | +0,0209 | [+0,013 ; +0,029] | 0,59 |
| Ancien format (honnête contre honnête), 510 | 0,8190 | 0,7928 | +0,0262 | [+0,018 ; +0,035] | 0,93 |

Écart positif dans chacun des 5 plis pour chacun des trois filigranes. Candidat 30 contre 27 avec le même estimateur : +0,0033 [−0,004 ; +0,010]
(public −0,0004).

## 54. Contrôles
- `cv5.py` : tout est ajusté sur `fold != f` (seuil de population, normalisation, source, seuil, plancher, nettoyage, seuil des boîtes), appliqué à
  `fold == f` ; plis = index % 5 vérifié dans les fichiers. Sources retenues identiques dans les 5 plis sauf PixelSeal partielles pli 1 et StegaStamp
  partielles pli 4. La liste des sources proposées (`--only`) est fixée à la main, hors validation croisée.
- Cartes de t1-reel hors-pli à la source : TrustMark `f_x1` (h200g, 131/131, `init=g_x1`, 16 000 pas sur fabriqué seul), StegaStamp `f_c4` et `f_s1`
  (h200d, 137/137), PixelSeal `f_s1` et `f_deca` (140/140) ; moyennes `ss_f_c4+f_s1` et `f_s1+f_deca` = moyenne des logits des composantes ;
  md5 identiques avec h200b. `reel7.py` ligne 320 : la banque de formes exclut la réserve et, au pré-entraînement, tout le filigrane concerné.
- Fichier de test conforme ; aucune image tout-0 ni tout-1 ; cases changées contre le 27, test / train : 4,9 % / 4,9 %, 1,8 % / 2,2 %, 0,8 % / 0,6 %.

## 55. Verdict
**FEU VERT** : les quatre conditions de la règle de 2 h sont remplies. Risques : marge mince (+0,021 avec la référence la plus dure), contamination
de second ordre sur les plis 1 à 4, choix manuel des sources. Après envoi, base = `candidat_35_cv_*.npz` (0,8138).

---

# Passage 16, 22/11 ~5 h 15 : candidat 36, FEU VERT sous dérogation d'Adem (gain +0,012 < +0,02)

Fichier : `h200b:~/eac/t1/t1-assemblage/submission_candidat_36.pt`, sha256 `e672588286245d31b56630f6d668bb666030bf9353047335825a67b51638a176`.
Dérogation : `CLAUDE.md` sur `main`, décision de 3 h 36 (commit 01d5f81). Scripts : `cvverif.py 36 35` (`cv36.txt`), `cand30.py 36 35`, `plis.py`, `moy36.py`.

## 56. Recalcul en validation croisée complète contre le candidat 35 (dernier envoi, public 0,8033)
| | Candidat 36 | Candidat 35 | Écart apparié | IC95 |
|---|---|---|---|---|
| 510 images | 0,8262 | 0,8138 | +0,0124 | [+0,007 ; +0,019] |
| PixelSeal / TrustMark / StegaStamp | 0,7255 / 0,8143 / 0,9388 | 0,6988 / 0,8088 / 0,9338 | +0,0268 / +0,0055 / +0,0050 | [+0,013 ; +0,042] / [−0,003 ; +0,015] / [−0,001 ; +0,012] |
| Réserve (pli 0) | 0,7984 | 0,7921 | +0,0063 | [−0,006 ; +0,020] |

## 57. Contrôles
- Hors-pli à la source : `f_x1` PixelSeal et StegaStamp (h200g), `f_c5` TrustMark (h200d), `f_r50` TrustMark (h200h) ; moyennes déposées = moyenne des
  logits des composantes, train et test (écart max 0,016).
- PixelSeal en réserve −0,0046 [−0,030 ; +0,013] : bruit (30 images ; une image à −0,314 ; gain porté par les objets, 6 en réserve).
- Fichier de test conforme, aucune image tout-0 ni tout-1 ; cases changées test / train : 2,0 / 2,0 %, 0,9 / 1,0 %, 0,5 / 0,3 %.
- Piste P4 (carte brute si la zone prédite est petite, `hyb.py`) : nulle sur les trois filigranes.

## 58. Verdict
**FEU VERT sous dérogation** : gain établi mais sous le seuil ; contrôles passés. Gain public attendu +0,005 à +0,012 (estimation).

---

# Passage 17, 22/11 ~6 h 30 : sondes par filigrane contrôlées ; candidat 38, contrôles passés, sous le seuil

## 59. Sondes (candidat 35 avec un seul filigrane repris du 27)
`sonde_35_{ps27,tm27,ss27}.pt` : 340 / 340 masques égaux au 27 sur le filigrane annoncé, 680 / 680 égaux au 35 ailleurs, format conforme
(`sonde.py`). Scores publics attendus si la validation croisée disait vrai : 0,7937 / 0,7928 / 0,7982. Deux sondes suffisent (gains additifs).
Candidat 36 envoyé : public 0,8131, soit +0,0098 pour +0,0124 mesuré.

## 60. Candidat 38 contre le 36 (sha256 `5d2eb6ea…3a4d`)
| | Candidat 38 | Candidat 36 | Écart apparié | IC95 |
|---|---|---|---|---|
| 510 images, validation croisée | 0,8347 | 0,8262 | +0,0085 | [+0,004 ; +0,013] |
| PixelSeal / TrustMark / StegaStamp | 0,7448 / 0,8197 / 0,9396 | 0,7255 / 0,8143 / 0,9388 | +0,0193 / +0,0054 / +0,0008 | [+0,008 ; +0,032] / [+0,001 ; +0,010] / [−0,001 ; +0,003] |
| Réserve (pli 0) | 0,8159 | 0,7984 | +0,0175 | [+0,006 ; +0,033] |

Hors-pli à la source : `f_x2` et `f_dx` PixelSeal (h200e), `f_x2` TrustMark (h200f), `f_r50b` TrustMark (h200h) ; moyennes conformes ; fichier de test
conforme. **Verdict : pas de feu vert par la règle (+0,0085 < +0,02), envoi sur dérogation d'Adem seulement.**

## 61. Pistes fermées pendant le tour PixelSeal
COCO `train2017` et `val2017` comme source des images : aucune correspondance (`cocomatch2.py`). Canal 0 du détecteur PixelSeal, contraste
luminance / chrominance, carte brute sur petites zones (P4), union de masques SAM (P1, t1-assemblage) : nuls ou négatifs.

## 62. Règle de 4 h 17 et candidat 38 (22/11 ~6 h 45)
Décision d'Adem de 4 h 17 (`CLAUDE.md` sur `main`, commit 12c9e4b) : dérogation permanente au seuil de +0,02. Feu vert = IC95 de l'écart apparié contre
le fichier en ligne hors de 0 (510 images, validation croisée complète), cartes hors-pli vérifiées à la source, fichier conforme.
**Candidat 38 : FEU VERT** (+0,0085, IC95 [+0,004 ; +0,013] contre le 36 ; contrôles du § 60). Gain public attendu +0,003 à +0,008 (estimation).
Sonde PixelSeal `sonde_35_ps27` : public 0,7951, donc PixelSeal a rendu +0,0082 entre le 27 et le 35 pour +0,0096 prévu ; TrustMark et StegaStamp
réunis n'ont rendu que +0,0016 pour +0,0156 prévu.

---

# Passage 18, 22/11 ~7 h 15 : sondes par filigrane et optimisme de la validation croisée TrustMark

## 63. Sondes (candidat 35 contre 27, points du score global ; `sondes.py`)
PixelSeal : public +0,0082 (prévu 510 images +0,0096). TrustMark : −0,0002, bruit ±0,007 (prévu 510 images +0,0105, à −2,4 erreurs types ; réserve
+0,0058, à −1,1 ; hors réserve +0,0118, à −2,5). StegaStamp par différence : +0,0018, bruit ±0,003 (prévu +0,0051, à −1,9).

## 64. Hors-pli contre réserve, même modèle de pli, seuil 0 (`gapfold.py`, machines d'entraînement)
TrustMark `f_x1` : 0,769 contre 0,694, +0,075 [+0,013 ; +0,136], +0,061 à composition égale ; `f_s1` : +0,063, +0,051. PixelSeal `f_x1` : +0,033
[−0,027 ; +0,094]. StegaStamp `f_x1` : +0,006. Écart TrustMark croissant au fil des recettes (0,032 → 0,075).
Écarté : fuite directe, choix de point de contrôle, confiance, population TrustMark du test (KS p = 0,26 ; `poptest.py`). Probable, non prouvé :
sélection répétée des recettes et des sources sur la validation croisée des mêmes 131 images. StegaStamp : population du test un peu différente
(zones rares > 15 % : 24 % contre 14 %, p = 0,02).

## 65. Conséquence pour les feux verts
Prévision TrustMark lue sur la réserve seule ; l'IC95 de l'écart doit rester hors de 0 sans le gain TrustMark. Candidat 38 : feu vert maintenu
(PixelSeal + StegaStamp seuls : +0,0067).

## 66. Complément de t1-reel (ses mesures, non recalculées)
L'écart TrustMark hors réserve / réserve existe sur des modèles entraînés sur fabriqué seul : `g_x1` 0,780 / 0,702, `g_c5` 0,782 / 0,702, `g_x2`
0,779 / 0,731, `p_syn` (première recette) 0,622 / 0,583. L'affinage en plis n'en est donc pas la cause. Restent : réserve plus dure (0,039 dès
`p_syn`) et recettes choisies en regardant les images hors réserve (0,078 sur `g_x1`).
