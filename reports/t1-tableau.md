# T1 : tableau de bord des méthodes (candidat 38, 2026-11-22 02:56)

mIoU officiel sur le vrai train. **Réserve** = images d'index % 5 == 0 (30 à 39 par filigrane), jamais utilisées pour régler : c'est le chiffre honnête (bruit ±0,03).
« brut » = prédiction telle que déposée (seuil 0). « réglé » = post-traitement réglé hors réserve par `assemble.py` : lissage, seuil par population, fourchette de fraction, nettoyage ; pour les partielles, rectangle ajusté ou accrochage de forme (masque SAM ou rectangle le plus proche du masque).
Pleines / partielles = score sur la réserve, par vraie population. La validation croisée (4 plis hors réserve) sert à choisir ; la réserve sert à vérifier.
« pas hors-pli » = modèle qui a vu les images hors réserve à l'entraînement : aucun réglage possible, son seuil 0 est gardé et seule la réserve le note.

## Candidat 38 contre la soumission v6 (public 0,706)

| | v6, réserve | Candidat, réserve (honnête) | Pleines (rés.) | Partielles (rés.) | Hors réserve | Train entier après re-réglage du seuil (optimiste) |
|---|---|---|---|---|---|---|
| pixelseal | 0,640 | **0,709** | 0,730 | 0,673 | 0,754 | 0,746 |
| trustmark | - | **0,809** | 0,733 | 0,874 | 0,826 | 0,822 |
| stegastamp | - | **0,930** | 0,918 | 0,944 | 0,942 | 0,940 |
| ensemble | - | **0,819** | 0,792 | 0,849 | 0,840 | 0,836 |

Fichier : `/home/adem/eac/t1/t1-assemblage/assemblage_38.pt` (16,734,650 octets, uint8 [1020,128,128]). Contrôle : le même écrivain appliqué au train puis re-noté avec `wmloc.utils.miou_image` = 0.8361 (attendu 0.8361).

## pixelseal

| Source (session/méthode) | Réserve brut | Réserve réglé | Pleines (rés.) | Partielles (rés.) | Validation croisée hors réserve (pleines / partielles) | Déposée |
|---|---|---|---|---|---|---|
| assemblage/rang-reelp3-pred | 0,694 | 0,709 | 0,730 | 0,673 | 0,734 / 0,761 | 02:52 |
| reelp3/pred | 0,684 | 0,697 | 0,730 | 0,641 | 0,734 / 0,710 | 02:44 |
| reelp2/pred | 0,661 | 0,674 | 0,701 | 0,628 | 0,726 / 0,689 | 01:37 |
| assemblage/rang-reelp2-pred | 0,654 | 0,668 | 0,701 | 0,611 | 0,726 / 0,750 | 02:09 |
| assemblage/rang--G | 0,632 | 0,640 | 0,635 | 0,650 | 0,655 / 0,707 | 19:19 |
| reeld/pred | 0,627 | 0,638 | 0,662 | 0,595 | 0,675 / 0,649 | 21:52 |
| pixelseal/pred (= v6) | 0,640 | 0,629 | 0,634 | 0,621 | 0,655 / 0,713 | 22:53 |
| seg_v2dec_pixelseal/pred (pas hors-pli) | 0,573 | 0,573 | 0,596 | 0,534 | - / - | 17:48 |
| **CHOIX DU CANDIDAT 38** | | **0,709** | 0,730 | 0,673 | | |

Images partielles de la réserve, selon le post-traitement (chaque étape s'ajoute à la précédente) : assemblage/rang--G : seuil 0,607, boîte du décodeur 0,650 ; assemblage/rang-reelp2-pred : seuil 0,581, boîte du décodeur 0,611 ; assemblage/rang-reelp3-pred : seuil 0,643, boîte du décodeur 0,673 ; pixelseal/pred : seuil 0,621 ; reeld/pred : seuil 0,556, boîte du décodeur 0,595 ; reelp2/pred : seuil 0,608, boîte du décodeur 0,628 ; reelp3/pred : seuil 0,626, boîte du décodeur 0,641.

Boîtes de la recherche par décodeur lues : 85 images de train, 161 de test (t1-pixelseal/box_train.json, t1-pixelseal/box_test.json).

Choix : pleines = `assemblage/rang-reelp3-pred` {"assemblage/rang-reelp3-pred": 1}, recette {"mode": "seuil", "sigma": 0, "t": 0.051, "clamp": 0.005, "area": 0} ; partielles = `assemblage/rang-reelp3-pred` {"assemblage/rang-reelp3-pred": 1}, recette {"mode": "seuil", "sigma": 0, "t": -0.941, "clamp": false, "area": 32, "box": 12.76}. Population reconnue par le taux de bits justes de l'image entière (seuil 0.79) : exactitude sur la réserve 1.000, 52% du test classé « pleines ».

## trustmark

| Source (session/méthode) | Réserve brut | Réserve réglé | Pleines (rés.) | Partielles (rés.) | Validation croisée hors réserve (pleines / partielles) | Déposée |
|---|---|---|---|---|---|---|
| assemblage/rang-reelx3-pred | 0,753 | 0,809 | 0,733 | 0,874 | 0,778 / 0,873 | 02:52 |
| assemblage/rang-reelx2-pred | 0,747 | 0,799 | 0,709 | 0,877 | 0,768 / 0,869 | 02:09 |
| reelx3/pred | 0,720 | 0,784 | 0,733 | 0,827 | 0,778 / 0,842 | 02:45 |
| reelx2/pred | 0,707 | 0,771 | 0,709 | 0,824 | 0,768 / 0,839 | 01:56 |
| assemblage/pl-rang-reelg-pred | 0,694 | 0,760 | 0,679 | 0,829 | 0,725 / 0,847 | 22:46 |
| assemblage/rang-reelg-pred | 0,690 | 0,758 | 0,675 | 0,829 | 0,713 / 0,855 | 22:16 |
| reelg/pred | 0,678 | 0,743 | 0,675 | 0,802 | 0,713 / 0,814 | 22:01 |
| **CHOIX DU CANDIDAT 38** | | **0,809** | 0,733 | 0,874 | | |

Images partielles de la réserve, selon le post-traitement (chaque étape s'ajoute à la précédente) : assemblage/pl-rang-reelg-pred : seuil 0,703, boîte du décodeur 0,829 ; assemblage/rang-reelg-pred : seuil 0,703, boîte du décodeur 0,829 ; assemblage/rang-reelx2-pred : seuil 0,783, boîte du décodeur 0,877 ; assemblage/rang-reelx3-pred : seuil 0,788, boîte du décodeur 0,874 ; reelg/pred : seuil 0,682, boîte du décodeur 0,802 ; reelx2/pred : seuil 0,727, boîte du décodeur 0,824 ; reelx3/pred : seuil 0,741, boîte du décodeur 0,827.

Boîtes de la recherche par décodeur lues : 86 images de train, 164 de test (t1-assemblage/boxtm2/boxes_trustmark_train_cadre.json, t1-assemblage/boxtm2/boxes_trustmark_test_cadre.json).

Choix : pleines = `assemblage/rang-reelx3-pred` {"assemblage/rang-reelx3-pred": 1}, recette {"mode": "seuil", "sigma": 0, "t": -0.16, "clamp": 0.005, "area": 8} ; partielles = `assemblage/rang-reelx3-pred` {"assemblage/rang-reelx3-pred": 1}, recette {"mode": "seuil", "sigma": 0, "t": -0.954, "clamp": false, "area": 0, "box": 7.0}. Population reconnue par le taux de bits justes de l'image entière (seuil 0.76) : exactitude sur la réserve 1.000, 51% du test classé « pleines ».

## stegastamp

| Source (session/méthode) | Réserve brut | Réserve réglé | Pleines (rés.) | Partielles (rés.) | Validation croisée hors réserve (pleines / partielles) | Déposée |
|---|---|---|---|---|---|---|
| assemblage/rang-reels2-pred | 0,930 | 0,930 | 0,918 | 0,944 | 0,950 / 0,934 | 02:09 |
| reels2/pred | 0,923 | 0,924 | 0,918 | 0,931 | 0,950 / 0,911 | 01:37 |
| assemblage/rang-reelg-pred | 0,915 | 0,914 | 0,894 | 0,938 | 0,924 / 0,932 | 23:40 |
| reelg/pred | 0,907 | 0,908 | 0,894 | 0,926 | 0,924 / 0,915 | 23:38 |
| **CHOIX DU CANDIDAT 38** | | **0,930** | 0,918 | 0,944 | | |

Images partielles de la réserve, selon le post-traitement (chaque étape s'ajoute à la précédente) : assemblage/rang-reelg-pred : seuil 0,931, boîte du décodeur 0,935 ; assemblage/rang-reels2-pred : seuil 0,941, boîte du décodeur 0,941 ; reelg/pred : seuil 0,916, boîte du décodeur 0,921 ; reels2/pred : seuil 0,926, boîte du décodeur 0,926.

Boîtes de la recherche par décodeur lues : 72 images de train, 135 de test (t1-assemblage/boxnet2/boxes_stegastamp_train_cadre.json, t1-assemblage/boxnet2/boxes_stegastamp_test_cadre.json).

Choix : pleines = `assemblage/rang-reels2-pred` {"assemblage/rang-reels2-pred": 1}, recette {"mode": "seuil", "sigma": 0, "t": -0.228, "clamp": false, "area": 8} ; partielles = `assemblage/rang-reels2-pred` {"assemblage/rang-reels2-pred": 1}, recette {"mode": "seuil", "sigma": 0, "t": -0.89, "clamp": false, "area": 8}. Population reconnue par le taux de bits justes de l'image entière (seuil 0.68) : exactitude sur la réserve 0.939, 54% du test classé « pleines ».

Fichiers ignorés : reel/p_dec0(2plis) [trustmark] : couverture partielle (train 92/170, test 340/340) ; reel/p_mixr(2plis) [trustmark] : couverture partielle (train 91/170, test 340/340) ; reel/mix(4plis) [stegastamp] : couverture partielle (train 143/170, test 340/340) ; reel/mix(4plis) [trustmark] : couverture partielle (train 144/170, test 340/340).
