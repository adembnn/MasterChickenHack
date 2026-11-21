# T1 : tableau de bord des methodes (candidat 1, 2026-11-21 17:37)

mIoU officiel sur le vrai train. **Reserve** = images d'index % 5 == 0 (30 a 39 par filigrane), jamais utilisees pour regler : c'est le chiffre honnete (bruit ±0,03).
« brut » = prediction telle que deposee (seuil 0). « regle » = lissage, seuil par population, fourchette de fraction et nettoyage regles hors reserve par `assemble.py`.
Pleines / partielles = score sur la reserve, par vraie population. Mis a jour a chaque nouvelle prediction deposee.

## pixelseal

| Source (session/methode) | Reserve brut | Reserve regle | Pleines (res.) | Partielles (res.) | Hors reserve regle | Deposee |
|---|---|---|---|---|---|---|
| assemblage/occl64 | 0,514 | 0,522 | 0,532 | 0,504 | 0,522 | 17:37 |
| assemblage/unetv1 | 0,277 | 0,452 | 0,436 | 0,478 | 0,451 | 17:18 |
| **CHOIX DU CANDIDAT 1** | | **0,522** | 0,532 | 0,504 | 0,522 | |

Choix : pleines = `assemblage/occl64` {"assemblage/occl64": 1} ; partielles = `assemblage/occl64` {"assemblage/occl64": 1}. Population reconnue par le taux de bits justes de l'image entiere (seuil 0.79) : exactitude reserve 1.000, 52% du test classe « pleines ».

## trustmark

| Source (session/methode) | Reserve brut | Reserve regle | Pleines (res.) | Partielles (res.) | Hors reserve regle | Deposee |
|---|---|---|---|---|---|---|
| assemblage/unetv1 | 0,559 | 0,566 | 0,567 | 0,565 | 0,596 | 17:17 |
| **CHOIX DU CANDIDAT 1** | | **0,566** | 0,567 | 0,565 | 0,596 | |

Choix : pleines = `assemblage/unetv1` {"assemblage/unetv1": 1} ; partielles = `assemblage/unetv1` {"assemblage/unetv1": 1}. Population reconnue par le taux de bits justes de l'image entiere (seuil 0.76) : exactitude reserve 1.000, 51% du test classe « pleines ».

## stegastamp

| Source (session/methode) | Reserve brut | Reserve regle | Pleines (res.) | Partielles (res.) | Hors reserve regle | Deposee |
|---|---|---|---|---|---|---|
| assemblage/unetv1 | 0,675 | 0,682 | 0,603 | 0,777 | 0,661 | 17:17 |
| **CHOIX DU CANDIDAT 1** | | **0,682** | 0,603 | 0,777 | 0,661 | |

Choix : pleines = `assemblage/unetv1` {"assemblage/unetv1": 1} ; partielles = `assemblage/unetv1` {"assemblage/unetv1": 1}. Population reconnue par le taux de bits justes de l'image entiere (seuil 0.68) : exactitude reserve 0.939, 54% du test classe « pleines ».

## Candidat 1 : score local

| | Reserve (honnete) | Pleines (res.) | Partielles (res.) | Hors reserve | Train entier apres re-reglage (optimiste) |
|---|---|---|---|---|---|
| pixelseal | **0,522** | 0,532 | 0,504 | 0,522 | 0,523 |
| trustmark | **0,566** | 0,567 | 0,565 | 0,596 | 0,589 |
| stegastamp | **0,682** | 0,603 | 0,777 | 0,661 | 0,664 |
| ensemble | **0,590** | 0,566 | 0,619 | 0,592 | 0,592 |

Fichier : `/p/scratch/training2666/destors1/MasterChickenHack/Hackathon_Watermark_Localisation_Task/adem/t1-assemblage/submission_candidat_1.pt` (16,734,690 octets, uint8 [1020,128,128]). Controle : le meme ecrivain applique au train puis re-note avec `wmloc.utils.miou_image` = 0.5919 (attendu 0.5919).
