# T1 · session t1-assemblage : formes, objets et assemblage final

Mis à jour le 21/11 à 19 h 45. Décision débloquée : quel fichier soumettre pour le track 1, et que faire de l'idée « formes et objets » d'Adem.
Mots employés (une fois) : **mIoU** = le score officiel, moyenne par image de l'IoU de la zone propre et de l'IoU de la zone filigranée (IoU = surface commune
divisée par surface totale couverte) ; **réserve** = les images de train d'index % 5 == 0 (30 PixelSeal, 39 TrustMark, 33 StegaStamp), jamais utilisées pour
régler quoi que ce soit ; **hors-pli** = prédiction faite par un modèle qui n'a pas vu l'image à l'entraînement ; **SAM** = Segment Anything (Meta, licence
Apache 2.0), un réseau qui découpe une image en objets sans connaître la tâche ; **logit** = la sortie brute d'un réseau avant seuil (> 0 = filigrané).

## 1. Où on en est
| Étape | Réserve | Statut |
|---|---|---|
| Candidat 1 (17 h 39) : U-Net v1 + occlusion PixelSeal, seuil par population | 0,590 | dépassé |
| Soumission v3 de l'orchestrateur (t1-reel `mixr` + seg v2 PixelSeal) | 0,663 | public 0,674 |
| Soumission v4 (PixelSeal = `run_D_res512` de t1-pixelseal) | 0,679 | public 0,697 |
| Soumission v6 (StegaStamp `p_mixr`, PixelSeal fusion de t1-pixelseal) | 0,692 | public 0,706, base à battre |
| Candidat 4 : vote par région sur `mixr` | 0,697 | pas envoyé (+0,005 seulement contre la v6) |
| Candidat 5 : + boîte du décodeur TrustMark, vote par région sur `p_mixr` et PixelSeal | 0,723 | remplacé par le 6 |
| **Candidat 5, soumis par l'orchestrateur (soumission 1540)** | **0,723** | **public 0,7384** (v6 : 0,7059) |
| Candidat 6 : + boîtes TrustMark affinées | 0,727 | vérifié, +0,004 contre le 5 : pas d'envoi ; correction de décalage retirée (fuite de 0,003 mesurée par le vérificateur) |
| Candidat 7 : vote TrustMark refait sur `p_mixr`, boîtes sans correction | 0,726 | pas d'envoi |
| Candidat 8 : + boîtes StegaStamp | 0,722 | pas d'envoi |
| **Candidat 11 : cadres des rectangles calés au pixel (TrustMark par t1-residus, StegaStamp par `r3ss.py`)** | **0,737** | remis au vérificateur ; +0,014 contre le 5 sur la réserve, environ +0,017 en écart apparié sur les 510 images |

| **Candidat 17 : + au moins 0,5 à 4 % de cases propres sur les images pleines, 66 cadres TrustMark sur le test** | **0,738** | remis au vérificateur ; verdict sur le 11 : gain réel d'environ +0,015, sous le seuil |

| **Candidat 22 : base TrustMark `f_s1` / `f_s1a` de t1-reel sous le vote par région, les cadres calés et les boîtes** | **0,751** | remis au vérificateur à 22 h 30 ; +0,028 contre le candidat 5 sur la réserve |

Verdicts du vérificateur (22 h 45) : candidat 20, feu vert (+0,0263 contre le 5, IC95 [+0,010 ; +0,045]) ; candidat 22, feu vert (+0,0284,
IC95 [+0,012 ; +0,047]), à envoyer à la place du 20. Prochain seuil après envoi du 22 : 0,7714 de réserve.
Candidat 24 (meilleur hors ligne, 0,7536) : le 22 + la fusion des images pleines TrustMark de t1-residus (`pfuse3.py`, re-tamponnage + effaceur,
pleines hors-pli 0,716 → 0,724 sur la base `f_s1`) + 72 cadres calés sur le test. +0,002 contre le 22, pas de demande d'envoi.

**Candidat 22 envoyé par l'orchestrateur à 23 h 50 (soumission 1984) : 0,7695 public, premier du track 1** (prédit +0,028, obtenu +0,031).
Candidat 26 (hors ligne, 0,7538) : le 24 avec les pleines PixelSeal sur la carte `f_deca` de t1-reel (réserve 0,647 → 0,667) ; PixelSeal hors
réserve +0,013, réserve +0,0004. Prochain seuil : 0,7714.

**Candidat 30 (22/11, 0 h 10) : réserve 0,7761, +0,0246 contre le 22 envoyé, remis au vérificateur.** PixelSeal 0,647, TrustMark 0,769,
StegaStamp 0,912. Le saut vient de la carte StegaStamp en pleine résolution de t1-reel (brute : 0,907 sur la réserve contre 0,842), sous le
vote par région (partielles hors-pli 0,903 → 0,922) et les cadres calés. Défaut de mon réglage trouvé et corrigé à cette occasion : la grille de
seuils, faite de quantiles seuls, était trop grossière pour une carte tranchée ; mon assemblage faisait alors moins bien que la carte brute sur
les images classées pleines (−0,018). Avec une grille régulière en plus, StegaStamp passe de 0,899 (candidat 28) à 0,912.

**22/11, 0 h 50 : l'agent de Yassir a envoyé `submission_candidat_27.pt` : 0,7935 public.** C'était une sortie intermédiaire de `assemble.py`
(déjà avec la carte StegaStamp `f_s1`, mais PixelSeal à 0,626), pas un candidat : erreur de nommage de ma part, corrigée (intermédiaires renommés
`assemblage_<n>.pt`, `LISEZMOI.md` dans le dossier). Nouvelle base : candidat 27, réserve 0,7627, seuil 0,7827.
Contre le 27 : candidat 30 +0,0133 (IC95 du vérificateur [+0,005 ; +0,023]) ; candidat 33 (le 30 + mélange de cartes PixelSeal retenu en
validation croisée : `f_deca` + `G` sur les pleines) +0,0143, réserve 0,7771. Gain réel, sous le seuil de +0,02 : pas de demande d'envoi.

**22/11, 2 h : nouvelle règle de feu vert** (orchestrateur) : écart apparié ≥ +0,02 contre le dernier envoi en validation croisée sur les 510 images,
IC95 hors de 0, positif sur les trois filigranes ; la réserve ne sert plus que de contrôle (elle a été regardée une trentaine de fois).
Outil : `cv5.py` refait toute la procédure (population, normalisation, choix de source, seuils, seuil des boîtes) sur 4 plis et note le 5e.
- Candidat 30 contre le 27 : +0,0033, IC95 [−0,004 ; +0,010]. Il n'était donc pas établi comme meilleur ; envoyé, il a fait −0,0004 au public.
- **Candidat 35 contre le 27 : +0,0252, IC95 [+0,017 ; +0,034]** (0,8138 contre 0,7885 ; PixelSeal +0,029, TrustMark +0,031, StegaStamp +0,015 ;
  réserve 0,7922 contre 0,7627). Remis au vérificateur à 2 h 35. Bases : `tm_f_x1`, `ss_f_c4+f_s1`, `f_s1+f_deca` de t1-reel, avec le vote par
  région refait sur chacune (partielles hors-pli 0,781 → 0,811 ; 0,914 → 0,934 ; 0,652 → 0,697), les cadres calés et les boîtes.

Candidat 22 par filigrane (réserve) : PixelSeal 0,648 (inchangé depuis le candidat 5) ; TrustMark 0,764 (pleines 0,673, partielles 0,843) ; StegaStamp
0,842. Fichier `h200b:~/eac/t1/t1-assemblage/submission_candidat_22.pt`, validé. Le saut vient des cartes TrustMark à pleine résolution de t1-reel
(réserve brute 0,663 puis 0,678 contre 0,598) : le vote par région, refait dessus, donne 0,795 sur les partielles hors-pli (0,735 pour la carte seule).
Le candidat 20 est la même chose avec `f_s1a` partout (0,749). Le PixelSeal `v3sam` de t1-pixelseal (0,641 brut) ne bat pas celui du candidat 5 (0,648).

Verdict du vérificateur sur le candidat 11 : chiffres reproduits, +0,0136 sur la réserve (IC95 [−0,004 ; +0,033]), +0,0157 sur les 510 images
(IC95 [+0,008 ; +0,024]). Sa demande, appliquée dans le 17 : 18 images TrustMark et 3 StegaStamp du test sortaient « tout filigrané », ce qui vaut
au plus 0,5 ; il n'y en a plus. Les candidats 13 et 15 sont à ignorer (borne erronée, corrigée). Calage par re-tamponnage sur PixelSeal : propriété
présente (R3 0,76 au cadre, 0,94 à 6 px), gain nul en pratique (0,840 → 0,849), non intégré.

Calage au pixel (idée de t1-residus) : l'encodeur du filigrane ajoute 2 à 3 fois moins d'énergie quand on lui redonne une image déjà filigranée
dans exactement le même cadre ; on cherche le rognage qui minimise cette énergie, puis on rétrécit le cadre de 3,5 px (le masque vrai est plus petit
que le cadre du filigrane). TrustMark : recouvrement 0,866 → 0,958 sur 37 rectangles, réserve 0,688 → 0,722. StegaStamp : R3 moyen 0,70 au vrai
cadre contre 0,93 à 6 px, recouvrement 0,906 → 0,973 sur 8 rectangles ; hors réserve +0,017, réserve −0,004 (13 partielles).
Même idée sur les images pleines TrustMark (`restamp.py`) : AUC 0,57 sur 12 images ; le contraste existe quand le cadre est trouvé (0,49 dans le
filigrane contre 1,08 dans le trou), mais il ne l'est que 3 fois sur 12.

Les candidats 5, 7 et 8 sont à égalité dans le bruit de la réserve (±0,03 par filigrane). Le post-traitement des partielles est arrivé au bout de ce
que les cartes actuelles permettent. Mesures de la fin de soirée :
- Boîtes StegaStamp (`boxnet.py`, recherche amorcée par la carte du réseau, 7 s par image) : 82 % des rectangles retrouvés à z ≥ 5 avec un recouvrement
  de 0,854, aucun objet pris à tort ; gain nul dans l'assemblage (partielles en validation croisée 0,834 → 0,836), car le réseau fait déjà 0,82 dessus.
- Vote v2 avec union de masques SAM (`rank2.py`) : pas mieux que le v1 (TrustMark 0,700 contre 0,712 hors-pli).
- **Images pleines (`holes.py`, cartes hors-pli)** : StegaStamp, 212 trous touchés sur 281 mais mal dessinés (parfaits : 0,974 au lieu de 0,864) ;
  TrustMark, 170 vrais trous sur 279 jamais touchés ; PixelSeal, 151 sur 272. Retirer les faux trous ne vaudrait que +0,015. Le prochain gain est
  dans la détection des trous par les réseaux, pas dans la fusion.

Candidat 6 par filigrane (réserve, v6 → candidat 6) : TrustMark 0,597 → 0,691 (partielles 0,585 → 0,755) ; PixelSeal 0,636 → 0,648 ; StegaStamp 0,842 → 0,843.
Fichier `h200b:~/eac/t1/t1-assemblage/submission_candidat_6.pt` (16 734 690 octets, validé, contrôle train 0,7474 = attendu).

## 1 bis. Ce qui fait le gain du candidat 6 (tout est réglé hors réserve)
- **Vote par région appris** (`rank.py`) : pour chaque image partielle, un petit réseau (27 indices par masque SAM : accord avec la carte, décodeur, forme)
  prédit le gain de mIoU si on rend ce masque à la place du masque seuillé ; appris en 4 plis hors réserve. Partielles hors-pli : StegaStamp sur `p_mixr`
  0,806 → 0,834 (objets 0,793 → 0,859), TrustMark sur `mixr` 0,649 → 0,697 (objets 0,634 → 0,724), PixelSeal sur `G` 0,650 → 0,686 (objets 0,719 → 0,848).
  Il dégrade un peu les rectangles (−0,00 à −0,04) : c'est la boîte du décodeur qui les traite.
- **Boîte du décodeur TrustMark** (`bs.py` de t1-residus, 247 images) : rectangles z médian 9,6, objets 4,2 ; boîte rendue si z ≥ 5.
  Partielles TrustMark en validation croisée : 0,697 → 0,792.
- **Affinage des boîtes** (`boxref.py`) : le z sature, donc on remonte la marge brute du décodeur avec un rognage exact, puis on retire le décalage
  médian relatif par bord. Recouvrement avec la vraie boîte (41 rectangles) : 0,790 → 0,858 → 0,870. Partielles TrustMark : 0,792 → 0,806 (réserve 0,755).

## 2. Mission A : l'idée « formes et objets », mesurée
**Test oracle** (plafond : chaque région reçoit la vraie étiquette majoritaire ; sortie de `shapes_oracle.py`, 510 images, et `sam_oracle2.py` / `samsel.py`) :
| Découpe | Régions par image | Plafond mIoU |
|---|---|---|
| Grille carrée de 32 px | 256 | 0,819 |
| Superpixels SLIC (régions de couleur homogène) | 242 | 0,826 |
| Superpixels SLIC | 994 | 0,908 (grille 16 px, 1 024 cases : 0,902) |
| Felzenszwalb (autre découpe par couleur) | 123 | 0,661 |
| SAM ViT-H, toutes images (60 images) | 308 | 0,71 |
| SAM, images pleines | | 0,67 |
| SAM, partielles « rectangle » (130 images) | | 0,57 |
| **SAM, partielles « objet » (123 images)** : un seul masque SAM | 67 candidats | **0,929** (rien rendre : 0,444) |

Ce que ça veut dire :
- Les contours de couleur n'apportent rien : une découpe par couleur ne fait pas mieux qu'une grille de même finesse. Les taches propres des images pleines sont des
  formes lisses tirées au hasard, et les rectangles sont posés n'importe où (planche `E:\EAC-lab\t1\planche_train.png`).
- **L'idée marche pour une famille précise** : les images partielles dont la zone filigranée est un objet greffé, soit une image sur quatre. Là, le bon
  masque SAM existe 91 % du temps et vaut 0,93.
- Le point dur devient : choisir le bon masque parmi ~67 sans la vérité. Mesuré sur les 123 objets (mIoU, part de bons choix) : décodeur quand on ne garde
  que la région 0,598 (33 %) ; chute du décodeur quand on efface la région 0,594 (33 %) ; les deux 0,626 (37 %) ; carte du U-Net v1 StegaStamp 0,773 (61 %).
  Le décodeur seul ne suffit pas ; une bonne carte + SAM, si.

**Accrochage de forme** (`asm.snap_core`) : on remplace le masque flou de la carte par la forme candidate qui lui ressemble le plus (union d'au plus 3 masques
SAM, ou rectangle ajusté), si leur IoU dépasse un seuil réglé hors réserve. Mesuré hors-pli sur les cartes `mixr` (sortie de `snapdiag.py`) :
| Filigrane | Objets : seuil → forme SAM | Rectangles : seuil → forme | Toutes partielles |
|---|---|---|---|
| TrustMark (65 images) | 0,642 → 0,735 | 0,667 → 0,643 | 0,649 → 0,680 |
| StegaStamp (69 images) | 0,783 → 0,827 | 0,781 → 0,778 | 0,782 → 0,800 |
Gain réel mais modeste (+0,01 à +0,015 sur le score d'un filigrane) : la carte désigne le bon objet moins souvent que le plafond ne le permet.
**Rectangle ajusté à la carte** (rectangle de somme de logits maximale, `asm.best_rect`) : aucun gain (StegaStamp 0,781 → 0,781 ; candidat 1 : 0,768 → 0,741).
Les rectangles se retrouvent par le décodeur (le filigrane y est calé sur le rectangle : rogner la bonne boîte fait relire le message), pas par la carte :
c'est la recherche de boîte de t1-residus et t1-pixelseal, que l'assemblage fusionne (§3).

**Lois de formes du train** (`shapes_oracle.json`) :
| Population | Part filigranée, médiane [10 % ; 90 %] | Composantes | Rectangles parfaits | Touche le bord |
|---|---|---|---|---|
| Pleines (255) | 0,94 [0,74 ; 0,99] | 2 [1 ; 8] | 0 % | 73 à 87 % |
| Partielles (255) | 0,04 [0,005 ; 0,2 à 0,3] | 1 [1 ; 5] | 51 % | 27 à 35 % |
Rapport hauteur/largeur des rectangles : TrustMark médiane 1,5 (90 % sous 2,6), PixelSeal et StegaStamp jusqu'à 16. Aucun alignement sur une grille de 4 ou 8 px.
**Population sans la vérité** : taux de bits justes du décodeur sur l'image entière > 0,75 = pleine. Exact à 97,6 % (PixelSeal), 98,2 % (TrustMark), 98,8 %
(StegaStamp). Les 10 erreurs sur 510 sont des cas intermédiaires (partielles de 21 à 44 %, pleines de 50 à 80 %) ; une régression logistique sur les bits des
quarts d'image ne fait pas mieux (`popdiag.py`). Seuil simple gardé.

## 3. Mission B : `assemble.py` (bibliothèque `asm.py`)
Ce qu'il fait, dans l'ordre :
1. lit toutes les prédictions déposées (formats dans la note partagée), vérifie la couverture (170 images de train et 340 de test par filigrane) ;
2. range chaque image en pleine ou partielle (bits justes) ;
3. règle, hors réserve et par population, une recette par source : lissage, seuil, fourchette de la part filigranée, nettoyage des petites composantes ;
   pour les partielles, en plus : rectangle ajusté, accrochage de forme, **fusion avec la boîte du décodeur** (si la boîte a une confiance z au-dessus
   d'un seuil réglé, on rend la boîte, sinon le réseau) ;
4. choisit en validation croisée (4 plis hors réserve) la meilleure source ou un mélange (moyenne de logits centrés-réduits, ajout glouton) ;
5. écrit `submission_candidat_<n>.pt` (uint8 [1020,128,128], 16,7 Mo), le relit, re-note le même écrivain sur le train avec `wmloc.utils.miou_image`
   et passe `tools/validators/validate_t1.py` ;
6. écrit le tableau de bord `reports/t1-tableau.md`.
Un modèle qui a vu les images hors réserve à l'entraînement (« pas hors-pli » : seg v2, `run_D_res512`) affiche 0,84 dessus contre 0,53 à 0,62 sur la
réserve : rien ne peut être réglé honnêtement sur ses sorties. L'assemblage garde alors son seuil 0 et ne le retient que s'il bat le choix de 0,03 sur la réserve.

## 4. Critique séparée (`reports/critique/t1-assemblage-tour1.md`) et corrections
Verdict : pas de fuite de la réserve, écriture du fichier juste. Corrigé : choix des sources et des mélanges en validation croisée (plus sur le score de
réglage) ; re-réglage final limité au seuil ; fourchette des partielles élargie à 35 %. Non retenu après mesure : classement pleine/partielle plus riche.
Reste ouvert : choisir le masque qui maximise l'IoU attendue image par image au lieu d'un seuil global.

## 5. Ce qui est mesuré, ce qui est supposé
- Mesuré : tous les chiffres des tableaux (sorties de commandes sur le vrai train). La réserve prédit le public à +0,01 près (0,663 → 0,674 ; 0,679 → 0,697).
- Supposé : que les partielles du test ont la même part d'objets et de rectangles que le train (51 % de rectangles).
- Bruit : ±0,03 sur un filigrane en réserve ; un écart plus petit ne décide rien.

## 6. Ce qui a échoué et pourquoi
- Vote par région sur toutes les images : plafond 0,71, les formes ne suivent les objets que pour un quart des images.
- Rectangle ajusté à la carte : la carte est trop floue sur les rectangles, le rectangle trouvé n'est pas meilleur que le masque seuillé.
- Classement pleine/partielle par régression : pas mieux que le seuil.
- **Pixels partagés entre images** (`shared.py`) : si deux images du jeu venaient de la même scène, leur différence donnerait le masque exact.
  1,4 million de blocs 16x16 comparés sur les 1 530 images : aucune collision, à la même position ni ailleurs.
- **Autoencodeur du générateur** (`vaetest.py`) : encoder puis décoder l'image avec le VAE de FLUX, SD3, SDXL ou SD 1.5 pour retrouver l'image propre.
  Erreur de reconstruction dans les zones propres : 2,4 à 3,8 niveaux (FLUX, le meilleur), plus que le filigrane (1,2 niveau) ; AUC de l'énergie de
  l'écart 0,50 à 0,57. Les images ne sont pas assez proches de la sortie d'un de ces VAE pour que l'écart révèle le filigrane.
- **Couture au bord des rectangles** (`seam.py`) : le vrai bord ne ressort pas dans le saut entre lignes voisines (13 % contre 12 % au hasard).
- **Bibliothèque finie de formes** (`shapedup.py`) : meilleur IoU d'une tache propre avec celle d'une autre image, médiane 0,36 (0,62 à translation près).
- Calcul en local sur le PC : chargement trop lent ; tout tourne maintenant sur h200b.

## 7. Outils ajoutés
`segment-anything` 1.0 (paquet léger) installé dans `.venv` sur Jülich, h200 et h200b ; poids `sam_vit_h_4b8939.pth` (2,5 Go, Apache 2.0) téléchargés depuis
les nœuds de login. Scripts : `E:\EAC-lab\t1s\t1-assemblage\` (copie de travail sur h200b `~/eac/t1/t1-assemblage/`).
