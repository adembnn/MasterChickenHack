# T1 · session `t1-reel` : segmenteur entraîné, recalé sur les vraies images (TrustMark, StegaStamp, PixelSeal)

État au 22/11, vers 2 h 15 (horloge des machines). Décision débloquée : quelles cartes de localisation servent de base à la soumission T1 pour chaque filigrane.
Scripts : `E:\EAC-lab\t1s\t1-reel\` (version courante `reel7.py`, `predict.py`), H200 `~/eac/t1/t1-reel/` (h200b, h200d, h200g), Jülich `adem/t1-reel/` (premiers jobs).

Mots (une fois) : **segmenteur** = réseau qui dit, pour chaque case de l'image, « filigrané » ou « propre » ; **pli** = un cinquième des images mis de
côté pendant l'entraînement pour mesurer ; **CV** = validation croisée, score sur les 136 images hors réserve, chacune prédite par un modèle qui ne l'a
pas vue ; **réserve** = 34 images jamais utilisées pour apprendre ni régler ; **pleine** = image presque entièrement filigranée avec un trou propre ;
**partielle** = image propre avec un rectangle ou un objet filigrané ; **résidu** = ce que l'outil de filigrane ajoute à l'image ; **toile** = le cadre sur
lequel l'outil a calculé son résidu ; **logit** = score brut du réseau (positif = filigrané) ; **AUC** = chance qu'une case filigranée ait un score plus haut
qu'une case propre ; **affinage** = courte fin d'entraînement sur les vraies images ; **IoU du trou** = IoU de la classe propre sur les pleines.

## 1. Cartes livrées (mIoU officiel, seuil 0, sans post-traitement ; sorties de `reel*.py agg`)
| filigrane | carte | CV 5 plis | pleines | partielles | réserve | pleines | partielles | IoU du trou, hors-pli / réserve |
|---|---|---|---|---|---|---|---|---|
| TrustMark | `p_mixr` (départ de la nuit) | 0,650 | 0,626 | 0,675 | 0,598 | 0,623 | 0,578 | |
| TrustMark | **`f_s1`** (dans le candidat 22, 0,7695 public) | **0,735** | 0,716 | 0,755 | **0,678** | 0,673 | 0,682 | 0,485 / 0,402 |
| StegaStamp | `p_mixr` | 0,833 | 0,872 | 0,795 | 0,842 | 0,834 | 0,851 | |
| StegaStamp | **`f_s1`** | **0,913** | 0,927 | 0,900 | **0,907** | 0,898 | 0,919 | 0,863 / 0,804 |
| PixelSeal | `f_deca` (tronc décodeur affiné) | 0,643 | 0,663 | 0,625 | 0,627 | 0,670 | 0,553 | |
| PixelSeal | **`f_s1` + `f_deca`** (moyenne) | **0,669** | 0,687 | 0,652 | **0,656** | 0,692 | 0,594 | 0,387 / 0,442 (U-Net seul) |
Sortie collée pour les trois lignes livrées :
```
| f_s1 | trustmark | 5 | 0.725 | 0.735 | 0.716 | 0.755 | 0.993 | 0.000 | 0.735 | 0.678 | 0.673 | 0.682 | 0.678 |
| f_s1 | stegastamp | 5 | 0.910 | 0.913 | 0.927 | 0.900 | 0.999 | -0.750 | 0.915 | 0.907 | 0.898 | 0.919 | 0.907 |
| f_s1+f_deca | pixelseal | 5 | 0.668 | 0.669 | 0.687 | 0.652 | 0.989 | -0.500 | 0.670 | 0.656 | 0.692 | 0.594 | 0.652 |
```
(colonnes : variante, filigrane, plis, CV sans moyenne sur symétries, CV, CV pleines, CV partielles, AUC, seuil réglé en CV, CV à ce seuil, réserve, réserve
pleines, réserve partielles, réserve au seuil CV.) Fichiers : h200b `~/eac/t1/t1-reel/f_s1/pred_{train,test}_{trustmark,stegastamp}.npz` et
`f_s1+f_deca/pred_{train,test}_pixelseal.npz` ; copies `E:\EAC-lab\t1s\t1-reel\preds\` ; poids des 5 plis TrustMark et StegaStamp `weights\`.
Vérification de `predict.py` (régénère les prédictions depuis les poids) sur TrustMark `f_s1a` : `réserve recalculée depuis les poids, mIoU seuil 0 : 0.6632`
(tableau : 0,663), `masques identiques à 0.9999992819393382`.

## 2. La recette qui a produit ces cartes
1. **Exemples fabriqués fidèles**, vérifiés par mesure sur le réel : pleine = image filigranée en entier puis pixels propres recollés dans un trou ;
   rectangle = image source filigranée dans son propre cadre puis collée ; objet = filigrane au cadre de l'image entière gardé sur une forme. Fonds =
   images du hackathon des deux autres filigranes (propres pour le nôtre), formes = vrais masques des autres filigranes.
2. **Toile agrandie** (découverte, puis précisée par t1-residus) : le filigrane réel n'est pas calculé sur le cadre 512 mais sur une toile carrée de côté
   512 + s (s médian 32, jusqu'à ~80 ; rognage décentré ; rectangles : + 4 px). Preuve StegaStamp, 40 pleines : réel = bits justes 0,959, marge du décodeur
   5,87, logit d'un réseau « fabriqué seul » +9,1 au centre et +4,8 aux bords ; kit au cadre 512 : 0,988 / 6,92 / +11,7 / +12,5 ; kit sur toile +8 px : 0,984
   / 6,45 / +9,3 / +4,4. Effet : un réseau entraîné sans aucune vraie image passe de 0,658 à 0,820 (StegaStamp) et de 0,512 à 0,617 (TrustMark) en CV.
3. **Première couche en pleine résolution** (pas 1 au lieu de 2, poids pré-entraînés gardés) : le levier principal. À nombre de pas égal sur fabriqué seul,
   TrustMark réserve 0,691 à 8000 pas contre 0,588 à 20 000 pas pour la couche de pas 2 ; StegaStamp 0,891 contre ~0,83 ; PixelSeal : le U-Net apprend
   enfin (0,612 ; avec le pas 2 il restait à AUC 0,66).
4. **Pré-entraînement long sur fabriqué (8000 pas), puis affinage 1000 pas sur les vraies images, 5 plis**, moyenne des plis pour la réserve et le test,
   moyenne avec l'image retournée horizontalement (pas plus : 8 symétries font baisser la réserve TrustMark de 0,678 à 0,643).

## 3. Ce qui améliore encore (mesuré à pas égal, pas encore livré)
| variante TrustMark (un modèle, fabriqué seul, 8000 pas) | réserve | pleines | IoU du trou, hors réserve / réserve |
|---|---|---|---|
| `g_s1` : ancienne toile étroite | 0,691 | 0,683 | non relevé (0,39 à 5000 pas) |
| `g_c1` : toile réelle + trous à leur vraie place | 0,678 | 0,686 | 0,529 / 0,424 |
| `g_c4` : idem + tête d'appoint qui prédit le résidu signé (idée du critique, tour 3) | **0,707** | 0,726 | 0,549 / 0,496 |
StegaStamp avec la recette `g_c4`, à 5000 pas : réserve 0,901, trou 0,890 hors réserve (`g_s1` finissait à 0,891 / 0,839). Les trois filigranes sont
relancés avec cette recette sur h200g (`g_x1` puis `f_x1`, 12 000 à 16 000 pas), livrés à 2 h 50 sur h200b :
| livré (dossier h200b `~/eac/t1/t1-reel/`) | CV | réserve | IoU du trou, hors-pli / réserve |
|---|---|---|---|
| TrustMark `tm_f_x1/` | **0,769** | 0,696 | 0,563 / 0,427 |
| StegaStamp `ss_f_x1+f_c4/` (moyenne de deux modèles) | **0,931** | 0,923 | 0,910 / 0,846 |
| PixelSeal `ps_f_x1+f_s1+f_deca/` (moyenne de trois) | **0,682** | 0,661 | 0,438 / 0,459 |

## 4. Mission « trou propre des pleines » : ce qui est établi
- Sur `f_s1` TrustMark, par taille de trou (hors-pli) : < 2 % de l'image 0,24 ; 2-8 % 0,48 ; 8-20 % 0,65 ; > 20 % 0,60. Le meilleur seuil par image
  ne donnerait que 0,58 : la carte limite, pas le seuil.
- Critique tour 3, mesuré sur ces prédictions : les erreurs sont des régions entières, pas des bords (corriger toutes les cases à 2 cases du bord : 0,485 →
  0,520 au mieux ; 48 % des erreurs à plus de 16 px du bord) ; les petits trous sont « avalés ».
- `alpha.py` : en multipliant le résidu par 0,7 / 1 / 1,5 / 2 sur des pleines fabriquées fixes, l'IoU du trou vaut 0,498 / 0,627 / 0,689 / 0,694, et pour les
  trous < 2 % : 0,186 / 0,349 / 0,387 / 0,385. Doubler le signal ne sauve pas les petits trous : c'est la décision du réseau sur une petite zone qui limite.
- `detect.py` : la force du résidu TrustMark varie de ±76 % dans une image ; les faux trous tombent là où l'encodeur écrit peu (0,90 niveau contre 1,43).
- Formes (`shapes.py`, `band.py`) : aucune pleine n'a une zone filigranée en rectangle plus petit que l'image ; mon décalage aléatoire supprimait les trous
  qui longent 3 ou 4 côtés (1 % contre 31 % dans le réel), corrigé (`--pshift 0.15`).

## 5. Ce qui a échoué, et pourquoi (tout est mesuré)
- **Cartes tirées du message ou du décodeur**, toutes à AUC 0,49 à 0,77 contre 0,93 à 0,99 pour le réseau : corrélation avec le motif re-calculé (même dans
  le bon cadre), motif moyen du message, gradient du décodeur, accord du décodeur position par position, « effaceur » TrustMark, effacement par fenêtres lu
  par le décodeur, carte de re-tamponnage de t1-residus en fusion (+0,007). Le contenu de l'image écrase un signal de 1,3 niveau sur 255.
- **Tronc = décodeur du filigrane** : TrustMark CV 0,599 contre 0,650 au même stade (carte trop grossière, 8 px par case) ; utile seulement pour PixelSeal, en
  moyenne avec le U-Net.
- **Sans effet ou nuisible à pas égal** : drapeau pleine/partielle en entrée (0,603 contre 0,629) ; modèle dédié aux pleines (trou 0,256 contre 0,33) ;
  perte de Lovász (approximation dérivable de l'IoU) générale ou sur la classe propre ; perte sur la carte 512x512 avant regroupement (instable) ; aucune
  réduction sur deux étages (0,614 contre 0,640) ; tige à filtres passe-haut ; « force attendue du résidu » en canal d'entrée (fabriqué 0,73, réserve 0,50 :
  ne se transfère pas au réel) ; mélange de deux vraies images ; moyenne de plusieurs variantes d'un même modèle ; greffes d'objets réalistes par masques
  SAM (partielles 0,742 contre 0,751) ; trous alignés sur des régions SAM (hypothèse réfutée : IoU 0,28 entre régions SAM et vrais trous).
- **Post-traitement** (seuils par population, petites composantes, remplissage, rectangle, « jamais vide ») : +0,00 à +0,006 en CV emboîtée.
- **Recalage de rectangle par le décodeur** en recherche locale : IoU de boîte 0,82 → 0,86 seulement (fait mieux, en balayage complet, par t1-residus).
- Dégradations après filigrane (JPEG, WebP, flou, bruit, redimensionnement, zoom, rotation) : aucune n'explique l'écart au réel ; c'était la toile.
- Incident : h200d préemptée vers 22 h 25 avec des poids non rapatriés (retrouvés au redémarrage). Depuis, poids et prédictions sont copiés sur le PC à chaque
  étape ; des affinages sont aussi tombés par manque de mémoire GPU quand neuf entraînements partageaient la carte (relance automatique ajoutée).

## 6. Mesuré / supposé
Mesuré : tout chiffre de ce rapport, avec la commande qui l'a produit (`agg`, `holes.py`, `alpha.py`, `pad.py`, `shapes.py`, `band.py`, `fuse.py`, journaux
`log.jsonl` des entraînements). Bruit : ±0,017 en CV (136 images), ±0,03 sur la réserve (34 images), davantage par population.
Supposé : que la loi de la toile mesurée sur TrustMark vaut pour StegaStamp (non mesuré) ; que les 30 % publics du test ressemblent au train (la part
prédite filigranée est la même, 0,49 à 0,50) ; que l'avance de `g_c4` à 8000 pas se retrouve après affinage (en cours de mesure).

## 7. Critiques (un agent séparé à chaque tour)
- Tour 1 (`reports/critique/t1-reel-tour1.md`) : 4 défauts corrigés avant le premier job (fonds COCO attribués presque tous aux objets, confusion avec le
  filigrane des autres méthodes dans les exemples « alignés », passe-haut calculé en demi-précision, plantage d'une option).
- Tour 2 (`t1-reel-tour2.md`) : test de la marge par image, calibré sur fabriqué ; reprise aux points de contrôle corrigée.
- Tour 3 (`t1-reel-tour3.md`) : diagnostic « régions entières, pas bords » ; jeu fabriqué de contrôle rendu fixe ; tête d'appoint du résidu (retenue).

## 8. Pour reprendre
`python reel7.py train --method <m> --suffix <réserves> --fold -1 --nreal 0 --s1 1 --bs 16 --aux 1.0 --pshift 0.15 --pblob 0.15 --steps 8000 --tag g`
puis, pour f de 0 à 4 : `... --fold f --init g --nreal 6 --steps 1000 --lr 1e-4 --warm 0.1 --tag f`, puis `reel7.py agg --tag f --write 1`.
Prédictions test depuis les poids : `python predict.py --method <m> --tags <variante>[+<variante>] --tta 2`.
