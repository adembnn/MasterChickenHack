# T1 · session `t1-reel` : segmenteur entraîné, recalé sur les vraies images (TrustMark, StegaStamp, PixelSeal)

Rapport final, 22/11, 10 h 55 (horloge du PC). Décision débloquée : quelles cartes de localisation servent de base à la soumission T1 pour chaque filigrane.
Code : `tracks/t1-watermark/t1-reel/` (version courante `reel12.py`, `c2.sh` / `c3.sh` pour les chaînes, `ens.py` / `ens2.py` / `cmp.py` pour les moyennes et les
comparaisons, `predict.py` pour régénérer depuis les poids). Copie de travail, prédictions et poids : `E:\EAC-lab\t1s\t1-reel\` (`preds\`, `weights\`).

Mots (une fois) : **segmenteur** = réseau qui dit, pour chaque case de 4 x 4 pixels, « filigrané » ou « propre » ; **pli** = un cinquième des images mis de
côté pendant l'entraînement pour mesurer ; **CV** = validation croisée, score sur les images hors réserve, chacune prédite par le modèle qui ne l'a pas
vue ; **réserve** = images d'index % 5 == 0 (30 à 39 par filigrane), jamais utilisées pour apprendre ni régler ; **pleine** = image presque entièrement
filigranée avec un trou propre ; **partielle** = image propre avec un rectangle ou un objet filigrané ; **IoU du trou** = IoU de la classe propre sur les
pleines ; **toile** = cadre sur lequel l'outil de filigrane a calculé son résidu ; **affinage** = courte fin d'entraînement sur les vraies images ;
**IC95** = intervalle à 95 % de l'écart apparié image par image (10 000 tirages, `cmp.py`). Bruit typique : ±0,017 en CV, ±0,03 en réserve.

## 1. Cartes à prendre (h200b `~/eac/t1/t1-reel/`, clés `names` / `logits` float16 N x 128 x 128, > 0 = filigrané)
| filigrane | dossier | composition | CV | réserve | IoU du trou hors pli / réserve |
|---|---|---|---|---|---|
| TrustMark | **`tm_x2+c5+r50b+3hp/`** | moyenne de 6 : `f_x2`, `f_c5`, `f_r50b` + 3 à canaux passe-haut `f_x1hp`, `f_xhp3`, `f_xhp5` | 0,787 | **0,743** | 0,587 / 0,506 |
| StegaStamp | **`ss_f_x1+f_c4/`** | moyenne de 2 | 0,931 | 0,923 | 0,910 / 0,846 |
| PixelSeal | **`ps_pop_C/`** | par population prédite : pleines = 9 généralistes ; partielles = `f_st`, `f_x7st`, `f_x8st`, `f_x2`, `f_x3` | 0,727 | 0,703 | 0,492 / 0,522 |
Fichiers par population, pour le mélange de t1-assemblage : `ps_pleines_9/`, `ps_partielles_5/`, `tm_pleines_x2+x1hp/` ; chaque modèle seul :
`ps_modeles/<tag>/`, `tm_modeles/<tag>/`. Écarts appariés contre les cartes livrées précédemment (sortie de `cmp.py`) :
```
TrustMark tm_x2+c5+r50b+3hp contre tm_f_x2+f_c5+f_r50b : hors réserve 0.7874 contre 0.7849 +0.0025 IC95 [-0.0072 ; +0.0118] ; réserve 0.7426 contre 0.7198 +0.0228 IC95 [+0.0013 ; +0.0495]
PixelSeal ps_pop_C contre ps_pop_x2+x2b+x1+dx__st+x2 : hors réserve 0.7267 contre 0.7232 +0.0035 IC95 [-0.0013 ; +0.0082] ; réserve 0.7029 contre 0.6973 +0.0056 IC95 [-0.0028 ; +0.0140]
PixelSeal ps_pop_x2+x2b+x1+dx__st+x2 contre ps_f_x2+f_x1+f_dx : hors réserve +0.0195 IC95 [+0.0066 ; +0.0336] ; réserve +0.0128 IC95 [-0.0009 ; +0.0305]
```
Après assemblage (mesure de t1-assemblage, contre le candidat 38) : TrustMark à un modèle passe-haut +0,0027 en réserve, PixelSeal `ps_pop_C` +0,0088 ;
les gains portés par les partielles sont en grande partie déjà captés par le vote par région et les boîtes. Ce qui reste à gagner est le trou des pleines.
Vérification de `predict.py` (régénère depuis les poids) sur PixelSeal `f_x2` : `réserve recalculée depuis les poids, mIoU seuil 0 : 0.6819` (tableau 0,682) ;
`test : 340 images, masques identiques sur 0.9999989 des cases`. Poids de tous les modèles livrés sur le PC (`weights\`, 5 plis chacun).

## 2. Progression des cartes livrées (CV / réserve)
| étape | TrustMark | StegaStamp | PixelSeal |
|---|---|---|---|
| départ de la nuit `p_mixr` | 0,650 / 0,598 | 0,833 / 0,842 | 0,489 / 0,514 |
| première couche pleine résolution `f_s1` (candidat 22) | 0,735 / 0,678 | 0,913 / 0,907 | 0,669 / 0,656 (avec tronc décodeur) |
| toile réelle + tête résidu `f_x1` | 0,769 / 0,696 | 0,931 / 0,923 | 0,682 / 0,661 |
| moyennes de modèles | 0,785 / 0,720 | 0,931 / 0,923 | 0,704 / 0,685 |
| moyenne par population (PixelSeal) | | | 0,723 / 0,697 |
| final | 0,787 / 0,743 | 0,931 / 0,923 | 0,727 / 0,703 |

## 3. La recette (ce qui a marché, dans l'ordre de ce que ça a rapporté)
1. **Première couche en pleine résolution** (pas 1 au lieu de 2, poids pré-entraînés gardés) : le levier principal. À pas égal sur fabriqué seul,
   TrustMark réserve 0,691 à 8000 pas contre 0,588 à 20 000 pas ; PixelSeal : le U-Net apprend enfin (avec le pas 2, AUC 0,66).
2. **Toile agrandie** (découverte ici, précisée par t1-residus) : le filigrane réel est calculé sur une toile carrée de 512 + s px (s médian 32, jusqu'à ~80
   pour TrustMark, ~100 pour PixelSeal), rognée au hasard. Un réseau entraîné sans aucune vraie image passe de 0,658 à 0,820 (StegaStamp) et de 0,512 à
   0,617 (TrustMark) en CV.
3. **Exemples fabriqués fidèles** : pleine = image filigranée en entier puis pixels propres recollés dans un trou aux vraies formes, gardées à leur place
   (`--pshift 0.15`) ; rectangle = source filigranée dans son cadre puis collée ; objet = filigrane de l'image entière gardé sur une forme. Fonds = images
   des deux autres filigranes.
4. **Tête d'appoint qui prédit le résidu signé** (`--aux 1.0`, idée du critique, tour 3) : réserve TrustMark 0,707 contre 0,678 à 8000 pas.
5. **Pré-entraînement long sur fabriqué (16 000 à 24 000 pas), affinage 1000 pas sur le réel, 5 plis**, moyenne des plis, image + image retournée.
6. **Moyennes de modèles de la même recette, autres graines et autres exemples fabriqués** : +0,01 à +0,02 en CV à chaque fois.
7. **PixelSeal : première couche à filtres passe-haut appris** (`--st 1 --hp 2`) pour les partielles : partielles 0,719 contre 0,701 ; vrais rectangles de
   petit côté < 64 px lus à 0,289 contre 0,202 ; mais pleines 0,666 contre 0,706. D'où la **moyenne par population** (`ens2.py`) : population décidée par
   la prédiction (plus de la moitié des cases > 0), juste sur 170 / 170 images.
8. **TrustMark : canaux passe-haut en entrée** (`--hp 2`, image moins sa version floutée, 3 échelles) : CV égale, réserve nettement meilleure (`f_x1hp`
   0,746 seul contre 0,729 pour le meilleur modèle sans ; trou des pleines en réserve 0,519 contre 0,468 dans la moyenne avec `f_x2`). Six modèles de ce
   type entraînés : réserve 0,746, 0,720, 0,723, 0,746, 0,735 ; la moyenne à 3 (0,743) fait mieux que la moyenne à 5 (0,739).

## 4. Ce qui a échoué (tout est mesuré, à pas égal contre un témoin)
- **Cartes tirées du message ou du décodeur** (motif re-calculé, gradient, accord position par position, effaceur, effacement par fenêtres, re-tamponnage) :
  AUC 0,49 à 0,77 contre 0,93 à 0,99 pour le réseau. Le contenu écrase un signal de 1,3 niveau sur 255.
- **Tronc = décodeur du filigrane** : utile seulement pour PixelSeal, en moyenne (`f_dx`).
- **Variantes d'entraînement sans effet ou nuisibles** : drapeau pleine/partielle, Lovász, perte à 512 px, 8 symétries à l'inférence, mélange de vraies
  images, greffes SAM, trous alignés sur SAM, force attendue du résidu en entrée, sous-réduction s2, tige passe-haut sur TrustMark.
- **Pistes « trous » d'Adem**, lues à 8000 pas (mIoU hors réserve / réserve, témoin `g_c5` 0,756 / 0,683) : petits trous sur-représentés 0,757 / 0,694
  (neutre, PixelSeal négatif après affinage 0,673 / 0,645 contre 0,683 / 0,651) ; trous déplacés au hasard 0,749 / 0,704 (signes opposés) ; paires
  contrefactuelles de Yassir 0,747 / 0,686 (négatif) ; classifieur vraie / fabriquée AUC 0,51 au bord et dans le trou (rien ne trahit le fabriqué).
- **Tour PixelSeal du vérificateur** : petits rectangles sur-représentés 0,682 contre 0,683 ; perte doublée près des frontières 0,769 / 0,701 contre
  0,762 / 0,703 ; détecteur local de vignettes AUC 0,74 / 0,86 / 0,89, sous la carte de `f_st` sur le même test (0,84 / 0,93 / 1,00) ; adaptation au test
  image par image −0,02 hors réserve, −0,02 à −0,06 en réserve (elle grossit la zone et casse des images justes).
- **Spécialistes** : pleines seules (`xfu` 0,702 / 0,719 sur les pleines contre 0,723 / 0,732 pour la moyenne), partielles seules (`xsp` 0,702 / 0,621),
  modèle PixelSeal dédié aux trous avec canaux passe-haut (en dessous du généraliste au même pas). **Modèle TrustMark dédié aux trous** (`tmhole`, 90 % de
  pleines fabriquées, affinage sur les vraies pleines seulement) : avant affinage, trou 0,560 / 0,458 contre 0,502 / 0,420 ; après affinage 0,553 / 0,475,
  et il fait baisser la réserve dans la moyenne (0,737 contre 0,743).
- **Pseudo-étiquetage des images de test** : 0,763 contre 0,775 et 0,759 contre 0,762. **Affinage deux fois plus long** : +0,001 à +0,005, non établi.
- **Post-traitement** (seuils, composantes, remplissage, rectangle, « jamais vide ») : +0,00 à +0,006 en CV emboîtée. Seuil relatif pour les images
  prédites partielles sans aucune case (logit lissé > maximum − 1) : PixelSeal 0,7280 / 0,6974 contre 0,7232 / 0,6973, TrustMark 0,7891 / 0,7270 contre
  0,7849 / 0,7198 ; transmis au vérificateur, non livré dans mes fichiers.

## 5. Ce qui est établi sur les trous et les petites zones
- Les erreurs sont des régions entières, pas des bords (critique tour 3) ; doubler le résidu ne sauve pas les petits trous (`alpha.py`) : c'est la
  décision du réseau sur une petite surface qui limite, pas la force du signal.
- Taille des vrais trous : médiane 5,6 % (TrustMark), 5,9 % (PixelSeal) ; 25 % et 20 % sous 2 %. Les petits trous sont déjà mal lus sur le fabriqué
  (< 1 % : 0,28 fabriqué / 0,49 réel), donc ce n'est pas un écart de domaine.
- Petits rectangles PixelSeal : 74 % des vrais rectangles ont un petit côté < 64 px ; même fabriqués et vus à l'entraînement, ils sont lus à 0,095 sous
  32 px et 0,254 de 32 à 64 px. Sur la carte PixelSeal livrée, les pleines au trou < 8 % portent 36 % de la perte, les rectangles < 3 % en portent 26 %.
- TrustMark : la validation croisée est optimiste de 0,05 à 0,08 sur la réserve, dès les modèles entraînés sans aucune vraie image (0,782 / 0,702) ; aucune
  fuite par la banque de formes (`jumeaux.py` : aucun masque identique entre filigranes). Seule la réserve TrustMark s'est retrouvée au public.

## 6. Mesuré / supposé ; critiques
Mesuré : tout chiffre de ce rapport, avec la commande qui l'a produit (`reel12.py agg`, `ens.py`, `ens2.py`, `cmp.py`, `trous.py`, `bandes.py`, `disc.py`,
`p3a.py`, `p5.py`, `p5cmp.py`, `rects.py`, `pertes.py`, `rel.py`, `adapt.py`, journaux `log.jsonl`). Supposé : que les 30 % publics du test ressemblent au
train (part prédite filigranée 0,497 à 0,50 au test comme au train) ; que la loi de la toile mesurée sur TrustMark vaut pour StegaStamp.
Critiques séparées : tours 1 à 3 (`reports/critique/t1-reel-tour1.md` à `-tour3.md`), tour PixelSeal du vérificateur (`t1-tour-pixelseal.md`) ; le
vérificateur a recalculé et confirmé la carte PixelSeal par population (+0,0195 hors réserve, IC95 [+0,007 ; +0,033]). Tableau des pistes sur l'issue #3.

## 7. Incidents
- h200d préemptée vers 22 h 25 (poids retrouvés au redémarrage) ; depuis, poids et prédictions copiés sur le PC à chaque étape.
- Panne des nœuds 6 à 10 vers 5 h 15 (h200f, g, h) : perdu un PixelSeal à 15 000 pas et trois affinages, tous repris au retour des nœuds depuis leurs points
  de contrôle. Sur h200 et h200b l'environnement Python est `~/eac/t1/.venv` ; ailleurs `/home/hermes/eac/kit/.venv`.
- Deux lancements tombés par manque de mémoire graphique (cartes partagées avec T2 / T3) : la chaîne `c3.sh` relance désormais le pré-entraînement.
- Tous les entraînements t1-reel arrêtés à 10 h 50, cartes rendues.

## 8. Pour reprendre
`bash c2.sh <filigrane> <réserves> <pas> <tag> "--s1 1 --bs 16 --aux 1.0 --seed <g> [--hp 2 | --st 1 --hp 2]" ""` : pré-entraînement `g_<tag>` sur fabriqué
seul, affinage 5 plis `f_<tag>`, tableau `logs/agg_f_<tag>_<filigrane>.txt` et `runs/f_<tag>/<filigrane>/pred_{train,test}_<filigrane>.npz`.
Moyennes : `python ens.py <filigrane> <dossiers…> --out <dossier>` ; par population : `python ens2.py <filigrane> --pleines <…> --partielles <…> --out <…>` ;
comparaison appariée : `python cmp.py <filigrane> <nouveau> <base>`. Prédictions test depuis les poids : `python predict.py --method <m> --tags <a>[+<b>] --tta 2`.
