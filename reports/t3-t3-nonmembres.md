# T3 · session `t3-nonmembres` : estimer le gold avec des images « non-membres » fabriquées

Décision débloquée : quel niveau de loss viser sur le forget set, peut-on tirer des images générées une cible par image, et a-t-on le droit de s'en servir à l'entraînement.
État au 21/11, 22 h 45. Code et mesures : `E:\EAC-lab\t3\t3-nonmembres\` (PC), `~/eac/t3/t3-nonmembres/` (H200 `h200` et `h200b`). Critique séparée : `reports/critique/t3-nonmembres-tour1.md`.

## Vocabulaire
- **Membre / non-membre** : une image est membre si le modèle l'a vue pendant son entraînement. Les 3 500 images du forget set sont membres d'`orig` et non-membres du gold.
- **Loss** : entropie croisée à température 1 entre la sortie du modèle et la classe de l'image ; m et s sont sa moyenne et son écart-type sur un groupe d'images (ce que note le score de loss).
- **Image-vers-image, force f** : le générateur ajoute à l'image source une part f du bruit de diffusion, puis la redessine. f = 0 rend l'image, f = 1 une image entièrement nouvelle.
- **IP-Adapter** : module qui guide la génération par le contenu d'une image de référence sans en reprendre les pixels.
- **Texte-vers-image** : génération à partir d'une légende écrite automatiquement (Florence-2) pour l'image source.
- **AUC** : probabilité qu'un signal classe une image « ratée par le gold » au-dessus d'une image « réussie » (0,5 = hasard, 1 = parfait).

## 1. Ce qui a été fabriqué
Poids téléchargés sans jeton, licences vérifiées sur la fiche Hugging Face : SDXL base 1.0 (OpenRAIL++-M), VAE fp16 de madebyollin (MIT), IP-Adapter SDXL (Apache-2.0), FLUX.1-schnell en GGUF Q8 de city96 avec encodeurs et VAE de `ostris/OpenFLUX.1` (Apache-2.0 ; le dépôt officiel de FLUX.1-schnell exige un compte), Florence-2-large (MIT).
900 images sources (20 par classe, 45 classes), générées en 768 px puis réduites à 224 px. Neuf jeux : SDXL image-vers-image aux forces 0,30 / 0,50 / 0,70 / 0,85, FLUX image-vers-image 0,60, SDXL + IP-Adapter (échelle 0,6), texte-vers-image SDXL et FLUX. Fichiers `img/<type>.npz` (`images`, `src` = indice dans `forget_images.npz`, `y` = classe).
Ce que montrent les planches (`meas/planche_*.jpg`, regardées une à une) : à force ≤ 0,50 l'image est la même, seules les textures changent ; à 0,70 la composition reste et l'objet est redessiné ; IP-Adapter rend une autre prise de vue du même objet dans le même éclairage ; le texte-vers-image rend un objet du même nom dans un style plus net et plus clair que le jeu, et change parfois de classe (une lanterne devient des théières).
Classes absentes du forget set : l'inversion du modèle (optimiser une image pour la classe k) ne donne que des textures illisibles, sur lesquelles `orig` est sûr à 100 % : écartée. En revanche les images les plus activantes parmi 56 000 images Stable Diffusion 1.4 nomment une trentaine des 55 classes absentes (théières 9-11, champignons 13/14/16/17, grange 24, phares 26/27, fauteuil à bascule 34, pains 55/56, bretzel 58, montgolfière 61, horloges 64/74, violoncelle 75…) : `meas/topact_*.jpg`. Aucune image de ces classes n'a été générée (calcul réservé à la calibration).

## 2. Mesures d'`orig` sur ces images (mode évaluation, fp16, repondéré par les effectifs de classe du forget set)
| Images | Précision | m | s | Part loss < 0,01 | Part loss > 5 | Copies modifiées : précision, m / s |
|---|---|---|---|---|---|---|
| forget set (membres) | 1,000 (par construction) | 0,0003 | 0,002 | 1,00 | 0 | 0,99 ; 0,03 / 0,2 |
| SDXL force 0,30 | 0,996 | 0,01 | 0,07 | 0,96 | 0 | 0,98 ; 0,06 / 0,48 |
| SDXL force 0,50 | 0,967 | 0,12 | 0,67 | 0,84 | 0,01 | 0,94 ; 0,22 / 0,97 |
| SDXL force 0,70 | 0,785 | 0,96 | 2,30 | 0,53 | 0,07 | 0,76 ; 1,18 / 2,59 |
| IP-Adapter | 0,604 | 2,13 | 3,49 | 0,33 | 0,18 | 0,59 ; 2,21 / 3,51 |
| SDXL force 0,85 | 0,548 | 2,87 | 4,28 | 0,28 | 0,24 | 0,53 ; 3,04 / 4,35 |
| texte-vers-image SDXL / FLUX | 0,35 / 0,33 | 5,4 / 5,1 | 5,6 / 5,0 | 0,17 | 0,45 | – |
La précision se lit par rapport à la classe de l'image source, elle-même la prédiction d'`orig` (aucune étiquette n'est fournie). Les copies modifiées sont les quatre familles du simulateur (recadrage + symétrie, couleur douce, flou doux, bruit + recadrage) ; la couleur est la plus coûteuse (force 0,70 : 0,70 de précision, 1,48 / 2,86).
Forme de la distribution : elle est bimodale. `orig` reste très sûr quand il se trompe (confiance 0,88 à 0,92), la loss des erreurs vaut 4,3 à 6,2 en moyenne, avec une queue au-delà de 10 pour 2 à 9 % des images.
Erreurs : 68 % vont vers une autre des 45 classes du forget set (lanternes 5-8, bateaux 29-33, plantes 19/21) ; la mauvaise classe n'est le 2e choix d'`orig` sur l'image source que dans 30 à 42 % des cas. Matrices de confusion complètes dans `gold_estime.json`.

## 3. Laquelle de ces lignes ressemble au gold ?
Trois recoupements, dont deux indépendants des générateurs.
**a. Les scores réels du leaderboard.** Pour chaque ligne prise comme gold, score de loss officiel de sept fichiers notés, puis reste = score réel ÷ score de loss (= prédiction × rétention) :
| Gold supposé | 1180 | 1197 | `PG_T85` | 1265 | `PG_T70` | `f20_l48c` | `f20_l814c` |
|---|---|---|---|---|---|---|---|
| départ 0,55 / 1,17 | 0,03 | 0,03 | 0,04 | 0,04 | 0,05 | 0,39 | score de loss nul : réfuté |
| force 0,70 | 0,06 | 0,06 | 0,08 | 0,07 | 0,10 | 0,08 | 0,35 |
| IP-Adapter | 0,09 | 0,09 | 0,13 | 0,12 | 0,16 | 0,13 | 0,17 |
| force 0,85 | 0,11 | 0,11 | 0,16 | 0,14 | 0,19 | 0,16 | 0,19 |
Seules les deux dernières lignes donnent un reste régulier. Les autres exigent que `f20_l814c` (mêmes images basculées que `f20_l48c`, erreurs plus tranchées) ait un score de prédiction 3 à 4 fois meilleur, ce qui n'a pas de mécanisme. Ce recoupement, fait à 21 h 40 sur des fichiers à 20 % de bascules au plus, écarte les golds trop bas mais ne borne pas le gold par le haut : les scores de 22 h le font (§4).
**b. La sonde température de `t3-sondes`** (`TS_tau4` = 0,0057) : elle impose une courbe entre G (moyenne de Σ p_gold·logit d'`orig`) et D × KL. Mes lignes donnent force 0,70 : G 18,6, D × KL 5 ; IP-Adapter : 15,6 et 16 ; texte-vers-image : 9,2 et 58. Elles croisent la courbe entre la force 0,70 et IP-Adapter (G ≈ 17).
**c. La calibration dans le simulateur** (ViT-B entraîné de zéro sur 40 000 images Stable Diffusion 1.4, vrais golds). Références : `orig` simulé a 0,992 sur ses membres, 0,701 sur de vrais non-membres (1,52 / 2,82) ; le gold a 0,68 à 0,70 sur le forget set (1,34 à 1,36 / 2,3 à 2,5). L'hypothèse de base « `orig` sur vrais non-membres ≈ gold sur forget set » tient donc à 15 % près.
| Mêmes générateurs, 360 images | variations des membres | variations de vrais non-membres |
|---|---|---|
| SDXL force 0,70 | 0,883 ; 0,33 / 0,89 | 0,608 ; 1,83 / 2,98 |
| IP-Adapter | 0,747 ; 1,05 / 2,02 | 0,619 ; 2,19 / 3,41 |
| SDXL force 0,85 | 0,775 ; 0,76 / 1,54 | 0,597 ; 2,18 / 3,44 |
À la force 0,70 l'écart membre / non-membre reste presque entier (0,275 contre 0,291 sans variation) : cette force n'efface pas la mémorisation, elle ajoute seulement la dégradation propre au générateur (−0,09 de précision). La ligne « force 0,70 » est donc une borne optimiste pour le gold.
Avec IP-Adapter il reste 44 % de l'écart, compensé en partie par la dégradation du générateur (−0,08) : la ligne « IP-Adapter sur membres » (0,747 ; 1,05 / 2,02) est la plus proche de la vérité (0,68 à 0,70 ; 1,34 à 1,52 / 2,3 à 2,8), mais reste trop basse de 25 à 45 % sur m et de 15 à 40 % sur s. Prise comme élève, elle obtient un score de loss de 0,81 dans le simulateur.
Transposé au vrai jeu, IP-Adapter réel (2,13 / 3,49) × ces facteurs donnerait m 2,7 à 3,1 et s 4,0 à 4,9 ; les scores réels montrent que c'est encore trop bas (§4).
Limite de cette transposition : dans le simulateur la force 0,85 efface moins que l'IP-Adapter (0,76 / 1,54 contre 1,05 / 2,02), alors que sur le vrai jeu c'est l'inverse (2,87 / 4,28 contre 2,13 / 3,49). Les facteurs ne sont donc pas transférables d'un générateur à l'autre ni d'un jeu à l'autre. Ce que la calibration établit : toutes les lignes « variation de membres » sous-estiment la loss du gold (il reste 26 à 94 % de l'écart membre / non-membre) ; l'ampleur vient du recoupement 3a.

## 4. Estimation livrée (`gold_estime.json`) et son bilan face aux scores réels
| | Précision | m | s | Statut au 21/11, 22 h 40 |
|---|---|---|---|---|
| Hypothèse de départ de l'équipe | 0,85 | 0,55 | 1,17 | réfutée (3a) |
| Force 0,70 | 0,79 | 0,96 | 2,30 | mesuré ; trop optimiste (3a, 3c) |
| IP-Adapter, force 0,85 | 0,55 à 0,60 | 2,1 à 2,9 | 3,5 à 4,3 | mesuré ; **trop optimiste aussi** : le score réel monte encore de 35 % à 50 % d'images basculées (0,2179 → 0,2779, m 2,89 → 3,70 à s constant) |
| **Fourchette retenue** | **0,33 à 0,50** | **3,7 à 5,4** | **4,5 à 5,6** | bas de précision = lignes texte-vers-image (mesuré) ; haut = ce qu'impose le score de `f50_l814c` |
| Copies modifiées | 0,02 à 0,05 de précision en moins | +0,1 à +0,4 | +0,0 à +0,3 | mesuré sur toutes les lignes ; la couleur coûte le plus |
Forme, commune à toutes les lignes : distribution bimodale, des images quasi sûres et des erreurs très tranchées (loss des erreurs 5 à 8 en moyenne, queue au-delà de 10 pour 5 à 20 % des images).
**Bilan des prédictions écrites avant résultat (issue #5).** Deux sur trois sont fausses :
- 21 h 53, `f30_l1218c` : prédit 0,12 à 0,16, observé 0,1998 ;
- 22 h 10, `f35_l814c` : prédit ≈ 0,21, observé 0,2179 ;
- 22 h 14, `f40` puis `f50_l814c` : prédit un plateau (0,19 à 0,22), observé 0,2779 à 50 %.
Cause : j'ai traité les lignes IP-Adapter et force 0,85 comme la valeur centrale alors que la calibration (3c) disait seulement qu'elles sous-estiment l'oubli, sans donner de combien. Ce que la piste a établi de solide : le sens (un gold bien plus mauvais que 0,85 de précision, annoncé à 20 h 22 quand le meilleur score de l'équipe était 0,03), l'ordre de grandeur de s (4 à 5), et la forme bimodale. Ce qu'elle n'a pas su donner : m, c'est-à-dire la part d'images que le gold rate. Le leaderboard reste l'instrument de mesure pour cela.
Réserve de fond : la sonde `G10_PG_T70` impose une précision de validation d'`orig` ≥ 0,78, alors que le gold rate au moins la moitié du forget set. Le forget set n'est donc pas un échantillon ordinaire des données d'entraînement : soit il regroupe des images atypiques de leurs classes, soit son retrait appauvrit fortement ces classes. Mes images texte-vers-image (objet de même description, autre rendu) sont dans ce cas pour `orig` (0,33), ce qui explique qu'elles soient la ligne la plus proche.

## 5. Deuxième usage : cibles et ancrage (mesuré à part, dans le simulateur, formule officielle, sans entraînement)
- **Cible par classe : aucune valeur.** Élève = moyenne exacte des golds par image : oubli 0,79. Élève = un autre gold seul : 0,59. Élève = moyenne par classe des mêmes golds : 0,00. Élève = sortie d'`orig` sur un vrai non-membre tiré dans la classe : score de loss 0,94, prédiction 0.
- **Quelles images** : deux golds classent les images par difficulté presque de la même façon (corrélation de rang 0,85). Un signal gratuit tiré d'`orig` seul, l'entropie de sa sortie à température 2 sur l'image, retrouve les images que le gold rate avec une AUC de 0,82 (67 % de bonnes dans le premier cinquième, 33 % au hasard) ; la loss d'`orig` sur une variation générée : 0,78 ; l'augmentation de couleur forte : 0,58. Sur le vrai `orig`, plus saturé, ce signal gratuit ne recoupe que faiblement mes variations (AUC 0,54 à 0,58) et moyennement les images basculées par `f20_l814c` (0,69) : il reste à départager par un envoi.
- **Vers quelle classe** : c'est le facteur dominant (images du gold + 2e choix d'`orig` : oubli 0,04 ; images du gold + classe du gold : 0,27), et les variations ne la donnent pas mieux que le 2e choix d'`orig` (42 % contre 34 %, sur 31 images).
- **Ancrage / rétention** : `code/retproxy.py` applique la formule de rétention sur mes images (classes du forget set seulement). `f20_l814c` : 0,61 sur variations proches, 0,89 sur texte-vers-image ; `PD_f30` : 0,65 et 0,95 ; `PG_T70` : 0,56 et 0,83 ; adoucissement global `TS_tau4` : 0,36 à 0,39. Les candidats abîment donc nettement les images proches du forget set qu'`orig` n'a jamais vues, ce que la réserve de `t3-methodes` (des membres) ne peut pas montrer.
- **Porte des 70 %** (`meas/retproxy_porte.json`) : rapport de précision candidat ÷ `orig` sur ces non-membres, variations proches / texte-vers-image : fichier à 20 % de bascules 0,86 / 0,98 ; 30 % : 0,77 / 0,93 ; 40 % : 0,72 / 0,94 ; 50 % : 0,64 / 0,92 (porte passée, 0,2779) ; 60 % : 0,56 / 0,89. Avec 45 % d'images de validation dans les classes du forget set, le fichier à 60 % ne passe à coup sûr que si la précision de validation d'`orig` est ≥ 0,875 ; le fait connu est ≥ 0,78 à 0,84.
- Classes absentes : 620 images SDXL de 31 classes nommées (`img/abs_t2i.npz`) ; `orig` retrouve la classe visée pour 11 d'entre elles (théières 9, 10, 11 ; amanite 13 ; plante en pot 20 ; phare rayé 26 ; fauteuil à bascule 34 ; bretzel 58 ; montgolfière 61 ; violoncelle 75 ; vitrail 96), précision moyenne 0,31. Sur ces images les fichiers à 20–60 % de bascules gardent toute leur précision (rapport 1,00) et la KL passe de 0,11 à 0,28 (`meas/retproxy_absentes.json`). Limite : `orig` étant peu précis sur ces images, l'instrument sous-estime les dégâts d'une classe détruite (rapport `G10_PG_T70` / `PG_T70` : 0,70 mesuré ici, 0,64 au leaderboard).
Conclusion : rien de mesuré ne justifie d'entraîner sur ces images. Elles servent d'instrument : rétention et porte sur des non-membres, que les autres sessions n'ont pas.

## 6. Zone grise de l'énoncé
« The unlearning method has to work from the original model and the forget set alone. » Des images dérivées du forget set par un modèle génératif public font entrer dans la méthode un modèle appris sur d'autres données. Usage fait ici : mesure et réglage uniquement, aucun candidat entraîné dessus. Question à poser aux organisateurs si l'équipe veut aller plus loin :
> "For the Unlearning task: may the unlearning procedure use (a) public pretrained generative models (e.g. Stable Diffusion) to synthesise variations of the forget images, used as extra unlabeled inputs on which the unlearned model is kept close to the original one; (b) such synthetic images only offline, to choose hyper-parameters, never in the weight update; (c) no external pretrained model at all? We currently assume (b) is allowed and (a) is not."

## 7. Ce qui tourne encore, et limites
- Calibration : classes ImageNet et Stable Diffusion 1.4 ; les facteurs de correction (×1,28 à ×1,45 sur m, ×1,15 à ×1,40 sur s) ne sont pas garantis sur le vrai jeu. Le résultat de 21 h 53 est compatible avec les facteurs hauts.
- Simulateur : classes ImageNet grossières et `orig` simulé moins saturé que le vrai (confiance T=2 de 0,87 contre 0,985) ; les recettes de bascule construites sur ses logits y donnent un score de prédiction nul alors que la vraie tâche les paie (0,1487) : les conclusions du §5 sur le choix des images et des classes sont à confirmer par un envoi.
- 900 images (20 par classe) : intervalle à 95 % de ±0,03 sur une précision, ±0,25 sur m.
- Jülich : mon job à 4 GPU a été annulé à 20 h 37 (règle « un seul GPU ») ; la suite a tourné sur h200b, partagée (4,5 s par image pour IP-Adapter au lieu de 2,2).

## Fichiers
`gold_estime.json` (par type : précision, m, s, quantiles, copies modifiées, précision par classe, matrice de confusion, distribution moyenne à T=2 par classe), `meas/stats_ponderees.json`, `meas/retproxy.json`, `meas/logits_<type>.npz`, `img/<type>.npz`, planches `meas/planche_*.jpg`, scripts `code/` (`gen_sdxl.py`, `gen_flux.py`, `cap.py`, `evalnm.py`, `confront.py`, `sens.py`, `sens2.py`, `selsim.py`, `clssim.py`, `simscore.py`, `retproxy.py`).
