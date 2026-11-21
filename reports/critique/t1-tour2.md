# T1 localisation de filigrane : critique adverse, tour 2

Sam. 21/11, ~17 h 30. Lu : tour 1, sujet, kit (watermarks.py, data.py, utils.py), exp2.py, seg.py, seg2.py, seg2.sbatch, note « labos », planche du train (regardée).
Décision débloquée : les 3 prochaines expériences et la première soumission de ce soir. Aucun job lancé.
[MESURÉ] = vos chiffres. [DÉDUIT] = calcul tiré de ces chiffres ou lecture du code. [À VÉRIFIER] = hypothèse, avec le test qui tranche.
Mots (une fois) : **résidu** = ce que l'embedder ajoute à l'image ; **cadre natif** = la taille à laquelle l'embedder calcule son résidu (PixelSeal 256,
TrustMark 256, StegaStamp 400) avant de l'agrandir à la taille de l'image ; **niveau** = 1/255, le plus petit pas d'un pixel 8 bits ; **bf16** = format
de calcul rapide sur 16 bits qui ne garde qu'environ 3 chiffres significatifs ; **AUC** = chance qu'une case filigranée ait un score plus haut qu'une case propre.

## 1. Diagnostic des résultats

**1.1 PixelSeal n'apprend rien, même sur synthétique : suspect n° 1 = le bf16 sur l'entrée.** [DÉDUIT, à vérifier par E1]
- seg.py:122 (et seg2.py:176, et l'évaluation l.103/154) : tout le réseau tourne sous `autocast(bfloat16)`, y compris la première convolution,
  dont l'entrée est l'image normalisée (valeurs entre −2,1 et +2,6). Un niveau vaut 0,0039/0,226 = 0,017 après normalisation. En bf16, l'écart entre
  deux nombres voisins est 0,0078 entre 1 et 2, et 0,0156 entre 2 et 4. Un résidu de ±1 niveau tombe donc sur 1 à 2 pas d'arrondi : l'erreur
  d'arrondi est aussi grande que le signal.
- L'ordre des résultats colle exactement à l'amplitude des résidus : PixelSeal (le plus discret, atténué par la JND, c'est-à-dire le masque de
  visibilité qui met ~0 dans les zones lisses : souvent < 1 niveau) ne passe pas ; TrustMark (~2 niveaux, PSNR ~43 dB) passe en partie ; StegaStamp
  (plusieurs niveaux) passe bien. Perte bloquée à 1,28 ≈ BCE 0,69 + Dice ~0,6 : le réseau prédit une constante, il ne voit rien.
- Ce n'est probablement PAS le stem à pas 2 : sur une image filigranée en entier, le résidu PixelSeal est calculé à 256 puis agrandi ×2, donc il est
  basse fréquence à 512. Une convolution 7x7 de pas 2 le voit très bien. Ce n'est pas non plus le stockage uint8 de X.npy : `wm.embed` arrondit
  déjà en uint8, et les PNG réels aussi. Synthétique et réel sont arrondis pareil.
- Correctif : calculer le stem en fp32 (sur A100, le fp32 des convolutions passe par TF32 : 10 bits, soit ~17 pas par niveau, ce qui suffit), ou
  mieux, donner au réseau un résidu déjà petit (`x − flou(x)`, multiplié par ~20) : là, le bf16 garde sa précision relative.

**1.2 Revue ligne à ligne de seg.py (le générateur est aussi utilisé par seg2.py)**
- Polarité des étiquettes (l.36, 78-80) : JUSTE. `holes = ~mk` = zone propre ; on y recolle `base` ; `mask = ~hole` = zone filigranée.
- `soft_paste` (l.59-63) : sens JUSTE dans les deux cas (plein : on recolle l'original dans le trou ; forme : on colle la version filigranée
  dans la forme). L'adoucissement ne touche que l'image : l'étiquette reste nette. Acceptable.
- Réduction du masque (l.86) : fonction officielle, JUSTE. Flips d'entraînement et TTA (l.107, 120-121) : axes cohérents pour l'image et le masque.
- `jitter` + `np.roll` (l.44) : le décalage fait le tour de l'image. Une forme qui touche un bord ressort au bord opposé, et un rectangle partiel
  peut être coupé en 2 à 4 morceaux. Ça abîme surtout le cas 3. Défaut mineur : décaler en remplissant de 0 plutôt qu'en faisant le tour.
- **Cas 3 (l.84-85), défaut réel** : les masques « partiels » réels comprennent les rectangles d'outpainting (extension d'une image vers une plus
  grande). Dans le vrai pipeline, le rectangle est une image source ENTIÈRE filigranée dans son propre cadre (échelle du motif = taille du
  rectangle/256). Ici on filigrane les 512 px puis on découpe la forme : le motif est au cadre 512. Pour les formes rectangulaires, utiliser le
  chemin du cas 2. Pour les objets transplantés, le cadre réel est inconnu (objet déplacé, peut-être redimensionné) : tirer une échelle au hasard.
- **Lois de taille** : `rand_rect` tire 1 à 35 % de l'image ; le réel partiel fait 1 à 18 % [MESURÉ]. La moitié des rectangles synthétiques sont
  plus gros que tout ce qu'on voit au test. Tirer les tailles dans la loi des boîtes réelles.
- Base COCO (l.69) : 5 000 photos JPEG pour 6 000 tirages (1 000 réutilisées), bruit de capteur et blocs JPEG. Le réel, ce sont des images
  générées lisses (voir la planche : peau, ciel, assiettes), donc un autre niveau de bruit et un seuil de sortie décalé (voir §3).
- Sélection (l.129) : meilleur epoch choisi sur « tous » (170). Pas de réel dans l'entraînement en v1, donc c'est presque honnête : seules les formes
  fuient. Mais 12 epochs × 1 choix ≈ +0,01 à +0,02 d'optimisme.

**1.3 exp2 (filtre adapté) n'est PAS prouvé mort : il y a deux défauts dans le code.** [DÉDUIT]
- exp2.py:27 : le gabarit (le motif attendu pour notre message) vient de `wm.embed`, qui arrondit en uint8. La note « labos » demandait
  explicitement le gabarit SANS quantification. Pour PixelSeal, la plupart des pixels du gabarit valent alors 0 (résidu < 0,5 niveau) : le filtre
  corrèle surtout du bruit d'arrondi. D'où des gains médians ≈ 0, dedans comme dehors.
- exp2.py:30-35 : `h = x − flou(σ = 1,5)` à 512 ne garde que les très hautes fréquences, alors que le résidu PS/TM d'une image pleine est le résidu
  256 agrandi ×2, donc surtout en dessous de 0,25 cycle/px. Le filtre regarde la bande où le signal n'est pas. C'est l'erreur inverse de celle de
  la question 4 du brief : des filtres SRM (filtres passe-haut fixes de la stéganalyse) à 512 tueraient aussi ce signal. Il faut passer au cadre natif d'abord.
- Coût du test refait (E5) : 3 min de GPU. Utile seulement comme canal d'entrée : ne pas y mettre de temps humain avant E1-E3.

**1.4 StegaStamp sur-apprend et oscille.** Trois causes empilées. (a) Bruit : 34 images en réserve donnent ±0,034 d'erreur type (0,2/√34). Entre
0,63 et 0,60, l'écart n'est pas significatif ; 0,63 → 0,54 sur les 170 images, en revanche, est réel (±0,015). (b) Calibration [À VÉRIFIER par E3] :
quand la perte synthétique tombe à 0,12, les logits grossissent et se décalent ; sur le réel, le seuil fixe 0 fait alors basculer des zones
entières d'un epoch à l'autre. Si l'AUC réelle est stable pendant que le mIoU oscille, c'est un problème de seuil, pas de représentation.
(c) Domaine : COCO bruité contre images générées lisses, rectangles trop gros, cas 3 au mauvais cadre. Le réseau apprend un détecteur qui marche très
bien sur notre synthétique et moins bien ailleurs.

**1.5 TrustMark plafonne à 0,585 alors que la perte baisse encore** : la perte synthétique continue de descendre mais le réel ne suit plus, c'est
l'écart de domaine. S'y ajoute la dégradation bf16 du §1.1, partielle pour TM. Le OneCycle (taux d'apprentissage qui monte puis redescend) finit à
12 epochs ; plus d'epochs sans corriger le domaine ne ferait qu'élargir l'écart, comme pour SS.

## 2. seg v2 est-il bien conçu ? Non, il faut le corriger avant de lire ses chiffres

1. **seg2.py:183, BUG de méthode** : le meilleur checkpoint est choisi sur « tous », qui contient les 136 images réelles apprises (l.163-174). On
   choisira l'epoch qui a le mieux mémorisé, et le chiffre rapporté sera gonflé. Choisir sur « reserve » (ou à epoch fixe) et ne rapporter que la réserve.
2. **Mémorisation** : 6 images réelles tirées avec remise par lot de 30, 250 lots par epoch, 14 epochs → chaque image réelle est vue ~150 fois,
   sans aucune augmentation. Il faut des retournements et des décalages de 4 px sur le réel pour le U-Net aveugle (pas pour la version qui reçoit
   le message : un retournement change le motif), 2 à 3 images réelles par lot, et des plis (k-fold).
3. **Comparaison v1/v2 faussée** : v1 évalue avec TTA (moyenne avec l'image retournée, seg.py:107), v2 sans (seg2.py:157). Et v2 change trois choses à la
   fois : tronc, mélange réel, TTA. Seul le duo TrustMark dec/unet isole quelque chose (l'architecture). Remettre la TTA dans le bras unet.
4. **bf16 toujours là**, y compris dans le décodeur PixelSeal réutilisé (seg2.py:123, `det.float()` puis autocast) : le tronc dédié au motif reçoit
   la même entrée abîmée qu'en v1. Si E1 confirme, le bras PixelSeal v2 est mort d'avance.
5. **À vérifier avant de lancer (crash ou canal faux)** : (a) seg2.py:128 et l.148 : `output_upscaling` peut changer la résolution, et `torch.cat`
   plante si `pp` n'est pas en 8x8. (b) seg2.py:126-127 refait `convnext` à la main et peut sauter une normalisation finale. Contrôle en 1 ligne
   sur CPU : `pp.mean((2,3))` doit égaler `wm.logits(x)` (le décodeur fait la moyenne puis la couche linéaire). Même contrôle pour TrustMark.
   (c) `trunk.train()` réactive le drop-path (blocs sautés au hasard) du ConvNeXt s'il est configuré : passer le tronc en `eval()` et ne laisser que les gradients.
6. **Entrée 256** : pour les images pleines, 256 est le cadre natif, donc aucune perte (c'est même mieux que 512). La carte la plus fine (64x64, soit
   8 px à 512) plafonne autour de 0,95 [MESURÉ : oracle 8 px], ce qui ne gêne pas. Le vrai défaut : les petits rectangles partiels (50-100 px)
   ne font plus que 25-50 px. Garder une branche à 512.
7. **Canal d'accord en 8x8 seulement** : chaque position voit presque toute l'image, donc la carte fuit (tour 1, §1.6). Sans danger, mais peu utile.
8. BN gelée (statistiques de normalisation figées) sur le ResNet-50 TrustMark : bon choix avec des lots de 30.
9. Seuil fixé à 0 : à régler par plis (E3). Coût caché : chaque `--tag` régénère ses 6 000 images, et v1 n'a pas sauvé les bits, donc impossible
   de réutiliser X.npy (seg2.py:91 attend B.npy). Les deux bras TrustMark génèrent la même chose deux fois.

## 3. L'écart synthétique → réel

**Ce qui diffère** [DÉDUIT de la planche et du code] : (1) les bases : photos COCO bruitées en JPEG contre images générées lisses, avec de grands
aplats où la JND de PixelSeal n'écrit presque rien ; (2) la loi des tailles partielles (1-35 % contre 1-18 %) ; (3) le cadre du motif sur les
rectangles d'outpainting (§1.2) ; (4) la chaîne réelle peut contenir un redimensionnement après filigranage (image générée en 1024, filigranée, puis
réduite à 512) : on ne le sait pas, mais E4 le dit en 10 min ; (5) les bords : un trou réel = pixels d'origine exacts (la planche montre des blobs nets,
sans contour visible), ce que notre simulateur fait déjà.

**La correction la moins chère, gratuite et du bon domaine : les images des AUTRES filigranes servent de bases propres.** Une image PixelSeal est
une image propre pour TrustMark et StegaStamp. On a 340 images réelles du train par méthode cible (510 − 170), plus 680 du test si le règlement
autorise d'utiliser leurs pixels sans étiquette [À VÉRIFIER dans le règlement] : de vraies images FLUX/SD à 512, de la bonne texture, sans aucune
génération. Le motif d'un autre filigrane est décorrélé du nôtre, et le test ne mélange jamais deux méthodes. Si ça gêne, on peut le retirer
grossièrement. En plus : les zones propres des images partielles de la même méthode (82-99 % de l'image, à rogner hors masque dilaté de 16 px).
Générer des images FLUX nous-mêmes : non, pas avant d'avoir mesuré ce gain. Recopier les « pixels d'origine » des images pleines réelles :
impossible (on ne les a pas), mais inutile, puisqu'on les fabrique sur les nouvelles bases.

**Mesurer l'écart** : 5 plis sur les 170 réelles (4 plis en mélange, le 5e en lecture, rotation). Rapporter par population (pleines / partielles),
l'AUC (sans seuil) ET le mIoU au seuil réglé sur les 4 autres plis, plus une courbe « score synthétique de validation contre score réel » par
epoch. Écart = mIoU(synthétique seul) − mIoU(synthétique + réel), mesuré sur le même pli.

## 4. Une architecture faite pour un résidu de ±1 à 3 niveaux

- **Travailler au cadre natif** : pour une image pleine, le résidu est le résidu natif agrandi. Réduire l'entrée à 256 (PS, TM) ou 400 (SS) le
  concentre sans rien perdre, et rend utiles les filtres passe-haut (SRM, ou `x − flou σ = 1-2`). Pour les rectangles partiels, l'échelle vaut
  taille/256 et varie : il faut deux entrées, l'une au cadre natif, l'autre à 512, ou une pyramide.
- **Stem de stéganalyse** : 2 à 3 couches 3x3 sans pas ni pooling (comme SRNet, le réseau de référence de Boroumand 2019), en fp32, sur
  [image, résidu passe-haut ×20, carte d'énergie du gabarit si E5 survit], puis l'encodeur ResNet-34 ImageNet et le décodeur U-Net jusqu'à 128.
  Coût : identique à v1 à 256.
- PixelSeal : réduire à 256 est l'option la plus prometteuse, mais seulement après E1 (sinon on ne sait pas si c'est le bf16 ou l'échelle).
  StegaStamp : garder le U-Net v1 (il marche) et ne corriger que les données. TrustMark : U-Net v1 + données corrigées d'abord ; le tronc du
  décodeur seulement si le contrôle §2.5 passe.
- Perte : ajouter le Lovász (approximation dérivable de l'IoU, tour 1 §3.1) une fois les données réglées ; pas avant.

## 5. Règle de décision pour la PREMIÈRE soumission (ce soir, sans attendre v2)

Pourquoi tout de suite : on n'est classé qu'avec les 3 tâches soumises, et en min-max (remise à l'échelle entre la pire et la meilleure équipe),
être premier en T1 vaut 1,0. Le classement garde notre meilleur score : une soumission ratée ne coûte rien (vérifier s'il y a un plafond de soumissions par jour).
- **StegaStamp** : seg v1, epoch 6 (0,63 sur les 170 ; réserve 0,67) [MESURÉ], seuil 0, TTA retournement.
- **TrustMark** : seg v1, epoch 7 (0,585 ; réserve 0,56). Ne le remplacer par v2 que si v2 gagne ≥ 0,04 sur la réserve (2 erreurs types) ET que
  l'écart « 136 apprises − réserve » reste < 0,10 (sinon c'est de la mémorisation).
- **PixelSeal** : occlusion 64 px, avec le seuil global appris sur le train (0,51 ; un seul seuil sur 170 images, donc biais ~0,01), s'il tourne sur
  les 340 images de test en < 30 min. Sinon, plancher sans localisation : décision par population. On calcule le z (écart au hasard, en
  écarts-types) de l'image entière avec le vrai message ; z fort → tout-1, z faible → tout-0. Valeur attendue ≈ 0,5 × 0,47 + 0,5 × 0,40 ≈ 0,43
  [DÉDUIT des fractions], à vérifier sur le train en 2 min.
- Attendu : ~(0,51 + 0,585 + 0,63)/3 ≈ 0,57 [estimation], ±0,011 d'erreur type sur les 306 images publiques.
- **Contrôles obligatoires** : (1) faire passer le script d'écriture sur le TRAIN, puis rescorer le .pt avec `miou_image`. On doit retrouver les
  chiffres locaux à 1e-4 près, ce qui attrape les masques transposés, inversés ou mal ordonnés. (2) Noms = ensemble exact du metadata de test, sans
  doublon. (3) Tenseur `torch.uint8` (ou bool), forme [1020, 128, 128], valeurs {0, 1}. Taille ≈ 16,7 Mo : sous 20 Mo, mais en int64 on
  passerait à 134 Mo et la soumission serait REJETÉE. (4) Fraction prédite moyenne par méthode sur le test ≈ celle du train (~0,5), sinon le seuil a dérivé.

## 6. Les prochaines expériences, classées (les 3 premières sont celles de ce soir)

| # | Expérience | Coût | Chiffre qui change la décision |
|---|---|---|---|
| E1 | **Sauvetage PixelSeal.** (a) Test 2 min : `wm.logits` sur les 170 réelles en fp32 puis sous autocast bf16 avec la normalisation ImageNet, et corr(bf16(x+r) − bf16(x), r) sur 20 paires synthétiques. (b) seg v1 PS relancé en fp32, entrée réduite à 256 + canal `x − flou`, 4 epochs. | 2 min + 15 min, 1 GPU | (a) chute de bit-acc ≥ 0,05 ou corr < 0,7 → bf16 coupable : enlever l'autocast du stem dans TOUS les bras (et v2). (b) perte < 1,0 à l'epoch 2 et mIoU réel ≥ 0,55 → remplace l'occlusion dans la soumission. |
| E2 | **Bases réelles croisées + 5 plis** : TM et SS (et PS si E1 passe), bases = images réelles des deux autres méthodes + COCO, tailles partielles tirées de la loi réelle, cas 3 rectangle au bon cadre, 2-3 réelles augmentées par lot, sélection sur le pli tenu à l'écart. 4 plis en parallèle sur un nœud. | 1 nœud × ~1 h par méthode | mIoU moyen des plis ≥ v1 + 0,03 → nouvelle recette par défaut. < +0,01 → l'écart n'est pas dans les bases : regarder E4. |
| E3 | **Calibration sans entraînement** : sauver les logits réels de chaque epoch (v1 SS/TM existants), tracer l'AUC par epoch et le mIoU au seuil réglé sur 4 plis ; ajouter un seuil par population (pleine si z_full fort). | 20 min, peu de GPU | AUC stable (±0,01) quand le mIoU oscille → c'est le seuil : seuil par plis, +0,02 à +0,05 espéré. AUC qui oscille aussi → arrêt précoce sur la réserve et données E2. |
| E4 | **Fidélité de la chaîne partielle** : bit-acc du vrai message sur le rognage de la vraie boîte des images partielles réelles, contre des rectangles synthétiques de même taille, filigranés (i) à la taille du rectangle, (ii) à ×2 puis réduits. | 10 min | Celle des deux variantes qui reproduit la bit-acc réelle (à ±0,03) devient la recette du cas partiel. Si le réel est ≪ les deux (< 0,7), il y a une autre transformation : tirer une échelle au hasard. |
| E5 | **Filtre adapté refait** : gabarit en float (sans arrondi uint8), calculé et corrélé au cadre natif (réduction à 256/400, passe-haut σ 1-2), 30 images/méthode. | 3 min GPU, 30 min de code | AUC ≥ 0,75 sur les pleines → canal d'entrée du U-Net. < 0,65 → enterré pour de bon. |

**À arrêter** : lire les chiffres de v2 tant que le §2.1 n'est pas corrigé ; l'Otsu par image ; la CAM brute ; les filtres SRM à 512 ; la génération
d'images FLUX ; tout nouvel epoch « pour voir » sans corriger les données (SS montre que ça dégrade le réel).
