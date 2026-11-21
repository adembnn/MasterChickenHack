# Tâche 1 : localiser un filigrane. État de l'art (2e passe) et plan

Date : 21 nov. 2026. Débloque : quelle méthode coder pour la tâche 1 d'ici dim. 22 nov. 12 h.
Légende : [PAPIER] = lu et démontré dans un article ou un code publié. [IDÉE] = notre déduction, jamais mesurée.
Aucun chiffre de mIoU ci-dessous n'est mesuré : je n'ai pas les données sur cette machine. Ce sont des fourchettes à vérifier.

## 0. Ce qui a été lu cette fois, et ce qui change

Lus en entier (texte complet, pas les résumés) : le kit (watermarks.py, utils.py, data.py, submission_template.py), le code de TrustMark
(trustmark.py), le code de PixelSeal (videoseal : pixel_decoder.py, extractor.py), les 57 diapos de Pierre Fernandez (texte seulement,
pas les images), PixelSeal (2512.16874), le banc d'essai des filigranes locaux (2609.16832), MaskMark (2504.12739), ReliaGuard (2611.16368),
TetherMark (2611.22345), MELB (2512.14994), APT (2608.30656), résumés + sections clés de NAWE, RemTraceNet, FissionGuard.
Interrogés : arXiv (partiel : l'API a refusé les dernières requêtes, erreur 429), Semantic Scholar (citations de PixelSeal : 21 articles ;
de WAM : 60 ; de TrustMark : 60), Exa, Parallel, GitHub.
Non trouvé : aucun article de SprintML (organisateurs) sur cette tâche ; aucune soumission ICLR 2027 qui localise des filigranes
post-hoc existants ; aucun papier qui localise PixelSeal, TrustMark ou StegaStamp tels quels. Notre tâche est donc inédite : l'état de
l'art donne des briques, pas la recette. Les diapos de Fernandez disent seulement « la solution de localisation, c'est WAM »
(le filigrane entraîné à segmenter), rien de plus pour nous.

Deux trouvailles qui changent le plan du premier rapport :
1. Les trois filigranes sont posés sur un rectangle découpé, puis recollés [PAPIER partiel + IDÉE]. Le kit le dit pour TrustMark
   (« patches »), et son code encode le patch redimensionné à 256 px puis le recolle avec un bord fondu de quelques pixels. Si PixelSeal
   et StegaStamp sont posés pareil (le wrapper embed() accepte « une image de taille quelconque »), le motif est calé sur le rectangle,
   pas sur l'image de 512. Le décodeur ne lit alors bien que si on lui donne exactement ce rectangle. Cela remplace la « fenêtre
   glissante » floue du premier rapport par une recherche de boîte exacte (méthode A ci-dessous). À vérifier en 10 min (expérience 0).
2. TrustMark embarque déjà un détecteur de boîte (Faster R-CNN) : `TrustMark(loadBBoxDetector=True).localize(image)` rend
   [x1,y1,x2,y2] [PAPIER : code adobe/trustmark]. Les poids sont téléchargeables (82,9 Mo, adresse répondant 200 : cai-watermark.adobe.net).
   Le kit ne le charge pas. Rien ne dit qu'il marche sur nos images retouchées : à tester sur le train.

## 1. Ce qu'on nous demande (5 lignes)

- Entrée : image 512x512 PNG, méthode (PixelSeal 256 bits, TrustMark Q 100 bits, StegaStamp 100 bits), message caché connu (test aussi).
- Sortie : masque 128x128 en 0/1 (1 = filigrané), 1020 masques, fichier .pt, 20 Mo max (bool = environ 17 Mo).
- Le filigrane n'est posé que dans un rectangle ; puis éditions : agrandissement (outpainting), retrait d'objet, ajout d'objet,
  transplantation (objet filigrané collé sur une image propre).
- Score : par image, moyenne de l'IoU (recouvrement) de la classe propre et de la classe filigrane, puis moyenne ; classe vide des deux côtés = 1.
- Calcul : 1 nœud, 4 A100 40 Go par job. 510 images train avec masques (170 par méthode), 1020 test. Classement public = 30 % du test.

## 2. État de l'art récent (ce qui sert vraiment)

| Papier | Date | Où | Idée | Code | Pertinence pour nous |
|---|---|---|---|---|---|
| Watermark Anything (WAM) 2411.07231 | nov. 2024 | ICLR 2025 | extracteur ViT + décodeur de pixels : 1 canal « filigrané ? » + 32 bits par pixel ; entraîné avec masques COCO, collage, retouche ; DBSCAN pour plusieurs messages | github.com/facebookresearch/watermark-anything | Preuve que « segmenter le filigrane » marche. Pas utilisable tel quel (nos filigranes ne sont pas WAM). Modèle à copier. |
| Pixel Seal 2512.16874 | déc. 2025 | arXiv (Meta) | embedder U-Net + JND (masque de visibilité), extracteur ConvNeXt-tiny à 256 px ; filigrane calculé à 256 puis agrandi | github.com/facebookresearch/videoseal | Décodeur global (moyenne sur 8x8 positions puis couche linéaire), donc une carte par position existe gratuitement. Aucune localisation dans l'article. |
| TrustMark 2311.18297 (ICCV 2025) | 2023-25 | ICCV 2025 | encodeur/décodeur toutes tailles ; décodeur ResNet-50 ; détecteur de boîte livré dans le dépôt | github.com/adobe/trustmark | Boîte prête à l'emploi (localize). Bord fondu de 1 % du côté (max 50 px). |
| StegaStamp 1904.05343 | 2019 | CVPR 2020 | résiste à l'impression ; détecteur de segmentation BiSeNet entraîné sur des StegaStamps collés au hasard (code d'entraînement non publié) | github.com/tancik/StegaStamp | Précédent : ils ont déjà appris un segmenteur sur « timbres collés au hasard ». Notre données synthétiques = même recette. |
| Banc local What Breaks Local Watermarks? 2609.16832 | sept. 2026 | arXiv | 55 transformations, 5 méthodes dont TrustMark et PixelSeal ; PixelSeal « sans sortie spatiale », TrustMark « boîte rectangulaire » | markbench (annoncé) | Dit noir sur blanc que l'inpainting/outpainting génératif met le décodage au hasard, même hors de la zone retouchée. Prépare à des fenêtres sans signal. |
| MaskMark 2504.12739 | avr. 2025 | arXiv | masque pendant le décodage + module de localisation U²-Net ; bat WAM en localisation ; 20 h sur 1 A6000 | non vérifié | Argument [PAPIER] : un décodeur n'a jamais vu de filigrane partiel et le contenu propre le perturbe : il faut du masquage ou un réseau de localisation en plus. |
| ReliaGuard 2611.16368 | 12 nov. 2026 | arXiv | lecteur « aveugle » : réponses du filigrane + contexte image + lecture multi-échelle à pleine résolution, entraîné sur éditions complètes et collées ; IoU 85,6 % (sortie complète), 94,9 % (collée) | non vérifié | Le plus proche de notre méthode B : un petit réseau qui lit les cartes du décodeur et sort un masque. Mais filigrane conçu pour. |
| APT 2608.30656 | 31 août 2026 | arXiv | signal fragile de localisation ; distingue édition « collée » (SP) de « régénérée en entier » (FR) | non vérifié | Distinction utile : si l'éditeur régénère toute l'image (FR), le fond propre perd aussi le filigrane ; si collée (SP), il reste. À regarder dans notre train. |
| FUSED 2608.28302 / DiffSeg30k 2511.19111 / TruFor / DiffusionPrint | 2025-26 | arXiv, CVPRW | localisent les zones inpainted par indices de falsification, sans filigrane ; IoU hors-domaine 30-45 % | github.com/AntonNuzhdin/FUSED, grip-unina/TruFor | Canal d'appoint seulement (bords de zone générée). Faible tout seul, et ils notent que restaurer les pixels intacts efface leurs indices. |
| Où est le filigrane ? MELB 2512.14994 | déc. 2025 | WACV 2026 | filigrane DWT par blocs qui sort une carte | non vérifié | Autre filigrane : sans usage direct. Seule leçon : carte par blocs + test statistique par bloc. |
| TetherMark 2611.22345 | 17 nov. 2026 | arXiv | authentifie l'instance contre la transplantation (message valide recollé sur une autre image) | non vérifié | Même menace que notre « transplantation », mais il change le filigrane. Aucune méthode pour nous. |
| Query-efficient scan-and-search 2611.02030, FissionGuard 2610.11625 | oct.-nov. 2026 | arXiv | localisation de filigranes de TEXTE (e-values, FDR) | non | Hors sujet (texte). Retiens seulement l'idée : tester des fenêtres et contrôler les faux positifs. |
| NAWE 2609.25972, RemTraceNet 2609.31694 | sept. 2026 | arXiv | extraction par débruitage ; détection de traces de suppression | non | Hors sujet. |

Ce que ce tableau dit : le « décodeur global qui ne sait pas localiser » n'a pas de solution publiée. Les briques publiées sont
(i) un réseau de segmentation entraîné sur des collages (WAM, MaskMark, ReliaGuard, StegaStamp-détecteur), (ii) un détecteur de boîte
(TrustMark), (iii) des cartes par position du décodeur. On assemble.

## 3. Méthodes classées (meilleur score attendu en haut ; chacune : principe, étapes, coûts, risque)

Vocabulaire une fois pour toutes : « logit » = score brut d'un bit avant le seuil (positif = bit 1). « Marge » = moyenne sur
les bits de (2m-1) x logit, où m est le bit du message connu : positive si le décodeur retrouve notre message, proche de 0 si bruit.
« CAM » (carte d'activation de classe) = on ne moyenne pas les positions avant la couche linéaire, on regarde ce que chaque position vote.

### Expérience 0 (10 à 30 min, avant tout) : dans quel cadre le filigrane est-il calé ?
- Sur 20 images train par méthode, sans retouche visible : (a) marge de `wm.logits(image entière)` ; (b) marge de `wm.logits(rognage du rectangle vrai)`,
  rectangle = boîte englobante de `mask` (512x512), rognage via `to_tensor(img)[:,:,y0:y1,x0:x1]` (le wrapper redimensionne seul ; anisotrope : voulu).
  Bruit pur : 50 % de bits justes, écart-type 3,1 points (256 bits) ou 5 points (100 bits).
- Si (b) >> (a) : cadre « rectangle » (le plus probable) : méthodes A. Si (a) >> (b) pour les images `full_watermark = true` :
  cadre « image entière restaurée » : méthodes D (fenêtres). Probablement un mélange : le tester séparément pour `full_watermark`.
- Même test en décalant le rectangle de 2, 4, 8, 16 px : donne la précision d'alignement à viser pour A.

### A. Recherche de la boîte exacte par le décodeur, puis découpe des trous (IDÉE ; gain attendu le plus fort ; pour les 3 méthodes)
Principe : si le motif est calé sur le rectangle R, la marge du décodeur est maximale quand on lui donne exactement R. Un décodeur qui
retrouve les 256 (ou 100) bits du message connu prouve statistiquement que la zone testée est filigranée (hasard : environ 1 chance sur 10^10).
Étapes (API du kit) :
1. Candidats de boîte : coordonnées multiples de 8 px, ratio max 16:1 (PixelSeal, StegaStamp) ou 2:1 (TrustMark). Pour TrustMark, ajouter
   la boîte de `localize()` ; pour tous, la boîte du masque grossier de la méthode C ou D comme point de départ.
2. Pour chaque boîte : rogner, `torch.nn.functional.interpolate` vers la résolution native (256 / 245 / 400, antialias), empiler par lots de 512,
   `wm.logits(lot)`, calculer la marge avec le message `item["message"]` (0/1 vers -1/+1).
3. Montée par coordonnées : chaque bord se déplace tant que la marge monte (pas 16, 8, 4, puis 2 px) ; 3 à 5 départs. Environ 2 000 à 6 000
   décodages par image.
4. Dans la boîte trouvée : carte des trous. Rerogner R, `wm.features()["per_position_logits"]` (8x8) donne le vote par case (PixelSeal, TrustMark ;
   StegaStamp : logits globaux seulement, donc fenêtres internes alignées sur R). Case dont le vote est négatif = trou (objet ajouté, retiré).
   Affiner à pleine résolution avec la corrélation de l'embedder (méthode B).
5. Masque 512 = boîte moins trous, puis `downsample_mask` du kit pour vérifier, puis 128x128.
Coût : GPU 0,5 à 2 h au total (PixelSeal et TrustMark très rapides ; StegaStamp passe par ONNX à 400 px, plus lent : limiter les candidats). Personne-heures : 6 à 8.
Risque : moyen. (1) Le cadre n'est pas « rectangle » pour toutes les images (voir exp. 0). (2) Les éditions génératives peuvent réduire la
marge sous le seuil de détection (le banc 2609.16832 dit que le décodage peut tomber au hasard) : alors on n'a pas de boîte ; repli sur C.
(3) Transplantation : la zone filigranée est la forme de l'objet, pas un rectangle : le rectangle englobant de l'objet sera trouvé, la forme
vient du réseau (méthode C). Convient à : les trois (c'est le point fort de TrustMark et PixelSeal ; StegaStamp plus bruité).

### B. Corrélation avec l'embedder, au pixel près (IDÉE ; training-free ; 4 à 6 h)
Principe : on possède l'embedder de chacun des trois. Un filigrane connu laisse une trace dans le signal haute fréquence ; on mesure où l'image
« contient déjà » le motif que l'embedder voudrait y mettre pour notre message. Étapes : r_m = embed(image, m) - image (via `wm.embed`),
r_faux = embed(image, m') - image pour 4 à 8 messages au hasard m' ; haute fréquence de l'image (image - flou gaussien) ; carte locale
c(m) = moyenne glissante de (hf x r_m) sur 16 à 32 px, par canal, puis moyenne ; score = c(m) - moyenne(c(m')). Les messages faux sont le
témoin : ils calibrent le niveau de hasard pour cette image, ce qui évite un seuil unique.
GPU : 2 passes d'embedder par message, environ 20 ms ; 1 h pour 1530 images. Risque : élevé (le motif de l'embedder dépend du contenu ; sur une
image déjà filigranée, embed ajoute un second filigrane, d'où l'usage de la corrélation et pas de la différence ; StegaStamp : le résidu est
calculé à 400 px puis agrandi, motif plus basse fréquence). À limiter à 2 h : si, sur 20 images train, la carte ne ressemble pas au masque, on arrête.
Utilité principale : entrée du réseau C. Convient surtout à StegaStamp (résidu fort) et PixelSeal.

### C. Petit réseau de lecture entraîné (preuve partielle dans WAM, MaskMark, ReliaGuard ; recette = IDÉE ; 12 à 16 h)
Principe : un U-Net léger lit : l'image (3 canaux), la CAM du décodeur (rognée et remise à l'échelle de l'image), la carte de boîte de A,
les cartes de B, et la précision de fenêtres alignées sur la boîte. Sortie : masque 128x128. Un réseau par méthode (3 jeux de poids).
Étapes :
1. Cartes d'entrée calculées une fois pour 1530 images et stockées en .npy (1 h GPU).
2. U-Net à encodeur ResNet-18/34 pré-entraîné (segmentation_models_pytorch, pip), perte BCE + Dice, 128x128, validation croisée 5 plis par méthode.
3. Données synthétiques (sinon 170 images par méthode ne suffisent pas) : images propres (source à décider : voir §6), rectangle tiré dans les
   ratios de la tâche, `wm.embed(rognage, message)` puis recollage ; édits simulés : agrandissement (recoller sur un canevas plus grand
   rempli d'une autre image), trous (coller des formes d'une autre image), transplantation (forme d'objet collée sur une image propre).
   20 000 images par méthode : embed 20 à 60 images/s, donc 2 à 6 h CPU/GPU en parallèle sur 4 GPU.
4. Entraîner d'abord sur le synthétique, puis affiner sur le train réel (retouches génératives : bords doux, fond cohérent).
5. Post-traitement : pour TrustMark, rectangle englobant (puis trous) ; tout : retirer les composantes de moins de 4 cellules, remplir les trous
   de moins de 16 cellules. Optimiser seuil et post-traitement directement sur le mIoU officiel du pli de validation.
GPU : génération 4 x 1 h ; entraînement 3 x 1 A100 x 1-2 h. Risque : moyen à élevé (écart synthétique/réel). Gain : plus grand si A échoue
sur les trous fins ; sinon corrige les bords de A. Convient aux trois.

### D. Fenêtre glissante avec le décodeur (reste utile si l'exp. 0 dit « image entière ») (IDÉE ; 3 à 5 h)
Piège de la première version : rogner 128 px puis agrandir à 256 change l'échelle du motif (PixelSeal : motif calé sur une grille 256 ; un rognage
de 128 px du 512 contient 64 px de grille, agrandi à 256 on lit un motif 4 fois trop gros). Correction [IDÉE] : réduire d'abord toute l'image
à la résolution native (256 / 245 / 400, antialias), puis ne garder qu'une fenêtre et mettre le reste à la moyenne de l'image (le masquage de
MaskMark, appliqué à l'entrée du décodeur). Fenêtres 64 et 128 sur la grille native, pas 16. Marge par fenêtre, moyenne par cellule 4x4, seuil
appris. Coût : environ 100 fenêtres x 1530 images x 3 méthodes : minutes sur un A100. Gain attendu : correct sur les gros rectangles, flou sur le reste.

### E. CAM brute avec le kit, sans entraînement (IDÉE ; 1 h ; filet de sécurité)
`feats = wm.features(x)["per_position_logits"]` (PixelSeal et TrustMark : 8x8). `vote = ((2*m-1)[None,:,None,None]*feats).sum(1)`. Seuil par image
(0 ou médiane). Un passage ; les cases font environ 64 px et débordent sur les voisines (champ de vue large du réseau). Utile comme canal de C et pour
deviner une boîte de départ pour A.

### F. Détecteur de boîte TrustMark prêt à l'emploi (PAPIER : code Adobe ; 0,5 à 1 h)
`TrustMark(use_ECC=False, secret_len=100, model_type="Q", loadBBoxDetector=True).localize(PIL_image)` rend la boîte normalisée ; masque = rectangle.
À mesurer sur les 170 images TrustMark du train : si l'IoU de boîte dépasse celui de A, l'utiliser comme départ de A. Ne sert qu'à TrustMark.

### G. Bases de sécurité (30 min à 2 h ; première soumission valide)
1. Tout 0 ; tout 1 ; choix par image : marge de l'image entière > seuil => tout 1, sinon tout 0 ; calibrer par méthode sur le train.
2. Rectangle central fixe (taille = médiane du train par méthode). 3. Médiane des masques train par méthode, seuillée à 0,5.
Pas d'estimation de score fiable : mesurer. Sert de plancher et prouve que la chaîne de soumission marche.

Tableau des estimations (toutes [IDÉE], à remplacer par la mesure du pli de validation) :

| Méthode | Proposé | GPU h | Personne-h | Risque | PixelSeal | TrustMark | StegaStamp |
|---|---|---|---|---|---|---|---|
| A boîte exacte + trous | PAPIER 0 / IDÉE | 1-2 | 6-8 | moyen | très bon | très bon | bon (plus lent) |
| C réseau de lecture | PAPIER partiel | 5-8 | 12-16 | moyen-haut | bon | bon | bon |
| B corrélation embedder | IDÉE | 1-2 | 4-6 | haut | moyen | moyen | bon |
| D fenêtres alignées | IDÉE | <1 | 3-5 | moyen | moyen | moyen | moyen |
| E CAM brute | IDÉE | <0,5 | 1 | moyen | moyen | moyen | non (logits globaux) |
| F boîte TrustMark | PAPIER (code) | <0,5 | 1 | moyen | non | bon | non |
| G bases | IDÉE | <0,5 | 0,5-2 | nul | plancher | plancher | plancher |

## 4. Les quatre verdicts

- La meilleure : A (boîte exacte par le décodeur) pour fixer la forme principale, puis C pour les trous et les bords. Raison : on teste un
  fait statistique net (message retrouvé ou non) au lieu d'estimer une carte floue. Condition : l'exp. 0 confirme le cadre « rectangle ».
- La plus difficile : C avec réglage fin sur les éditions génératives (données synthétiques réalistes, réseau par méthode, validation croisée) ;
  ou, encore plus dur, passer l'extracteur lui-même en « pixelwise » et l'affiner (voir ci-dessous).
- La plus technique : affiner l'extracteur comme segmenteur, à la WAM. Pour PixelSeal le dépôt a déjà `pixelwise: True` (couche 1x1 par position,
  sortie 1+256 canaux : canal 0 « filigrané ? ») : initialiser la couche 1x1 avec les poids de la couche linéaire actuelle, entraîner le canal 0 avec
  BCE sur le masque, dégeler le ConvNeXt à faible pas d'apprentissage. TrustMark : ResNet-50 convolutif, retirer le pooling, même procédé.
  StegaStamp : ONNX, non entraînable sans conversion (onnx2torch) : à écarter. 10 à 14 h par méthode ; GPU 3 à 6 h. Risque haut, gain incertain.
- Base sûre la plus rapide : G (1 à 2 h), puis E et F (1 h chacun) pour avoir une carte non triviale avant le dîner. Ordre conseillé : G, exp. 0, A, C.

## 5. Stratégie par filigrane

PixelSeal (256 bits, 256 px, ConvNeXt-tiny, ratio max 16:1) :
- Le plus de bits donc le test le plus net (bruit à 50 % +- 3 points) : A fonctionne le mieux ici. Rognage anisotrope vers 256x256 (un 16:1 est écrasé).
- Filigrane calculé à 256 puis agrandi, avec atténuation JND (plus fort là où l'oeil tolère : zones texturées). En zones lisses (ciel, mur)
  la marge est faible : attendre de faux « propre » là-bas [IDÉE]. Donner l'image à C lui permet d'apprendre cette règle.
- Carte par position 8x8 disponible (`features()`).
TrustMark Q (100 bits, 245 px, ResNet-50, ratio max 2:1 donc boîte presque compacte) :
- A + F ensemble. Forte contrainte de forme : post-traiter en rectangle, puis trous. Pas de correction d'erreur (100 bits bruts) : bruit à 50 % +- 5.
- Bord fondu de 1 % du côté (max 50 px) : le bord vrai est flou sur 5 px à 512, donc ne pas chercher mieux que 4 px (une cellule).
StegaStamp (100 bits, 400 px, ONNX, ratio max 16:1) :
- Pas de carte par position dans le kit : A (recherche de boîte) et D uniquement, plus B (résidu fort). Décodeur robuste à l'impression, donc tolérant au
  décalage : moins de précision du maximum ; faire la montée par coordonnées en pas 8 puis 4.
- ONNX Runtime 400 px : environ 300 à 800 images/s sur A100 (estimation) ; budget de candidats 2 fois plus petit.

## 6. Pièges

- Règle des cellules : une cellule 4x4 vaut 1 si au moins la moitié de ses pixels sont filigranés. Évaluer avec `utils.downsample_mask` et `miou_image`, jamais à la main.
- Classe vide = IoU 1 : une image sans zone propre (ou sans zone filigranée) rapporte 1 si on la prédit pareil, 0,5 si on se trompe de 1 cellule.
  Éviter les parasites isolés ; prévoir une décision « tout filigrané » / « tout propre ».
- Tous les filigranes sont connus au test : router par `item["method"]`, un jeu de seuils par méthode.
- Cadre du motif (exp. 0) : si le motif est calé sur le rectangle et qu'on lit l'image entière, la marge s'effondre et on conclura à tort « propre ».
- Aspect : rogner un rectangle 16:1 puis le redimensionner en carré ; ne pas rogner en carré par réflexe.
- Éditions génératives : le banc 2609.16832 montre que le décodage peut tomber au hasard même loin de la zone retouchée. Pas de marge = ne pas conclure
  « pas de filigrane » trop vite : tester avec plusieurs messages faux (témoin) avant de décider.
- Écart synthétique/réel : toute décision (seuil, modèle) se prend sur le train réel validé en 5 plis par méthode, jamais sur le classement public (30 % du test).
- Données d'entraînement synthétiques : images propres à trouver. Options : (a) copier une base publique sur le nœud de login (COCO val 1 Go) ; (b) cadrer
  les zones propres des images train (masque = 0) ; (c) demander aux organisateurs. Sans réseau sur les nœuds de calcul, tout préparer avant.
- Fichier de soumission : tenseur bool 1020x128x128 (17 Mo), noms sans .png, chargement `weights_only=True` : seulement tenseurs, listes, chaînes, dicts.
- StegaStamp via ONNX : par lots de 32 (déjà dans le wrapper), pas de gradient ; ne pas planifier d'optimisation par gradient pour ce filigrane.
- Aucune soumission lancée par ce rapport ; aucun appel payant.

## 7. Plan conseillé (durée réaliste, une à deux personnes)

1. 0 h 00-0 h 30 : exp. 0 + base G + première soumission de contrôle (rien d'autre). Mesurer le score local de G sur le train.
2. 0 h 30-3 h 00 : A sur TrustMark et PixelSeal, mesure sur train (3 minutes de GPU par passe). F en parallèle (autre personne).
3. 3 h-6 h : A sur StegaStamp ; cartes B, E stockées pour C ; générateur synthétique lancé (4 GPU).
4. 6 h-14 h : C par méthode, validation 5 plis, calibrage du post-traitement sur le mIoU officiel.
5. 14 h-17 h : choix final par méthode (A seul, C seul, ou A + C), 1 ou 2 soumissions au plus.
6. Si A échoue à l'exp. 0 : D + E + C ; la méthode la plus technique (extracteur affiné) seulement si une personne est libre.
