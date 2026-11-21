# T1 localisation de filigrane : critique adverse, tour 1

Sam. 21/11, ~17 h. Lu : sujet, kit (watermarks.py, utils.py, data.py), les deux plans, les quatre fichiers T1 de Titouan, exp0.py, exp1.py.
Décision débloquée : les 2-3 prochaines expériences et ce qu'on arrête. Aucun job GPU lancé. [MESURÉ] = vos chiffres d'exp0. [DÉDUIT] = calcul tiré
de ces chiffres. [À VÉRIFIER] = hypothèse, avec l'expérience qui tranche. Tout chiffre sans [MESURÉ] est une estimation.

## 1. Ce que disent vraiment les chiffres d'exp0

1. **La règle « classe vide = 1 » ne sert à rien ici.** [DÉDUIT] all0 = 0,253 ≈ (1 − frac)/2 = 0,2535 ; all1 = 0,247 ≈ frac/2. S'il existait des images
   entièrement propres ou entièrement filigranées, les masques constants y gagneraient 1 et dépasseraient ces valeurs. Ce n'est pas le cas :
   presque aucune image n'a de classe vide. Conséquences : (a) l'idée classée n° 1 par Titouan (I5, tri tout-0 / tout-1) et la règle « snap » valent ~0 :
   on les abandonne ; (b) chaque image contient les deux classes, et la plus petite pèse la moitié du score. Un trou de 3 % compte autant que le
   reste de l'image. La précision sur la petite classe est le levier principal.
2. **Le 0,66 de la vraie boîte n'est pas un plafond : il mélange deux populations.** fullwm vaut 0,500 pile pour les trois filigranes, donc le jeu
   est construit moitié-moitié. Moitié A (partielle) : filigrane posé sur un rectangle, puis éditions. Moitié B (full_watermark) : toute l'image
   est filigranée, puis une zone est remise à l'état propre. En B, si la zone remise ne coupe pas tout un bord, la boîte du masque couvre l'image
   entière. La boîte prédit alors tout-1 : IoU propre = 0, score ≈ frac/2 ≈ 0,4. [DÉDUIT, à vérifier par E1] 0,66 ≈ 0,5 × (~0,88 sur A) + 0,5 ×
   (~0,44 sur B). Lecture correcte : sur A, « boîte puis trous » est déjà presque au maximum ; les points perdus sont dans B et dans les trous.
   Le plan (méthode A d'abord, trous ensuite) optimise la moitié déjà facile.
3. **fill_lt_095 ≈ 0,52 et fullwm = 0,50 désignent sans doute la même population** [À VÉRIFIER : croiser les deux colonnes]. Si c'est le cas,
   « rectangle plein » ≈ partiel et « forme libre ou trouée » ≈ full_watermark. Prédire full_watermark revient alors à prédire le type de masque.
4. **acc_full ≈ 0,75 avec frac ≈ 0,5 : le décodeur fait une moyenne, il ne choisit pas une zone.** PixelSeal et TrustMark font une moyenne spatiale
   puis une couche linéaire : le signal est dilué en proportion de la surface filigranée. La moyenne de 0,75 est sûrement bimodale (~1,0 quand
   beaucoup de surface est filigranée, ~0,5 quand peu l'est). Elle ne veut pas dire « 75 % des bits survivent ». Conséquence directe : une
   fenêtre de 64 px représente 1/64 de l'image, donc son effet sur la marge (moyenne signée des logits, positive si notre message est lu) est
   ~64 fois plus petit. mask64 et crop64 n'échouent pas parce qu'ils sont mal réglés : ils sont trop faibles par construction.
5. **acc_box 0,81-0,85 < 1 ne prouve pas que les éditions ont détruit des bits.** En B, la boîte ≈ l'image entière, donc acc_box = acc_full sur
   cette moitié. Avec ~1,0 sur A et ~0,65 sur B, on retombe à 0,82. La dilution par les trous et le mélange des deux moitiés suffisent à
   l'expliquer. acc_out ≈ 0,50 [MESURÉ] apporte deux bonnes nouvelles : la zone propre donne bien du hasard (la comparaison avec des messages
   faux est donc valide), et aucun ré-encodage global n'a laissé de filigrane hors zone.
6. **La CAM 8x8 (vote du décodeur position par position) fait perdre des points.** cam_box (0,62) < bbox (0,66) : creuser des trous avec la CAM
   dégrade la vraie boîte. cam (0,35) dépasse à peine all0 (0,25). Le seuil 0 est fixe et chaque position voit presque toute l'image, donc les
   cases propres voisines « voient » le filigrane : la carte fuit. On arrête d'investir dans la CAM brute.
7. **La recherche de boîte exacte (méthode A) n'est pas confirmée.** Un gain de +0,06 à +0,10 entre image entière et vraie boîte, mesuré sur un
   mélange, ne montre pas le « pic net au bon cadre » attendu. Les décodeurs sont entraînés à résister au recadrage et au redimensionnement :
   la marge est sans doute plate autour de la bonne boîte. A reste une hypothèse tant que E3 n'a pas montré de pic.
8. **Le classement** : 0,26 ≈ masque constant, 0,39 = modeste, vraie boîte = 0,66. Il reste beaucoup de marge, et une équipe avec un segmenteur
   appris peut passer à 0,8 d'ici demain. Avec une remise à l'échelle min-max entre équipes, viser 0,45 ne protège de rien.

## 2. Défauts d'exp1

- **Fenêtres trop petites pour le signal.** Avec 256 bits (PS) ou 100 bits (TM, SS) et une moyenne globale, une fenêtre de 64-128 px ne porte
  pas assez de signal. Pour crop, il y a en plus un problème d'échelle : rogner 64 px puis agrandir à 256 étire le motif ×4, alors que l'image
  entière le réduit ×0,5. En B, l'erreur d'échelle est donc d'un facteur 8. En A, l'échelle dépend de la taille s du rectangle (motif étiré de
  s/256). Aucune taille fixe ne convient aux deux moitiés.
- **Le masquage par la couleur moyenne sort le décodeur de son domaine.** Un grand aplat à bords nets ne ressemble à rien de ce qu'il a vu
  à l'entraînement. StegaStamp (transformateur spatial + couches denses fixées sur la position) y réagit même en zone propre, donc le delta
  d'occlusion est biaisé même sur du propre. Il faut un remplaçant qui garde le contenu et retire seulement le filigrane (flou léger, médian
  3x3, aller-retour ×0,5 : à choisir avec E4).
- **Occlusion ou insertion.** Effacer une seule fenêtre change peu une marge saturée : quand le filigrane est fort et redondant (StegaStamp
  surtout), la carte est plate ; quand il est faible, elle n'est que du bruit. Un cache à la fois est le pire réglage. Mieux : des caches
  aléatoires couvrant ~50 % de l'image, puis une régression de la marge sur les cases (RISE, arXiv 1806.07421). Le décodeur reste dans sa
  plage sensible, et chaque case est estimée sur des centaines de tirages.
- **Seuil global sur des cartes non normalisées.** L'amplitude de la marge varie d'environ ×10 d'une image à l'autre (force, texture, surface).
  Un seuil commun met à 0 toutes les cellules des images faibles, qui tombent alors à ~0,25. Il faut normaliser chaque image par un z-score
  contre K = 64 messages aléatoires. C'est gratuit à condition de garder les logits signés, et pas seulement la marge, ce qu'exp1 ne fait pas.
  Otsu (seuil calculé sur l'histogramme de chaque image) est bien par image, mais casse quand le filigrane couvre plus de ~70 % ou moins de
  ~15 % de l'image.
- **Seuil réglé sur les données d'évaluation.** On garde le meilleur de 49 seuils × ~15 cartes. Le biais « meilleur tirage » vaut environ
  +0,01 à +0,02, soit l'ordre de grandeur des écarts entre cartes. Régler sur 4 plis, lire le 5e.
- **grad = |gradient|** : prendre la valeur absolue perd le signe et fait ressortir les textures (là où l'atténuation visuelle JND de PixelSeal met
  de la force), pas le filigrane. gradxin multiplie par (x − moyenne), c'est-à-dire par le contenu. Il faut multiplier le gradient signé par
  la haute fréquence de x, là où vit le motif.
- **Il manque** : (a) le canal 0 « détection » de PixelSeal, coupé par le kit (`[:, 1:]`), dont la carte par position n'a pas besoin du message ;
  (b) une base « énergie haute fréquence locale » : StegaStamp ajoute un résidu fort, qu'une simple variance locale du bruit peut déjà situer ;
  (c) une découpe par type plus fine que fullwm/partiel : fill ≥ 0,95, tertiles de frac, taille du trou ; (d) le score de la vraie boîte à côté
  de chaque carte, par population.
- **Bruit.** Avec n = 170 et un écart-type par image d'environ 0,2, l'erreur sur la moyenne est ~0,015, et ~0,01 en comparaison appariée. Le test
  sur 3 images ne vaut rien : n'en concluez pas que « l'occlusion est meilleure ».
- **Résolution.** Moyenner les fenêtres qui couvrent une cellule floute la carte à l'échelle de la fenêtre (±64 px pour 128). Les objets fins
  et les trous de moins de 128 px disparaissent, alors que d'après le point 1.1 ils font la moitié du score.

## 3. Ce que ferait un spécialiste (classé par gain / coût)

**1. Segmenteur appris sur données synthétiques faites avec les vrais embedders (gros gain, ~6-10 h-GPU, 1 personne).** C'est la pièce qui manque au
plan, pas une option pour plus tard. Le problème n'est pas de lire un message, c'est de reconnaître la texture du résidu (ce que l'embedder ajoute).
La stéganalyse (la détection de messages cachés) apprend très bien ce genre de texture quand l'algorithme est connu (SRNet, Boroumand 2019), et
WAM (arXiv 2411.07231) montre qu'une tête par pixel l'apprend aussi. Notre atout, qu'aucun article n'a : des données en quantité illimitée.
- Images propres : d'abord les zones propres des 510 images du train (le bon domaine), puis COCO val2017 / DIV2K copiés depuis le nœud de login.
- Formes : **réutiliser les 510 vrais masques du train** (retournés, décalés, échangés entre images) plutôt que d'inventer des générateurs. On obtient
  directement la bonne loi de surfaces, de trous et de contacts avec les bords.
- Deux simulateurs. Pour B : `wm.embed(image entière, m)` puis recoller les pixels propres d'origine dans la forme. C'est exactement la recette
  décrite par le sujet, donc zéro écart avec le réel. Pour A : `wm.embed(rectangle, m)` recollé, avec des tailles tirées de la loi des boîtes du
  train (l'échelle du motif dépend de s). Les trous sont faits par copier-coller propre, avec un bord adouci de 1-3 px, et pour une partie par un
  aller-retour dans l'encodeur-décodeur (VAE) de Stable Diffusion à l'intérieur du trou, pour imiter une texture générée. Aucune banque de
  diffusion (SD2/SDXL/FLUX) avant d'avoir prouvé que ces simulateurs ne suffisent pas.
- Réseau : PixelSeal = ConvNeXt du détecteur, TrustMark = ResNet-50 du décodeur, tous deux initialisés avec leurs poids (déjà accordés au motif),
  plus une tête légère FPN/U-Net qui sort du 128x128. Entrée à 512, sans réduire à 256 : un rectangle de 128 px perdrait la moitié de son résidu.
  StegaStamp : U-Net ResNet-34 ImageNet (poids préchargés) + un canal haute fréquence. Son résidu est fort, c'est probablement le plus facile.
  Un réseau par filigrane.
- Perte : entropie croisée binaire + Lovász-hinge par image sur les deux classes (arXiv 1705.08790), la meilleure approximation dérivable de la
  métrique. Aucune règle spéciale pour la classe vide.
- Augmentations : uniquement des décalages multiples de 4 px et des symétries. Pas de rééchantillonnage, de JPEG ni de flou global, qui
  effacent le signal.
- Conditionner le réseau sur le message (carte d'accord par position) seulement en v2, si la version aveugle échoue sur les zones lisses.

**2. Carte sans entraînement, mais propre (gain moyen, ~2 h ; filet de sécurité et canal d'entrée pour 1).** Caches aléatoires à ~50 % sur une
grille 16x16, avec le remplaçant choisi par E4. On régresse la marge, en z contre 64 messages faux, sur la grille du cache, avec 500 à 1000 passes
par image : environ 10 min par méthode pour PS et TM [À VÉRIFIER]. Pour PS et TM, deux variantes possibles. (a) Rogner à l'échelle native :
couper directement dans l'image déjà réduite à 256 ou 245, sans redimensionner. Ça marche si le réseau accepte toute taille grâce à sa moyenne
spatiale, ce qu'il faut vérifier dans `det.preprocess`. (b) Optimiser un masque doux (Fong et Vedaldi, arXiv 1704.03296).

**3. Densifier le décodeur sans entraînement (faible seul).** Passer les dernières étapes en pas 1 avec dilatation (l'astuce de DeepLab, arXiv
1606.00915), ou faire du « shift-and-stitch » (FCN, arXiv 1411.4038) : 16 décalages de 8 px donnent un vote 32x32. La carte est plus fine, mais
la fuite du point 1.6 reste. Ça ne sert que de canal d'entrée.

**4. Post-traitement par la forme et l'IoU espérée (+0,02-0,05, 1 h).** Candidats : la carte seuillée à plusieurs niveaux ; le rectangle de
somme maximale (algorithme de Kadane en 2D) sous les ratios du sujet (TM ≤ 2:1, PS/SS ≤ 16:1) ; ce rectangle moins ses trous. On choisit par image
celui qui maximise l'IoU espérée sous la carte de probabilités calibrée (Nowozin, CVPR 2014, décisions optimales pour l'IoU). Aucun paramètre
appris, donc aucun surapprentissage. On ne retire les composantes de moins de 4 cellules que si l'IoU espérée monte.

**5. Ensemble.** L'affinage en 5 plis sur le réel donne 5 modèles : moyenne de leurs probabilités, plus une moyenne sur 4 décalages de 4 px.
Gain attendu +0,01-0,02, presque gratuit.

## 4. La raison la plus probable d'un échec sur le test caché

**Toutes nos preuves viennent d'un décodeur qui fait une moyenne globale. Il dit « combien » de filigrane, jamais « où ».** Sur la moitié
full_watermark, le filigrane est partout sauf dans un objet remis à l'état propre. Toute carte tirée du décodeur sera positive presque partout :
on prédira presque tout-1, on manquera la classe propre, et ces images resteront vers ~0,4 quel que soit le réglage. Les petits objets
transplantés seront dilués sous le seuil. Ce n'est pas un écart entre train et test (le test a la même composition, 340 par méthode, moitié-moitié) :
c'est un plafond de la méthode, qu'on ne voit pas en regardant la moyenne.
Comment le voir dès ce soir : rapporter chaque score séparément sur full_watermark et sur partiel, et tracer le score par image en fonction de frac.
Si une carte fait < 0,5 sur full_watermark et > 0,8 sur partiel, c'est confirmé. exp1 calcule déjà best_fullwm / best_partial : à lire en premier.
Risque n° 2, une fois le segmenteur adopté : l'écart entre synthétique et réel. On le mesure en évaluant le modèle entraîné seulement sur du
synthétique sur les 510 vraies images, qu'il n'a jamais vues (E2).

## 5. Feu vert / feu rouge pour le prochain tour (5 expériences au plus)

| # | Expérience | Coût | Mesure | Ce qui change la décision |
|---|---|---|---|---|
| E1 | Relancer exp0 en enregistrant une ligne par image. Croiser bbox / acc_box / acc_full avec fullwm × (fill ≥ 0,95) × tertiles de frac, et fill_lt_095 avec fullwm. | 10 min GPU | mIoU de la vraie boîte par population | bbox ≥ 0,85 sur partiel et ≤ 0,5 sur full_watermark : priorité aux trous et à la moitié B, on arrête de peaufiner les boîtes. |
| E2 | **Pari principal.** Segmenteur aveugle PS + SS (simulateurs du §3.1, ~20 k paires par méthode, 45 min d'entraînement chacun), évalué tel quel sur les 510 vraies images. Contrôle de fidélité d'abord : acc_box sur rectangles synthétiques ≈ acc_box réel pour fill ≥ 0,95. | ~3 h-GPU sur 2 GPU | mIoU officiel sur le vrai train, par population | ≥ 0,60 : tout le monde dessus (affinage 5 plis, puis TM). 0,45-0,60 : affiner sur le réel et regarder les 20 pires images. < 0,45 : l'embed simulé ne colle pas au réel ; comparer les statistiques des résidus avant de toucher au réseau. |
| E3 | Netteté de la boîte (méthode A) : marge du rognage de la vraie boîte décalée de ±8/16/32 px et mise à l'échelle ×0,8/0,9/1,1/1,2, sur 30 images partielles par méthode. | 15 min | perte de z (contre messages faux) entre boîte exacte et boîte décalée de 16 px | perte < 2 : A est enterrée. Perte ≥ 4 : A sert à affiner les bords de la sortie de E2. |
| E4 | Choix du remplaçant : decode(T(x)) pour T ∈ {flou σ = 1 et 2, médian 3, ×0,5 puis ×2, JPEG 50} sur 30 images, puis RISE 16x16 avec le meilleur T et un z par image. | 1 h | précision ≤ 0,55 avec PSNR > 35 dB ; puis mIoU de RISE par population | RISE < 0,50 : simple canal d'entrée de E2, plus de temps humain dessus. RISE ≥ 0,60 : soumission de secours de ce soir. |
| E5 | Soumission ce soir : meilleure carte d'exp1 (z par image, seuil réglé hors pli) + post-traitement du §3.4. Elle teste la chaîne et pose un plancher. | 30 min | score public face au local (± 0,03) | public < local − 0,05 : chercher d'abord un bug d'ordre ou de format. |

**À arrêter maintenant** : la CAM 8x8 seuillée ; les fenêtres ≤ 96 px masquées ou rognées ; le tri tout-0 / tout-1 et le « snap » ; la recherche
de boîte A avant E3 ; la banque de diffusion SD2/SDXL/FLUX avant E2 ; les nouvelles recherches bibliographiques (cette passe n'a rien trouvé de
neuf sur la localisation post-hoc de ces trois filigranes).

Sources : [WAM 2411.07231](https://arxiv.org/abs/2411.07231) · [banc local 2609.16832](https://arxiv.org/pdf/2609.16832) ·
[MaskMark 2504.12739](https://arxiv.org/pdf/2504.12739v1) · [Where is the Watermark? 2512.14994](https://arxiv.org/abs/2512.14994v1) ·
[Pixel Seal 2512.16874](https://arxiv.org/pdf/2512.16874) · RISE 1806.07421 · Fong & Vedaldi 1704.03296 · Lovász 1705.08790 · DeepLab 1606.00915 ·
FCN 1411.4038 · Nowozin CVPR 2014 (ces six derniers cités de mémoire, liens non rouverts).
