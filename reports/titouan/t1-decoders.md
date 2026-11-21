# T1 : fonctionnement interne des décodeurs et évidence spatiale (angle t1-decoders)

Légende : [KIT] = lu dans watermarks.py/README/énoncé ; [WEB] = lu sur le web (URL citée) ; [MÉMOIRE] = connaissance de l'architecture, à vérifier en inspectant le modèle ; [HYP] = jugement, non mesuré. Tous les chiffres de débit/mIoU sont des estimations non mesurées. Je n'ai rien téléchargé ni exécuté. Les pages arXiv n'ont donné que les résumés (le PDF PixelSeal dépasse 10 Mo pour l'outil), donc le détail interne vient du kit, de la carte YAML videoseal et de ma mémoire.

Ce rapport va PLUS LOIN que reports/t1-watermark-etat-de-lart.md (fenêtre glissante, per_position_logits, saillance, résidu y sont déjà esquissés). Nouveautés ici : (1) l'hypothèse d'échelle « le motif est étiré sur le rectangle », qui change tout le choix des fenêtres ; (2) calibrage nul par messages aléatoires ; (3) déconvolution linéaire des fenêtres vers 128x128 ; (4) occlusion par flou et intégrales de gradients ; (5) rectangle de somme maximale (Kadane 2D) ; (6) règles de score propres à la métrique ; (7) budget de calcul chiffré et structure de code.

## 0. Ce que la métrique impose (guide toutes les décisions)

- IoU par classe, vide = 1. Conséquences [KIT, calculé] :
  - GT tout-1 (aucune cellule propre) : prédire UNE seule cellule propre donne IoU0 = 0 (P0 non vide, G0 vide), score image 0,5 au lieu de 1.
  - GT presque tout-1 avec un petit trou : prédire tout-1 donne IoU0 = 0, donc ~0,48. Les petits trous pèsent autant que les gros : la moitié du score image dépend de la classe minoritaire.
  - Un objet transplanté petit : prédire vide donne IoU1 = 0, ~0,5.
- Donc : (a) seuil non symétrique, optimisé pour l'IoU espérée par méthode ; (b) règle « snap » : si la fraction de cellules 1 prédite > a ou < b, forcer tout-1 / tout-0 (a, b choisis en validation croisée) ; (c) retirer les composantes minuscules uniquement si la classe est probablement vide.
- À mesurer en 10 min sur le train avant tout : fraction de masques tout-1, tout-0, rectangle pur, rectangle moins trou(s), forme libre ; distribution de (x0,y0,w,h,aspect) de la boîte englobante par méthode ; part des `full_watermark`. Ces histogrammes servent d'a priori (§8).

## 1. Hypothèse structurante : le motif est étiré sur le rectangle (à valider en Expérience 0)

Lecture du kit [KIT] : `embed(img_patch)` accepte « any size » ; les trois embedders travaillent à une résolution interne fixe, puis le résidu est ré-échantillonné à la taille du patch.
- StegaStamp : explicite dans le code : patch -> 400x400 bicubique, résidu (moyenne de canal retirée) -> taille du patch bilinéaire, ajouté.
- TrustMark : [MÉMOIRE] encode redimensionne en 256x256, résidu remis à la taille d'origine puis ajouté ; décodeur à 245 [KIT].
- PixelSeal : [MÉMOIRE] videoseal.embed calcule le résidu à 256 puis l'agrandit, atténuation JND à pleine résolution (« lowres_attenuation », cohérent avec « JND attenuation + inference simulation to reduce upscaling artifacts » [WEB arxiv.org/abs/2512.16874]).
Conséquence : si le patch filigrané mesure s px dans l'image 512, le motif est celui d'une grille 256 (ou 400) étirée d'un facteur s/256 (s/400). Décoder l'image 512 entière redimensionnée à 256 (facteur 0,5) montre le motif à l'échelle 0,5*s/256 : seulement correct si s = 512. Pour un patch de 192 px, l'échelle vue est 0,375 au lieu de 1 : mismatch ~2,7x, que le décodeur tolère plus ou moins (il a été entraîné avec des redimensionnements et recadrages, [MÉMOIRE]).
=> Décoder un crop dont la taille = celle du rectangle d'origine (puis redim. à la résolution du décodeur) remet le motif à l'échelle native et doit donner la meilleure bit accuracy. C'est un signal d'ALIGNEMENT (taille + position), bien plus informatif qu'une fenêtre arbitraire. Si c'est vrai : on cherche le rectangle, pas seulement des cellules.
- Rectangles non carrés (jusqu'à 16:1) : l'embedder a probablement écrasé le patch en carré anisotrope (à vérifier). Alors les fenêtres de recherche doivent être anisotropes (même rapport que le rectangle, redimensionnées en carré anisotropiquement). TrustMark : rapport <= 2 donc peu critique. [KIT]
- Outpainting : l'image filigranée a été étendue puis (probablement) remise à 512 : le rectangle wm est mis à l'échelle mais le motif reste proportionnel au rectangle : l'hypothèse tient.
- Transplantation : motif étiré sur le rectangle d'origine dont on ne garde que l'objet. Le crop du rectangle d'origine n'existe plus dans l'image de test (l'objet est collé ailleurs, à une autre échelle ?) : cas le plus dur, signal probablement faible.

Expérience 0 (30 min, 20 images train/méthode, 1 GPU) : pour chaque image, boîte englobante GT, puis bit accuracy de (i) image entière, (ii) crop boîte redimensionné carré, (iii) crop boîte avec marge +-32 px, (iv) fenêtres carrées de 128/192/256 centrées dans la boîte, (v) fenêtres hors boîte (nulle). Décision : si (ii) >> (i), le plan « recherche de rectangle » devient prioritaire ; sinon fenêtres glissantes ordinaires.

## 2. Les trois décodeurs en détail

### 2.1 PixelSeal (256 bits)
- Embedder [WEB carte YAML raw.githubusercontent.com/facebookresearch/videoseal/main/videoseal/cards/pixelseal.yaml, résumé non verbatim] : U-Net `unet_base_yuv_quant` à 1 canal d'entrée/sortie (donc canal Y luminance seulement, [MÉMOIRE] cohérent avec « yuv »), 8 blocs, BN, tanh final, fusion additive, échelle 0,2 ; atténuation `jnd_1_1`. Résolution de travail 256, 256 bits. Entraînement adversarial seul (pas de MSE/LPIPS), 3 étapes [WEB arxiv 2512.16874].
- Décodeur [KIT, WEB] : ConvNeXt-Tiny (profondeurs 3-3-9-3, dims 96->768) + `pixel_decoder` (upscaling `output_upscaling`, puis moyenne spatiale et couche linéaire). `logits()` = `det(resize 256)[:,1:]` ; canal 0 = détection. Le résumé YAML parle de 16 bits de sortie : probable erreur de résumé (le kit dit 256) ; vérifier `emb.shape` et `lin.weight.shape` au premier lancement.
- Entrée : image 512 -> 256 bilinéaire antialias (facteur 0,5) [KIT]. Sortie : 256 logits (>0 = 1) + 1 canal de détection.
- Décomposition exacte : logits = W * mean_p(emb_p) + b => logit_i = mean_p (W_i . emb_p) + b_i. La carte `per_position_logits` [KIT] est la contribution EXACTE de chaque position au logit (moyenne = logit). Résolution = celle de `output_upscaling` appliquée à 8x8 (probablement 16x16 à 32x32 ; vérifier). MAIS le champ réceptif du ConvNeXt (18 blocs conv 7x7 au stade 8x8 sur 256 px) est quasi global : une position « voit » toute l'image. La carte est donc plutôt grossière (fuites), utile surtout comme canal d'entrée du réseau d'appui (stage C) et pour un 1er pas.
- Placement du résidu : JND = masquage de luminance + texture : amplitude forte dans les zones texturées/lumineuses, faible dans les aplats (ciel, peau lisse) [MÉMOIRE sur jnd_1_1 ; à lire dans videoseal/modules/jnd.py]. Pas de renforcement spécifique sur les bords du rectangle. => Dans un aplat, l'évidence est ~0 PARTOUT (filigrané ou non) : « pas de signal » != « propre ». Il faut une carte de confiance (JND/variance locale) comme canal séparé et un repli sur l'a priori spatial.
- Robustesse : l'article PixelSeal annonce robustesse aux transformations courantes ; le chiffre 0,58 de bit accuracy sur le plus petit recadrage (aire 0,32) vient du rapport coéquipier (non revérifié ici). Avec 256 bits, sigma du hasard = 0,5/sqrt(256) = 3,1 points : 0,58 vaut z=2,6. Inpainting génératif : le contenu remplacé perd le motif ; les pixels conservés autour gardent leur résidu (si le pipeline d'inpainting recolle l'original hors masque ; si la VAE ré-encode toute l'image, le motif est atténué partout : à observer sur le train).
- Cartes disponibles : `convnext` 768x8x8, `embedding`, `per_position_logits` ; plus gradients autograd (PyTorch pur, donc différentiable) et stades intermédiaires ConvNeXt (96@64x64, 192@32x32, 384@16x16 à 256 px) accessibles via `det.convnext.stages` [MÉMOIRE, à vérifier].

### 2.2 TrustMark Q (100 bits, sans ECC)
- Embedder : U-Net/GAN à pertes spatio-spectrales [WEB arxiv.org/abs/2311.18297] ; résidu calculé à 256 puis ré-échantillonné ; image d'origine conservée à sa résolution + résidu [MÉMOIRE]. Pas de JND explicite (la qualité vient des pertes) ; le résidu est de haute fréquence, couvre tout le patch.
- Décodeur [KIT] : ResNet-50, entrée 245x245 normalisée [-1,1], GAP + fc -> 100 logits. Facteur de réduction 512 -> 245 : 0,48.
- Décomposition exacte (CAM) : logit_i = mean_p(W_i . layer4_p) + b_i, layer4 = 2048x8x8 (245/32 -> 8). Champ réceptif ResNet-50 ~ 480 px > 245 : global (fuites). Pour plus de finesse : Grad-CAM sur layer3 (1024x16x16, stride 16, cases de 32 px dans l'image 512) et layer2 (512x31x31).
- Robustesse : conçue pour résolution arbitraire et modifications in-place/out-of-place [WEB]. 100 bits : sigma = 5 points, donc un petit crop est plus bruité que PixelSeal. Rapport <= 2:1 : le patch entier est filigrané [KIT] : le masque est un rectangle compact -> a priori de forme fort.
- Cartes : layer1..4, per_position_logits [KIT].

### 2.3 StegaStamp (100 bits)
- Embedder [KIT] : encodeur ONNX WAVES à 400x400 ; résidu = sortie encodeur, moyenne par canal retirée, agrandi bilinéaire, ajouté (clip uint8). Résidu basse fréquence, coloré, dépendant du contenu, bien visible (pas de JND) ; très peu de bords nets.
- Décodeur [MÉMOIRE de l'article 1904.05343 ; à confirmer en listant les nœuds de `stega_stamp_logits.onnx`] : réseau de localisation spatial (STN, transformée affine apprise) puis 7 convolutions (32..128 canaux, 5 stride 2) puis flatten -> dense 512 -> dense 100. Pas de GAP : le décodeur est SENSIBLE À LA POSITION et au cadrage (flatten) ; le STN compense en partie les petits recadrages/rotations. Sortie utilisée : `gen_decoder/decoder_out/BiasAdd:0` (pré-sigmoïde) [KIT].
- Entrée : image -> 400 bicubique antialias (facteur 0,78 sur 512) [KIT]. ONNX, donc NON différentiable par PyTorch : pour le gradient, convertir avec onnx2torch (ou réimplémenter en 30 lignes depuis les poids numpy_helper) ~1 h ; sinon uniquement évidence par forward (fenêtres, occlusion). Les features intermédiaires s'obtiennent en ajoutant des sorties au graphe (comme fait pour les logits) [KIT].
- Robustesse : entraîné pour impression + photo (flou, bruit, jitter couleur, perspective) [MÉMOIRE] -> robuste aux ré-encodages, fragile aux recadrages forts non compensés par le STN, et aux remplacements génératifs de contenu. Logits fortement saturés (bits soit corrects soit aléatoires) : utiliser clip/tanh sur les logits dans les marges.
- Pas de décomposition spatiale exacte (dense 512 + ReLU entre les deux).

## 3. Évidence par fenêtres glissantes avec le message CONNU

Pour une fenêtre W (x0,y0,w,h) : crop -> redim. carré (antialias pour réduire, bicubique/bilinéaire pour agrandir) -> logits l in R^n. Statistiques :
- Accord dur : a = mean_i [ (l_i>0) == m_i ].
- Marge douce : s = mean_i (2 m_i - 1) * clip(l_i, -c, c), c ~ 4 (PS/TM), 2 (SS). Linéaire en l, bien mieux conditionnée que a pour la déconvolution (§5).
- Z NUL CALIBRÉ (nouveauté, gratuit) : le décodeur est biaisé (certains bits préférés, dépendant de la position et de l'échelle). On a 1020+510 messages de la même méthode : pour chaque fenêtre, calculer s_k pour K=64 messages ALÉATOIRES du jeu (mêmes logits, aucun forward en plus), z = (s - mean_k s_k) / std_k s_k. Sous H0 (pas de filigrane ici), la vraie marge suit la même loi que les nulles. Corrige le biais par échelle ET par position. Le seuil devient universel (z > 3) au lieu d'un seuil par méthode/échelle à régler.
- Variante binomiale : z_bin = (a - 0,5)/sqrt(0,25/n) valide seulement si les bits sont indépendants et non biaisés : faux en pratique, d'où le nul empirique.

### Multi-échelle et pas
- Carrées : tailles {64, 96, 128, 192, 256, 384, 512} px ; pas = taille/4 (chaque pixel couvert ~16 fois) ; alignées sur la grille 4 px (taille et pas multiples de 4). Nombre de fenêtres par échelle (pas = s/4, positions en 512) : 64: 29x29=841 ; 96: 18x18=324 ; 128: 13x13=169 ; 192: 7x7=49 (pas 48) ; 256: 5x5=25 ; 384: 3x3=9 ; 512: 1. Total ~1 420 / image.
- Anisotrope (si Exp. 0 le confirme) : petit côté s in {64,128,256}, rapports r in {2,4,8,16} horizontaux et verticaux, pas s/2 : ~+1 500 fenêtres. Ne pas le faire pour TrustMark (r<=2 : ajouter r=2 seulement).
- Résolution spatiale utile estimée [HYP] : PixelSeal fenêtre min ~96-128 px (256 bits donnent du signal même en petit), évidence localisable à ~32-64 px après déconvolution ; TrustMark fenêtre min ~160 px (100 bits, ResNet à 245), localisation ~64 px ; StegaStamp fenêtre min ~128 px, localisation ~64 px (flatten + STN sensibles au cadrage : les petites fenêtres dérivent). Si une fenêtre minimale z-test donne AUC < 0,6 sur le train, supprimer cette échelle (mesurer par échelle l'AUC « fenêtre dans le masque (>=90 % wm) vs fenêtre propre (>=90 % GT=0) »).
- Modes d'échec : objets fins / visages (< ~64 px) : sous la résolution, à prédire par a priori (bords de l'édition, §8) ; fenêtres mixtes (frontière) : marge intermédiaire, d'où la déconvolution ; aplats (PixelSeal JND) : z ~ 0 des deux côtés ; remplacement génératif qui conserve un peu de signal (diffusion locale conditionnée sur l'image) : faux positifs aux bords du trou.

## 4. Une seule passe : cartes de features et décomposition (très bon marché, grossier)
1. PixelSeal : contrib_p = sum_i (2m_i-1) * per_position_logits[i,p] (exact : la moyenne spatiale donne la marge). Z-normaliser avec les messages aléatoires comme en §3.
2. TrustMark : même chose sur layer4 (8x8) ; Grad-CAM sur layer3 (16x16) : garder le signe (pas de ReLU) : contrib = sum_c (mean_hw dM/dA_c) * A_c, avec M = sum_i (2m_i-1) logit_i.
3. Super-résolution par décalages : décoder l'image décalée de (dx,dy) in {0,16,32,48}^2 (rognage/re-padding réfléchi) et recoller les cartes : 16 passes -> carte 4x plus fine. Attention aux effets de bord de padding. Ne PAS utiliser de flips/rotations (non invariants pour ces décodeurs, détruisent le motif).
4. Toutes ces cartes se mettent en cache et alimentent le réseau d'appui (stage C) comme canaux supplémentaires.

## 5. Agrégation vers 128x128 : tomographie linéaire plutôt que moyenne
Moyenne simple (chaque cellule = moyenne des z des fenêtres qui la couvrent) floute et pénalise les petits objets. Mieux : la marge douce est approximativement ADDITIVE en surface pour les décodeurs à pooling (PS, TM) : s_W ~ sum_c A_Wc e_c (e_c = évidence de la cellule c, A_Wc = fraction de W couvrant c). Problème inverse creux :
  min_{e>=?} sum_W omega_W (s_W - A e)^2 + lambda_TV * TV(e) + lambda_0 * |e - e_prior|^2
- A est un opérateur « boîte » par échelle : se calcule par avg_pool2d / convolution sur la grille 128x128 (fenêtre s -> s/4 cellules), pas de matrice explicite. Résolution par ~200 itérations de gradient projeté ou Adam sur GPU (quelques ms/image).
- Poids omega_W = 1/var nulle (déjà dans le z) x confiance de texture.
- Sortie e(128x128) en unités de z-par-cellule : canal principal pour seuillage ou réseau d'appui.
- Sinon (repli simple, 1 h) : e = somme sur échelles de (carte de z moyennée) avec poids appris par régression logistique par échelle.
- Pour StegaStamp (saturé, position-dépendant) la linéarité est moins bonne : utiliser a dur et deconv. avec lien logistique, ou laisser le réseau d'appui faire la fusion.

## 6. Saillance par gradient (PS et TM seulement, sauf conversion de StegaStamp)
- Perte demandée : BCE(logits, m). Piège : si le message est déjà bien décodé, le gradient BCE sature (sigma'(.)->0) et la carte devient bruit numérique. Préférer la MARGE linéaire M = sum_i (2m_i-1) l_i (ou BCE sur logits/T, T=4) : gradient non saturé.
- Mais |dM/dx| seul mesure la sensibilité du décodeur (bords, textures), pas la présence du filigrane. Variantes par ordre de valeur [HYP] :
  a. Intégrales de gradients avec référence floue : x0 = blur_gauss(x, sigma=2-3) (retire le haut-fréquence où est le motif, garde le contenu). Attr(x)_p = (x-x0)_p * mean_{alpha in 8 pas} dM/dx(x0+alpha(x-x0)). Complétude : somme = M(x) - M(x0), donc auto-calibrée (la marge « gagnée » grâce au motif). Positive là où le haut-fréquence ALIGNE avec le message ; moyenne nulle ailleurs. Résolution pixel mais bruité : moyenner sur cellules >= 16x16 (voire pyramide 8/16/32). Coût 8-16 passes avant+arrière par image : négligeable.
  b. Occlusion par flou (§7).
  c. Grad-CAM couche 3 (TM) / stade 3 (PS) : grossier, stable.
  d. SmoothGrad (bruit sigma ~ 2/255, 8 échantillons) pour réduire le bruit.
- Mesurer l'AUC pixelique (cellule 4x4) sur 50 images train par méthode ; garder si AUC > 0,7 ou gain en entrée du réseau.
- StegaStamp : conversion onnx2torch (STN avec grid_sample est supporté ; risque de nœuds exotiques) sinon seulement §3 et §7.

## 7. Occlusion par flou et optimisation de masque (évidence « causale », échelle native conservée)
- Occlusion : image entière décodée à l'échelle habituelle, mais une fenêtre W est remplacée par sa version floue (sigma 2-3 px) : Delta_W = M(x) - M(x with W blurred). Delta > 0 si W contient du motif. Avantage sur le crop : le décodeur voit l'image dans son cadrage habituel (indispensable pour StegaStamp à flatten/STN ; aucun mismatch d'échelle). Inconvénient : signal d'une petite fenêtre sur la marge globale très faible (arrive à z ~ 1) ; ne marche bien que fenêtres >= 96 px, pas = s/2. Coût : (512/48)^2 ~ 100-150 fenêtres/échelle x 3 échelles ~ 400 passes/image.
- Optimisation de masque (PS/TM, éventuellement SS convertie) : paramétrer un masque doux M in [0,1]^{128x128} (sigmoïde, init 0,5) et minimiser  -margin_normée(x*(1-up(M)) + blur(x)*up(M))... ATTENTION au sens : on cherche la plus petite région dont le floutage détruit le message : min_M  [clip(margin(x_M)) pour atteindre 0] + lambda_1 |M|_1 + lambda_TV TV(M). Résultat : masque pixel-résolu. 150-300 itérations, ~3-5 s/image/méthode, 510 images/méthode -> 30-40 min. Risque : solutions adversariales (M qui détruit la marge via artefacts) ; parade : flou gaussien réel (pas de bruit appris), clamp M, et décodeur robuste au flou (en partie résistant à lui-même). À timeboxer 2 h sur 20 images.

## 8. A priori de forme et aspect (plus gros gain probable par image)
- Rectangle de somme maximale : sur e(128x128) (z par cellule, moins un seuil tau), trouver le sous-rectangle maximisant sum(e - tau) : Kadane 2D, O(128^3) = 2 M opérations (< 50 ms en numpy/torch). Ajouter la contrainte d'aspect : TM <= 2:1, PS/SS <= 16:1 (rejeter ou pénaliser les rectangles hors limites) et un a priori de taille/position tiré des histogrammes train (pénalité log-vraisemblance).
- Puis « sculpter » : dans le rectangle, retirer les cellules avec e < -tau2 (trous d'édition) par composantes connexes de surface >= 4 cellules ; hors rectangle, ne rien ajouter (sauf transplantation : l'objet n'est pas un rectangle).
- Sélecteur par image : features (fraction wm, ajustement du rectangle, rapport de e dans/hors rect.) -> classifieur logistique (3 classes : rect pur, rect+trous, forme libre) appris en validation croisée sur 510 images ; applique le post-traitement correspondant.
- Pour TrustMark : le patch entier <= 2:1 est filigrané [KIT] : rectangle presque toujours correct (sauf trous).
- Recherche de rectangle natif (si Exp. 0 positif) : pour chaque rectangle candidat R (grille 32 px, aspect autorisé, ~5-10 k candidats après filtrage), score = z du crop R redimensionné carré ; 1 300 passes au pas 64 -> raffinement pas 16 sur le top-20. Score de rectangle par maximum de vraisemblance avec pénalité d'aire pour éviter le biais « sous-rectangle ».

## 9. Calibrage par méthode (liste)
1. z nul par messages aléatoires (§3) : corrige biais d'échelle, de position, de méthode.
2. Par échelle : régression logistique de P(cellule wm) sur [z_s(cell), e(cell), confiance texture] ; sauvegarde de coefficients par méthode (3 jeux). Validation croisée en 5 plis stratifiés par méthode (170 images/méthode seulement : petits modèles).
3. Calibrage par cellule/position : tableau 128x128 de la moyenne et variance du z sur les cellules GT=0 du train (biais de bord/centre, effet de padding) ; soustraire. Effectif faible (170) : lisser (gaussien sigma 8 cellules) ou n'utiliser que 8x8 blocs.
4. Seuil d'IoU espérée : balayer tau in [0,2 ; 0,8] sur la probabilité calibrée, choisir le tau qui maximise le mIoU hors-pli ; puis règle « snap » (§0). Les seuils réglés sur le public (30 %) sont à éviter : tout se règle sur le train OOF.
5. TTA : décalages (pas de flips). Moyenne des probabilités.
6. Garde « pas de signal » : si max_W z < 2 (PS) / 2 (TM/SS) : aucune évidence => sortie = a priori de méthode (tout-0 ou tout-1 selon ce qui rapporte le plus sur le train pour ces cas).

## 10. Coût de calcul (estimations non mesurées ; seule la méthode de l'image est décodée)
Chaque image ne passe que dans SON extracteur : 510 images/méthode (170 train + 340 test).
- Fenêtres carrées (~1 420) : 510 x 1 420 = 724 k passes/méthode. Débits A100 estimés en fp16/channels_last [HYP] : PixelSeal ConvNeXt-T à 256 px ~6 GFLOP/passe : 1 500-3 000 img/s -> 4-8 min ; TrustMark ResNet-50 à 245 px ~5 GFLOP : 2 000-3 000 img/s -> 4-6 min ; StegaStamp ONNX 400 px, bs 32, CUDA EP, convs + dense 21 632x512 : 400-1 000 img/s -> 12-30 min. Fenêtres anisotropes : x2. Total tout confondu : < 1 h A100 pour les trois méthodes (1 GPU chacun en parallèle).
- Décalages 4x4 sur l'image entière : 16 passes x 510 : négligeable.
- Occlusion floue : ~400 passes/image : ~200 k passes/méthode : 2-5 min (PS/TM), ~10 min (SS).
- IG 16 pas : 510 x 48 passes équivalentes : < 1 min.
- Optimisation de masque : 30-40 min/méthode (PS/TM).
- Entraînement du réseau d'appui (stage C) : U-Net léger sur entrées 128x128 x ~12 canaux : minutes par pli sur 1 GPU.
- Si JURECA est saturé : réduire à tailles {96,128,192,256,384} et pas s/3 ; le plan tient sur 1 GPU en < 2 h. Ne pas oublier que la partie lourde est le redimensionnement : faire les crops en lot GPU avec `torch.nn.functional.interpolate` par taille de fenêtre (même taille = même tenseur), pas en PIL.
- Images test chargées une fois ; cache des logits (pas des images).

## 11. Structure de code concrète
```
wmloc2/                       # à côté de wmloc/ du kit, ne pas modifier le kit
  windows.py     # make_grid(sizes, strides, H=512) -> list[(s,y0,x0,h,w)] ; crops_batch(img, boxes, res) -> tensor
  logits_cache.py# pour (method, split): boucle images -> logits [n_windows, nbits] float16 -> cache/{method}/{split}/win_logits.pt
  nullcal.py     # z nul : load msgs, K=64 aléatoires -> z[n_windows] ; sauvegarde z.pt
  tomography.py  # opérateur boîte par échelle, solveur NNLS/TV -> e[128,128] par image ; cache/{method}/{split}/e.pt
  saliency.py    # IG / GradCAM / occlusion -> maps_{name}.pt [N,128,128] float16
  mask_opt.py    # optimisation de masque (PS/TM)
  prior.py       # histogrammes boîtes train, Kadane 2D, sculpture, sélecteur
  features.py    # empile canaux : z_s (par échelle), e, IG, occ, per_position_logits upsamplé, confiance JND, image 128 -> X [N,C,128,128]
  fuse.py        # U-Net léger (stage C) ; entraînement CV 5 plis par méthode
  predict.py     # chaîne -> masques 128x128 bool ; validations de format
  eval_oof.py    # mIoU hors-pli + par type d'édition (si dérivable)
```
Caches (.pt, tous float16 sauf indication) : win_logits (510 x 1 420 x nbits x 2 o = ~180 Mo pour 256 bits, ~75 Mo pour 100), z (510 x 1 420), e (510x128x128 = 17 Mo), maps saillance (17 Mo chacune), X (510 x C x 128 x 128 x 2 o = C x 17 Mo). Le tout < 2 Go : stockable en $SCRATCH et partageable. Indexation par `image_name` (dict) pour éviter les erreurs d'ordre. Fixer les graines et un `manifest.json` (version du kit, commit videoseal b1bf4c0 [KIT]).
Fiabilité de soumission : `predict.py` doit vérifier avant écriture : 1020 noms uniques = ceux de metadata.json test, shape [1020,128,128], dtype bool/uint8, valeurs {0,1}, taille < 20 Mo, rechargement par `torch.load(weights_only=True)`, et mIoU local de la MÊME fonction sur train OOF. Garder en secours le fichier « baseline all-0/all-1 par méthode ».

## 12. Astuces à fort gain, classées par risque croissant (gains = estimations [HYP])
Faible risque :
1. Seuil d'IoU espérée + règle « snap » tout-0/tout-1 (§0, §9.4) : +0,02 à +0,06 ; indispensable à cause de l'IoU=0 sur classe vide mal prédite.
2. Z nul par messages aléatoires (§3) : fiabilise tous les seuils ; +0,01 à +0,03, quasi gratuit.
3. A priori de méthode : TM rectangle (<= 2:1), contraintes d'aspect PS/SS (<= 16:1) pour rejeter les formes aberrantes : +0,01 à +0,03.
4. Garde « pas de signal » (§9.6) vers la valeur par défaut optimale du train.
Risque moyen :
5. Rectangle de somme maximale + sculpture de trous (§8), sélectionné par image : +0,05 à +0,10 si une grande part des masques sont des rectangles (outpainting) ; négatif sur transplantation d'où le sélecteur.
6. Tomographie linéaire (§5) au lieu de la moyenne : +0,02 à +0,05 sur les trous moyens.
7. Calibrage par cellule/position (§9.3) : +0,01 à +0,03 (effectif train limité).
8. Occlusion par flou (§7), surtout pour StegaStamp : +0,02 à +0,05.
9. Réseau d'appui qui fusionne e, z multi-échelle, saillance, confiance de texture : +0,05 à +0,10 par rapport à un seuillage (chiffre = estimation ; voir état de l'art du coéquipier).
Risque élevé :
10. Recherche de rectangle natif par crop aux échelle/position exactes (§1, §8) : potentiellement +0,10 si l'hypothèse d'échelle est vraie (très forte détection d'alignement), 0 sinon. Décidé par l'Expérience 0.
11. Optimisation de masque (§7) et IG pixéliques (§6) : gains de finesse pour objets fins/visages, 0 à +0,05 ; fragiles (bruit, pièges adversariaux).
12. Corrélation de résidu avec E(blur(x), m) - blur(x) (embedder déterministe) : résolution pixel possible, surtout pour StegaStamp/TrustMark ; variable ; timebox 2 h sur 20 images (déjà décrit comme D dans le rapport du coéquipier).
13. Pseudo-étiquetage des 1 020 images de test (transductif) : entraîner le réseau d'appui sur les masques confiants du test ; peut renforcer les erreurs ; ne l'utiliser qu'avec contrôle par OOF et à la fin.

## 13. Plan d'exécution proposé (heures indicatives, 1-2 personnes)
- H+0-1 : histogrammes train (§0), baselines tout-0/tout-1/par méthode ; Exp. 0 (crop GT bbox vs image entière) ; inspection du graphe ONNX et des formes `features()`.
- H+1-3 : fenêtres carrées + cache de logits + z nul ; AUC par échelle ; premier mIoU OOF avec moyenne et seuil + snap -> première soumission sûre.
- H+3-5 : tomographie, rectangle Kadane + sculpture + sélecteur ; occlusion floue.
- H+5-9 : IG/Grad-CAM, features.py, réseau d'appui par méthode en CV ; ajouter a priori.
- H+9-12 : recherche de rectangle natif (si Exp. 0 l'a validée), mask_opt, assemblage final, soumission de secours validée avant la fin (marge de 2 h avant 12:00 dimanche).

## Points non vérifiés (à ne pas croire sans test)
- Étirement du motif sur le rectangle et anisotropie (§1) : déduit du code StegaStamp du kit et de ma mémoire de TrustMark/videoseal.
- Détails du pixel_decoder (taille de la carte), du STN de StegaStamp, des noms de stades ConvNeXt ; exposants de débit A100.
- 0,58 à aire 0,32 pour PixelSeal : repris du rapport du coéquipier.
- Comportement des pipelines d'édition (recollage de l'original hors masque ou non).

## Sources
- Kit : /Users/titouandestors/Hack European Championship/EAC/reports/sources/t1/starter_kit/wmloc/watermarks.py, utils.py, data.py, download_models.py ; énoncé reports/sources/Watermark_Localization.txt.
- PixelSeal : https://arxiv.org/abs/2512.16874 (résumé seulement) ; carte https://raw.githubusercontent.com/facebookresearch/videoseal/main/videoseal/cards/pixelseal.yaml (résumé du fetch, pas verbatim) ; code https://github.com/facebookresearch/videoseal
- TrustMark : https://arxiv.org/abs/2311.18297 (résumé seulement) ; https://github.com/adobe/trustmark
- StegaStamp : https://arxiv.org/abs/1904.05343 ; ONNX WAVES https://github.com/umd-huang-lab/WAVES
- Rapport coéquipier : /Users/titouandestors/Hack European Championship/EAC/reports/t1-watermark-etat-de-lart.md
