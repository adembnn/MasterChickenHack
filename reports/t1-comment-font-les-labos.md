# T1 : comment les labos répondent à nos 4 questions, et quoi tester ce soir
Sam. 21/11/2026. Débloque : lesquelles des 4 idées deviennent les prochaines expériences, et avec quelle recette exacte.
[LU] = lu dans le papier ou le code (lien donné). [IDÉE] = notre adaptation, jamais mesurée. Aucun chiffre de mIoU ici n'est mesuré ;
les AUC/seuils des « tests de 30 min » sont des critères go/no-go à lire, pas des promesses. Rien lancé : ni GPU, ni API, ni soumission.
Vocabulaire (une fois) : **logit** = score brut d'un bit avant seuil ; **marge** = moyenne de (±1 selon le bit du message) x logit ;
**résidu** = ce que l'embedder ajoute à l'image (image filigranée moins image propre) ; **gabarit** (template) = le résidu qu'on
s'attend à voir pour notre message ; **filtre adapté** (matched filter) = mesurer la ressemblance entre l'image et le gabarit, c'est le
détecteur optimal d'un signal connu noyé dans du bruit ; **z-score** = écart à la moyenne du hasard, en écarts-types ; **AUC** = chance
qu'une case filigranée ait un score plus haut qu'une case propre (0,5 = hasard, 1 = parfait) ; **cellule** = case de 4x4 px du masque 128x128.
## 0. Ce que les labos font vraiment (la réponse courte)
Personne n'a publié la localisation post-hoc de PixelSeal, TrustMark ou StegaStamp tels quels (confirme le rapport état de l'art du 21/11).
Les labos ont deux réflexes :
1. **Entraîner le localisateur avec le filigrane** : WAM (Meta, ICLR 2025), MaskWM (« Mask Image Watermarking », arXiv 2504.12739 : le
   rapport précédent l'appelait MaskMark ; U²-Net, perte masque MSE), ReliaGuard (nov. 2026). Tous : masques rectangle / irrégulier / objet COCO
   collés pendant l'entraînement. [LU : WAM §4.2 ; MaskWM §3.2.] C'est notre U-Net sur données synthétiques, donc déjà la bonne voie.
2. **Tester avec des statistiques exactes** (Meta/Furon, DeepMind) : seuils calibrés sur le hasard, jamais « bit accuracy > 0,9 » à l'œil.
Ce qu'ils ne font PAS : réutiliser l'embedder comme filtre adapté pour localiser. Le gain possible est donc à nous, et le risque aussi.
Deux faits du code qui changent le plan : (a) la tête « pixelwise » de videoseal est juste `nn.Conv2d(C, nbits+1, 1)` à la place de
`nn.Linear` [LU videoseal/modules/pixel_decoder.py] : les poids de PixelSeal se recopient dedans, c'est déjà ce que fait `per_position_logits`
du kit. (b) L'article PixelSeal ne parle d'aucun masque ni de canal 0 ; l'entraînement n'a que recadrage, JPEG, etc. [LU arXiv 2512.16874
§3.2, §5.1]. Le canal 0 « détection » est donc probablement non entraîné : ne perdez pas de temps dessus (test : 5 min, voir §1).
## Q1. « On possède les embedders »
### (a) Ce que font les labos
- **Meta, wmforger** (Souček, Fernandez et al., NeurIPS 2025 spotlight, arXiv 2510.20468 ; code videoseal/wmforger) [LU §3.1-3.2] : un
  ConvNeXt-V2-Tiny entraîné sur SA-1B à dire laquelle de deux images est « la vraie » (perte de classement Bradley-Terry ; la fausse = image
  + artefact de Fourier synthétique, aucun filigrane dans l'entraînement). Il sort UN score par image. On estime le filigrane par montée de gradient :
  ŵ = argmax_δ R(x_w − δ), 50 pas, 768x768, ~6 s/image ; propre = x_w − ŵ. Sans message ni décodeur ; transfert mesuré : TrustMark et
  VideoSeal oui (retrait 0,60 et 0,49 de bit accuracy). Poids : `dl.fbaipublicfiles.com/wmforger/convnext_pref_model.pth`. Aucune carte de localisation
  dans le papier, seulement le résidu ŵ. C'est le seul estimateur de résidu « universel » public.
- **Adobe, ReMark** (ICCV 2025, Bui, Agarwal, Collomosse ; pip trustmark) [LU code 0.8.0] : `tm.removal` est un réseau de débruitage (KBNet) à
  256x256 ; `remove_watermark` fait `res = removal(x256) - x256` puis ré-agrandit `res` en bilinéaire. Il existe par variante (rm_C/Q/B/P) :
  il n'est entraîné que pour le résidu TrustMark. Le papier dit qu'un débruiteur dédié est nécessaire pour restaurer à >48 dB.
- **Stéganalyse classique** : le « WS residual » de Ker (ACM MM&Sec 2008) estime la couverture par filtrage, prend (stego − filtré) comme
  résidu et en déduit les positions de charge utile [LU diapos de Ker], mais seulement en mettant en commun des centaines d'images avec la même
  clé et les mêmes positions : pas applicable image par image. Fridrich/SRNet : un réseau apprend lui-même les résidus ; la version
  « selection-channel-aware » (Denemark-Boroumand-Fridrich 2016, de mémoire, non rouvert) donne au réseau la carte des endroits où l'embedder
  veut écrire. Équivalent chez nous : donner au réseau le gabarit.
- **Furon** (cs/0606034, TIFS 2007 [LU]) : le détecteur optimal d'un signal faible connu est un test de Neyman-Pearson ; c'est le filtre adapté.
### (b) Recettes sur l'API du kit
**R1. Filtre adapté avec le gabarit de l'embedder** [IDÉE ; c'est la proposition forte].
Idée : `embed(x_w, m) - x_w` ressemble à ce que l'embedder a déjà ajouté, car le U-Net de l'embedder dépend du contenu, qui n'a presque pas
changé. Là où le filigrane est présent, l'image contient un résidu corrélé à ce gabarit ; ailleurs, non. Pas à pas (x = image 512 en float) :
1. Gabarit exact, SANS quantification à 8 bits : TrustMark `wm.tm.encoder(cover256*2-1, secret)` (comme dans `encode`, retirer la moyenne par canal) ;
   PixelSeal `wm.model.embed(x, msgs=m, is_video=False)["imgs_w"] - x` ; StegaStamp `_run(...,["residual"])` puis ré-agrandi comme dans `embed`.
   Refaire pour K=8 messages au hasard m' : K gabarits « de contrôle ».
2. Haute fréquence de l'image : h = x − flou gaussien(σ=1,5). PixelSeal écrit dans la luminance seule (config `unet_base_yuv_quant`, `in_channels: 1`
   [LU pixelseal.yaml]) : corréler sur Y. TrustMark sur RGB. StegaStamp : résidu fort et basse fréquence, prendre σ=3.
3. Pour chaque m (vrai et K faux) : S_m(p) = lissage gaussien (σ=12 px) de <h, r_m>, et E(p) = lissage de <r_m, r_m>.
4. Carte de gain g(p) = S_m(p) / E(p) : c'est « quelle fraction du gabarit est présente ici ». ~1 en zone filigranée, ~0 en zone propre, donc
   **le seuil 0,5 n'a pas besoin d'être réglé par image** (avantage sur un z-score). Contrôle : g_m'(p) doit rester ~0 ; z(p) = (S_m − moyenne_K(S_m')) / écart-type_K(S_m').
5. Cadre : exact pour la moitié « image entière filigranée » (gabarit calculé sur les 512 px, même cadre que l'embedder, résidu ré-agrandi x2).
   Pour la moitié « rectangle » le motif est calé sur le rectangle (échelle s/256) : il faut d'abord la boîte (stage 1 : U-Net ou détecteur de boîte TrustMark), puis
   recalculer le gabarit sur ce rognage. Sans boîte, ne comparer qu'à l'échelle du cadre 512.
Contrôle de bon sens, 5 min, sur données synthétiques (où x0 est connu) : cos( embed(x_w,m)−x_w , x_w−x0 ) doit dépasser ~0,8. Sinon, R1 est morte.
**R2. Résidu estimé sans message** [IDÉE sur briques LU]. TrustMark : ŵ = −(`tm.removal(x256*2-1)` − x256) ; carte = |ŵ| lissée (16 px). Dans le cadre de la
boîte vraie ou de l'image entière 256. PixelSeal/StegaStamp : wmforger `optimize()` (6 s x 1530 images = 2,5 h-GPU, divisible par 4) donne ŵ ;
la carte est |ŵ| lissée ou, mieux, g(p) de R1 avec ŵ au lieu de h (ŵ est bien plus propre que le flou gaussien). Variante gratuite : h (R1) seul.
**R3. Ré-embedder « là où l'image change le moins »** [IDÉE, probablement faible]. L'embedder est additif, pas idempotent : il ajoute son résidu partout, filigrané
ou non, et son amplitude suit la texture (JND), pas la présence du filigrane. Test en 3 lignes : ‖embed(x_w,m)−x_w‖ dedans vs dehors. Attendu : indiscernable.
**R4. Inversion différentiable** (masque M tel que M·embed(x)+(1−M)·x ≈ image) [IDÉE, à écarter]. Sans x0, c'est sous-déterminé ; avec un débruiteur comme a priori on
retombe exactement sur R1/R2 (M ≈ g(p)), mais en 200 pas d'optimisation par image et un embedder ONNX non différentiable (StegaStamp). R1 est la version à forme close.
**R5. Tête pixelwise sur l'extracteur** [LU + IDÉE]. À 256 px d'entrée, ConvNeXt (pas 32) donne 8x8 ; 512 px donnerait 16x16 mais le motif y serait étiré x2 par rapport à
l'entraînement (le décodeur voit toujours le cadre 256). Donc : garder l'entrée 256 pour le décodeur, et décaler l'image de 8 px (16 décalages) avant réduction pour un vote 32x32.
C'est ce qu'apporte la tête du U-Net déjà en entraînement ; l'ajout utile est une perte de type WAM Eq. 5 (accord avec m seulement dans le masque) en perte auxiliaire.
### (c) Force attendue et pannes
- R1 : StegaStamp (résidu fort, ~30 dB) ≫ PixelSeal ≥ TrustMark (43 dB, faible, dans le bruit des zones texturées). [IDÉE] ordre de grandeur : z ≈ (amplitude/écart-type HF de l'image) x
  racine(nombre de pixels de la fenêtre) ; fenêtre σ=12 ≈ 1 800 px, rapport 0,3 → z ≈ 10 en zone texturée. Pannes : zones lisses ou sombres (la JND met le gabarit à ~0 : aucune information,
  c'est l'équivalent « selection-channel » : calculer E(p) et le fournir comme canal de confiance) ; gabarit décalé si le cadre est faux (rectangle) ; éditions génératives (VAE) qui
  lissent le résidu même hors zone (banc 2609.16832).
- R2 TrustMark : bon si ReMark ne « retire » pas aussi du contenu en zone propre (inconnu : il n'a vu que des images filigranées) ; hors cadre 256 le motif a une mauvaise échelle (s/2 au lieu de 256 si
  rectangle de côté s sur image 512) donc ReMark s'efface → toujours après la boîte. PixelSeal/StegaStamp : wmforger transfère mal (0,49-0,60 de retrait mesuré sur autres filigranes) : carte bruitée.
### (d) Coût
R1 : 0,5 h de code pour le test, 3 h pour la version complète (3 embedders, K=8) ; GPU : ~5 gabarits x 20 ms = 0,1 s/image → 1 530 réelles en 3 min ; 20 000 synthétiques x 3 méthodes ≈ 1,5 h-GPU.
R2 : 1 h de code ; GPU 2,5 h-GPU pour wmforger sur 1 530 images (pas sur 20 k synthétiques : trop long, utiliser h à la place).
### (e) Test de 30 min (go/no-go)
Sur 30 images train par méthode (10 par population fullwm / rectangle, boîte vraie pour les rectangles) : AUC de g(p) ou z(p) contre le masque, cellules 32 px.
Aller si AUC ≥ 0,85 sur fullwm ET ≥ 0,80 sur rectangle avec boîte vraie. Comparer à la CAM (0,35 mIoU ≈ AUC ~0,65) et lire l'écart avec les gabarits de contrôle (g_m' ≈ 0). 
Si 0,65-0,80 : n'utiliser que comme canal d'entrée du U-Net. Sous 0,65 : abandonner. Même passage : AUC de |canal 0| de PixelSeal (attendu ~0,5).
## Q2. « On connaît le message exact »
### (a) Ce que font les labos
- **Meta** : test binomial sur le nombre de bits qui concordent, FPR exact = 2^-k Σ C(k,i) [LU Stable Signature arXiv 2303.15435 §3.1, Éq. 1] ; borne de faux positifs de même type dans Pixel Seal [LU §3.1, Éq. 2].
  WAM : un pixel est filigrané si son score dépasse τ, réglé par FPR mesuré sur tous les pixels d'un jeu propre ; score image = proportion de pixels au-dessus de τ [LU WAM §3, Éq. 1].
  DBSCAN de WAM (ε=1, 1 000 px min.) regroupe les pixels par message décodé quand il y en a plusieurs, et prend des centres valides [LU §3, §5.5] : chez nous le message est unique et connu, inutile.
- **DeepMind** : SynthID-Image calibre par p-values conformes sur jeux de calibration (filigranés / non) et teste les bits avec Simes/Holm/Hochberg [LU arXiv 2510.09263 §7] ;
  AUCUNE localisation publiée (le papier n'en parle pas) ; seul le blog du SynthID Detector (20/05/2025) dit que le portail « met en évidence les zones où le filigrane est le plus probable », méthode non décrite [LU].
  SynthID-Text : score moyen / bayésien par jeton, dans Nature 2024 (de mémoire, non rouvert).
- **Furon/classique** : Neyman-Pearson, détecteurs de corrélation à étalement de spectre : la statistique optimale est une corrélation pondérée par la fiabilité de chaque composante.
### (b) Recette
1. **z-score exact sans tirage** [IDÉE, forme classique] : pour une image fixe, le message vrai m est indépendant du contenu s'il n'y a pas de filigrane ; donc s_i = (2m_i−1)·l_i a des signes
   aléatoires indépendants. Sous H0 : moyenne 0, variance Σ l_i². Donc **z = Σ_i (2m_i−1)·l_i / sqrt(Σ_i l_i²)** (test par inversion de signes). Pas de 64 messages faux ni de seuil par image :
   remplace la calibration par K messages du rapport critique (garde-la comme contrôle). Bits indépendants sous H0 : à vérifier sur les logits de messages au hasard.
2. **Pondération par fiabilité** [IDÉE ; classique] : z_w = Σ w_i (2m_i−1) l_i / sqrt(Σ w_i² l_i²), w_i = μ_i/σ_i² estimés sur les 170 images/méthode (μ_i = marge moyenne du bit i en zone filigranée).
   Utile si certains bits sont systématiquement faibles (TrustMark et StegaStamp n'ont pas de code correcteur ici).
3. **Par fenêtre** : même z sur chaque rognage candidat (boîte), pas sur une case de 64 px (le décodeur moyenne : signal /64, cf. critique tour 1 §1.4).
4. **Réseau conditionné par le message** [IDÉE] : donner au U-Net, en plus de l'image, 6 canaux = [h⊙r_m, h⊙r_m'] lissés à 16 px (R1), ou la carte g(p) et la confiance E(p).
   Le réseau n'a alors qu'à seuiller et à imposer la forme ; le critique l'avait mis en v2 : le test de R1 dit si on le monte en v1.
### (c) Force et pannes
Le z exact classe très bien l'image entière ou une boîte (le hasard donne z ~ N(0,1)), mais il ne localise pas : c'est un outil de décision, pas de carte. Gain direct sur l'IoU : faible
(décision « cette boîte est-elle filigranée ? », choix entre candidats de boîte, tri fullwm/rectangle). Panne : biais si les bits des logits sont corrélés sous H0 (z surestimé) ;
messages de test à 100/256 bits : seuil d'un faux positif à 1e-3 ≈ z=3.
### (d) Coût
Code 1 h, GPU nul (les logits existent déjà). Pondération : 0,5 h de plus.
### (e) Test de 30 min
Sur 170 images/méthode d'exp0 : tracer l'histogramme de z sur messages au hasard (doit être N(0,1) ; sinon l'hypothèse d'indépendance est fausse) ; AUC de z(boîte vraie) vs z(image entière) pour prédire
« fullwm ». Aller si l'AUC de z_full pour fullwm dépasse 0,9 (le tri des deux populations devient quasi gratuit) et si z_w bat z de ≥ 1 point de z moyen sur les cases de rectangle.
## Q3. « Les trous ont une forme d'objet »
### (a) Ce que font les labos
Ils utilisent les formes d'objets à l'ENTRAÎNEMENT, pas à l'inférence : WAM colle des masques « rectangle, irrégulier, inversé, plein, nul » et des masques de segmentation = union des objets COCO
(Fig. 6, §4.2, App. D.4) [LU] ; MaskWM tire 4 types de masques dont « Segment » depuis MS-COCO [LU §3.2]. Aucun papier lu ne fait de vote de présence de filigrane par segment SAM ; les usages de SAM en
criminalistique sont des réseaux appris (SAMIF, ACCV 2024 [LU]) ou du copier-coller par segments (ACM CTA 2024 [LU]). Ni CRF ni graph-cut dans WAM/MaskWM/ReliaGuard (rien lu).
wmforger entraîne sur SA-1B (images) [LU]. Donc : le vote par segments est une idée libre, non validée par les labos.
### (b) Recette (à n'essayer qu'après le test oracle)
1. Propositions : SAM2 « automatic mask generator » (poids publics SAM2.1 tiny/base, ou MobileSAM/FastSAM ; ~0,3-2 s/image) sur l'image 512, ou superpixels SLIC (skimage, CPU, 0 coût).
2. Score par segment : moyenne de la probabilité du U-Net (ou de g(p)) sur le segment ; masque = union des segments de score > 0,5, plus la probabilité brute là où SAM n'a rien proposé.
3. Raffinement par les bords : filtre guidé (guided filter, He 2010) du masque 128x128 remonté à 512, guidé par l'image, puis réduit ; ou DenseCRF (pydensecrf, installation Windows pénible).
4. Forme : un trou = composante connexe de taille ≥ 4 cellules ; boucher les trous < 16 cellules (déjà prévu).
### (c) Force et pannes
Point dur : en moitié « tout filigrané puis une zone remise propre », le bord de la zone N'EST PAS un contour d'image (les pixels propres sont les originaux ; la différence n'est que le gabarit,
invisible en RGB). Les segments SAM n'épousent alors pas le blob irrégulier ; ils épousent les objets, qui sont justement les trous « objet » du cas partiel (retrait/ajout d'objet). Gain attendu
(IDÉE) : +0,02 à +0,05 sur les trous d'objet, ~0 sur les blobs. Risque : fusionner un segment de 60 % propre et 40 % filigrané.
### (d) Coût
Oracle : 1 h. Pipeline : 3 h + 1 h-GPU (SAM2 sur 1 530 images). 
### (e) Test de 30 min, sans aucun détecteur de filigrane
Sur les 510 masques vrais du train : calculer les segments SAM2 (ou SLIC 600 segments) puis, avec le masque VRAI, choisir pour chaque segment la classe majoritaire (« snap oracle »)
et mesurer le mIoU officiel à 128x128. Aller si ≥ 0,90 (alors les bords collent, le vote par segments vaut le coup) ; entre 0,80 et 0,90, essayer seulement SLIC ; < 0,80 (attendu pour les blobs),
laisser tomber et garder le U-Net. Donner le score séparé fullwm / rectangle.
## Q4. « Les zones propres ont été générées ou collées »
### (a) Ce que disent les papiers (public, poids disponibles)
- **TruFor** (CVPR 2023, grip-unina/TruFor, poids sur leur dépôt) : Noiseprint++ (empreinte caméra) + Segformer, sort une carte, un score et une fiabilité. F1 pixel CocoGlide (inpainting GLIDE) 0,523 [LU Table 1] ;
  CAT-Net v2 0,434, ManTra-Net 0,516, MVSS-Net 0,486 [LU, mêmes sources]. **CAT-Net** (WACV 2021, github mjkwon2021/CAT-Net), **MVSS-Net**, **PSCC-Net**, **IML-ViT** : poids publics.
- Précision réelle sur inpainting par diffusion, hors domaine : IoU moyen TruFor 36,9 % sur SD1.5/2.1/XL/SD3/Flux (63,4 sur SD1.5 mais 7,6 sur Flux) ; CAT-Net 37,4 % [LU FUSED arXiv 2608.28302].
  GIFTBench (2610.01778) : TruFor F1 0,199 sur « AI-Edit » et 0,235 sur « Removal » hors domaine [LU]. DiffusionPrint (CVPRW 2026) : Noiseprint++ « active sur des régions authentiques » en inpainting totalement régénéré.
- **Régions « restaurées à l'original »** : INP-X (arXiv 2602.00192) : remettre les pixels d'origine hors de la zone éditée fait tomber les détecteurs à ~55 % (hasard) ; les détecteurs exploitaient la signature globale du VAE,
  pas la zone synthétisée [LU]. Une zone restaurée est donc un pixel authentique : aucun détecteur ne la voit comme « falsifiée », et FUSED note la même chose.
### (b) Ce que ça implique pour nous
Dans la moitié « tout filigrané puis blob restauré », le blob est AUTHENTIQUE et le reste porte le gabarit : un détecteur de falsification verrait au mieux une discontinuité entre les deux, et signale d'ordinaire la région
minoritaire, c'est-à-dire le blob propre : la polarité est fausse (à vérifier). Dans la moitié « rectangle dans une image propre », les zones générées (extension/outpainting, objet ajouté, retrait) sont négatives :
un détecteur d'inpainting pourrait là réellement aider à dire « ceci n'est pas filigrané », à condition que la génération soit par diffusion. Recette : exécuter TruFor sur l'image 512 (carte anomalie + carte de fiabilité) et
ajouter les 2 cartes (réduites à 128x128) comme canaux du U-Net, ou régression logistique par cellule [carte U-Net, anomalie] sur les images rectangle.
### (c) Force et pannes
Seul : au mieux quelques points d'AUC sur la moitié rectangle, nuls sur la moitié restaurée. Risque : fuites hors domaine (SD3/Flux : IoU 8-32 %) ; coût de licence (recherche non commerciale, à lire).
### (d) Coût
Poids à télécharger (taille non vérifiée, quelques centaines de Mo ; téléchargement à valider avec vous) ; 1 h de code ; GPU ~0,3 s/image.
### (e) Test de 30 min
60 images train (20/méthode) : AUC pixel de la carte d'anomalie TruFor contre le masque NÉGATIF (zone propre), séparée fullwm/rectangle. Aller si AUC ≥ 0,70 sur rectangle ; si l'AUC sur fullwm est < 0,5,
c'est la polarité inversée ci-dessus : ne l'utiliser que sur les images que le tri (z de Q2) classe « rectangle ». Si le gain ajouté au U-Net (régression logistique avec/sans) est < +0,01 de mIoU : arrêt.
## 5. Les 5 expériences à lancer, classées (gain attendu / coût)
| # | Expérience (qui la débloque) | Coût | Mesure et seuil |
|---|---|---|---|
| 1 | **Filtre adapté g(p)/z(p) avec gabarit de l'embedder** (R1 ; Q1+Q2). Test AUC sur 30 images/méthode ; si go, ajouter [x, h⊙r_m, h⊙r_m', E] comme canaux au U-Net en cours. | 0,5 h code + 3 min GPU ; version pleine 3 h + 1,5 h-GPU | AUC ≥ 0,85 fullwm / ≥ 0,80 rectangle (boîte vraie) ; puis mIoU U-Net avec vs sans canaux, par population |
| 2 | **Résidu estimé** (R2) : ReMark `tm.removal` (TrustMark, dans la boîte vraie ET image entière), wmforger pour PS/SS sur 20 images. | 1 h code, ~10 min GPU | AUC de |ŵ| lissée ≥ 0,80 ; test de fuite : |ŵ| sur pixels propres < 40 % de celle en zone filigranée |
| 3 | **Trouveur de cadre** : `TrustMark(loadBBoxDetector=True).localize()` (pip trustmark ≥ 0,9, dans un venv séparé : l'encodage est identique à 0,8.0, `diff` fait, mais ne pas changer l'environnement du kit) + z exact (Q2) pour décider fullwm vs rectangle. Débloque le stade 2 de 1 et 2. | 1 h | IoU de la boîte sur rectangles ≥ 0,70 ; AUC de z_full pour fullwm ≥ 0,90 |
| 4 | **Oracle SAM/SLIC** (Q3) sur les 510 masques vrais. Pas de détecteur nécessaire. | 1 h, 15 min GPU | mIoU « snap oracle » ≥ 0,90 : on monte le vote par segments ; sinon on écarte Q3 |
| 5 | **TruFor en canal d'appoint** (Q4) sur 60 images. | 1 h + téléchargement des poids (accord demandé) | AUC ≥ 0,70 sur rectangle ET gain U-Net ≥ +0,01 ; sinon on écarte Q4 |
Ordre de lecture ce soir : le 1 et le 4 en parallèle (aucune dépendance), le 2 ensuite, le 3 pour caler les cadres, le 5 en dernier. Ne pas coder R3 (ré-embedder) ni R4 (inversion) : couverts par R1 ; ne pas viser les cartes CAM/occlusion.
Limite de cette note : je n'ai lu ni le code complet de WAM ni ses poids ; TruFor, SAM2, wmforger et le détecteur de boîte TrustMark n'ont pas été exécutés ; Denemark 2016, SynthID-Text, He 2010 sont cités de mémoire.
Sources : arxiv.org/abs/2411.07231 (WAM) · arxiv.org/abs/2504.12739 (MaskWM) · arxiv.org/abs/2510.20468 + github.com/facebookresearch/videoseal/tree/main/wmforger · arxiv.org/abs/2512.16874 (Pixel Seal) ·
arxiv.org/abs/2303.15435 (Stable Signature) · arxiv.org/abs/2510.09263 (SynthID-Image) · blog.google/innovation-and-ai/products/google-synthid-ai-content-detector · github.com/adobe/trustmark ·
openaccess.thecvf.com/content/ICCV2025/papers/Bui_TrustMark_Robust_Watermarking_and_Watermark_Removal_for_Arbitrary_Resolution_Images_ICCV_2025_paper.pdf · arxiv.org/abs/cs/0606034 (Furon) ·
cs.ox.ac.uk/andrew.ker/docs/ker-acm08-slides.pdf (Ker) · github.com/grip-unina/TruFor · arxiv.org/abs/2608.28302 (FUSED) · arxiv.org/abs/2610.01778 (GIFTBench) · arxiv.org/abs/2602.00192 (INP-X) · arxiv.org/abs/2604.12443 (DiffusionPrint).
