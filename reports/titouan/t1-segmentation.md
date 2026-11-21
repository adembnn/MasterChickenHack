# T1 : segmentation apprise pour localiser le filigrane (conception + plan 18 h)

Décision débloquée : quelle architecture, quelles données synthétiques, quelle recette d'entraînement et quel plan horaire pour la partie « réseau de segmentation » de T1. Complète `reports/t1-watermark-etat-de-lart.md` (approche C), sans le répéter.
Légende : [LU] = lu cette session (URL) ; [MÉMOIRE] = connaissance de l'auteur, non revérifiée ici ; [HYP] = hypothèse/estimation jamais mesurée ; [KIT] = lu dans le starter kit local.
Limite : je n'ai lu que des résumés (arXiv) et le texte HTML de WAM. Je n'ai ni téléchargé de poids ni exécuté de code. Tous les débits (img/s) et heures GPU sont des ordres de grandeur [HYP].

## 0. Les 10 conclusions à retenir

1. Le meilleur point de départ n'est pas un U-Net ImageNet : c'est **l'extracteur lui-même transformé en segmenteur** à la WAM (backbone d'extracteur + décodeur de pixels, perte BCE sur le masque), avec **le message connu injecté sous forme de carte d'accord par position**. PixelSeal (ConvNeXt-V2-T) et TrustMark (ResNet-50) sont des torch.nn modules accessibles [KIT: watermarks.py, `features()` expose les couches]. StegaStamp n'est qu'un ONNX : pas de fine-tuning, donc un backbone générique + cartes pré-calculées.
2. **Le dispositif décisif pour TrustMark et StegaStamp : l'hypothèse de rectangle.** Leur décodeur redimensionne l'entrée à 245/400 px [KIT] et StegaStamp a un décodeur dense ancré en position (`gen_decoder/.../BiasAdd` est une couche dense, pas un pooling global) [KIT: docstring + nom du tenseur]. Une fenêtre glissante décalée ne retombe donc probablement pas sur le motif ; seul le décodage du crop exact du rectangle donne une bonne précision de bits [HYP, à tester en 30 min, voir §5.1]. D'où un pipeline en 2 temps : réseau (masque grossier + boîte) puis recherche de boîte vérifiée par `decode(crop)`, puis masque = boîte moins trous du réseau.
3. **Données synthétiques : recombiner, ne pas régénérer.** On précalcule une banque de (image propre, masque d'édition, contenu généré) et une banque de (crop filigrané, boîte, message). Le composite se fait à l'entraînement : `x = où(M_edit, généré, où(R, crop_filigrané, propre))`, étiquette = `R ET NON M_edit`. Les coûts lourds (diffusion, embedding) sont additifs et non multiplicatifs.
4. Perte : BCE + **Lovász-hinge par image sur les deux classes** + soft-IoU avec règle d'union vide + tête image « tout-1 / tout-0 / mixte ». Cible = `downsample_mask` officiel du 512 (règle des 50 %).
5. Pas d'augmentations par rééchantillonnage (rescale, rotation, flou, JPEG) : elles détruisent le signal filigrane. Seules : décalages multiples de 4 px, flips/rot90 (à valider), recombinaison synthétique, messages leurres.
6. 3 modèles (un par méthode), même code, même taille de grille ; poids initialisés depuis l'extracteur de la méthode quand c'est possible. Fine-tune réel en 5 folds (= validation croisée **et** ensemble en même temps).
7. Le bruit de mesure dépasse les petits gains : sur 306 images publiques, l'écart-type de la moyenne est environ 0,25/√306 ≈ 0,014 [HYP, pour un écart-type par image de 0,25] ; sur les 714 images privées ≈ 0,009. Ne pas arbitrer sur moins de 0,03 de classement public.
8. Budget calcul réaliste : environ 35 à 45 heures-GPU A100 au total [HYP], soit un nœud de 4 GPU pendant ~10 h, ou 1 GPU permanent de génération + 3 GPU d'entraînement.
9. Porte de décision à H8 : si le score OOF (sur train réel) < 0,60 on bascule en priorité sur « boîte vérifiée + baseline fenêtre » (§8).
10. Réserve la dernière heure à la fiabilité de la soumission (format, 1020 noms uniques, `weights_only=True`, bool 1020x128x128 = 16,7 Mo < 20 Mo).

## 1. Littérature : ce qui transfère

### 1.1 Localisation par filigrane natif
- **WAM** [LU: arxiv.org/html/2411.07231] : extracteur = encodeur ViT (~96 M de paramètres) + décodeur de pixels qui sort 1 + 32 valeurs par pixel (détection + bits) ; perte `1·BCE_pixel(masque) + 10·BCE_bits` (bits calculés seulement sur les pixels filigranés) ; COCO train 118 k, 256x256, masques : plein, rectangles, formes irrégulières, segmentations d'objets, inversés avec p=0,5 ; phase 1 : 300 époques, batch 16/GPU, 8 V100, ~2 jours ; phase 2 : 1-3 messages. Augmentations géométriques appliquées **aussi au masque**, et JPEG/flou/médian/couleur. Ils n'entraînent pas l'inpainting, ils l'évaluent avec LaMa. Leçon : la détection par pixel est une BCE simple ; la diversité des formes de masque est ce qui généralise. Nous n'avons que la détection à apprendre (message connu), donc bien moins de capacité nécessaire.
- **MaskWM / MaskMark** [LU: arxiv.org/abs/2504.12739 via recherche] : masque injecté en entraînement (variantes D et ED), module de localisation dédié, ~20 h sur un A6000 [d'après leur page NeurIPS] ; benchmark [LU: arxiv.org/abs/2609.16832, résumé seul] : l'inpainting/outpainting génératif peut anéantir la récupération du message, et récupération ≠ localisation. Conséquence directe : dans les zones inpaintées/outpaintées on ne peut pas compter sur la précision de bits, il faut des indices non-filigrane (texture générée, bords).
- **OmniGuard / EditGuard / TAG-WM** : second filigrane fragile pour détecter la manipulation [déjà cité par le rapport existant] ; non transférable (notre filigrane est imposé).

### 1.2 Localisation de falsification générique (image forensics) [MÉMOIRE sauf mention]
| Méthode | Entrée | Architecture | Ce qui transfère ici |
|---|---|---|---|
| MVSS-Net (ICCV 2021) | RGB + branche « bruit » (Bayar conv) + bords | FCN ResNet-50 à 2 branches, perte pixel + bords + tête d'image (GeM) | tête image-niveau (notre règle d'union vide), perte de bords |
| CAT-Net v2 | RGB + coefficients DCT JPEG | HRNet à 2 flux | **flux DCT inutile** : nos images sont des PNG, pas de grille JPEG |
| PSCC-Net | RGB | HRNet, supervision multi-échelle progressive, corrélation spatio-canal | supervision multi-échelle ; entraîné sur splicing synthétique |
| TruFor [LU: arxiv.org/abs/2212.10957, résumé] | RGB + empreinte de bruit apprise (Noiseprint++) | SegFormer fusionné, carte de fiabilité, entraînement auto-supervisé sur images réelles | fusion RGB + canal bruit ; carte de confiance utile pour décider « tout-1/tout-0 » |
| ManTra-Net | RGB + SRM + Bayar | CNN avec LSTM | filtres passe-haut fixes dès l'étage 1 |
| Approches SAM (ex. SAFIRE, MÉMOIRE) | RGB + prompts | SAM adapté | SAM sert chez nous surtout à **fabriquer les masques d'objets** et éventuellement à caler les frontières |

Résultat de recherche [LU: résumés trouvés via WebSearch, arxiv.org/html/2506.05466 (TGIF2), DiffusionPrint] : les localiseurs classiques gèrent le splicing mais échouent sur les images régénérées par diffusion (FLUX en particulier) ; le fine-tuning sur données de diffusion aide beaucoup ; les performances dépendent fortement du générateur utilisé. Donc : **les modèles forensics pré-entraînés ne transfèrent pas tels quels ; le signal filigrane est plus informatif que les traces de génération, et il faut fine-tuner sur du réel.** Aucun papier lu ne compare SRM contre RGB seul pour ce cas précis.

### 1.3 Représentations d'entrée : verdict [HYP argumenté]
1. **RGB pleine résolution (512), sans sous-échantillonnage précoce** : l'information utile est haute fréquence à ±1-3 niveaux de gris. Stem de type ConvNeXt (patch 4x4, pas 4) est acceptable ; ne jamais redimensionner l'entrée en dessous de 512 sauf pour reproduire exactement le prétraitement d'un extracteur (256/245/400).
2. **Carte d'accord au message** (le gros bénéfice) : pour PixelSeal/TrustMark, `a(p) = Σ_k s_k · z_k(p) / √n` où `s_k = 2m_k − 1` et `z_k(p)` = logit de bit k à la position p (`per_position_logits` du kit, 8x8 ; en reprenant le backbone à 512 px on obtient 16x16 puis un décodeur ramène à 128x128).
3. **Canal passe-haut fixe** (SRM 3 filtres ou Laplacien, sur 3 couleurs = 9 canaux) concaténé à RGB : peu coûteux, à tester en ablation (2 runs courts). Les fréquences explicites (FFT/DCT) : non prioritaire ; un CNN les apprend, et DCT-JPEG est sans objet.
4. **Corrélation de résidu** `HP(x) · (E(x,m) − x)` (idée 2.3.4 du rapport existant) : à ne tester que comme canal précalculé, timebox 2 h. Elle coûte un embed par image et exige que E soit appelé sur le composite.

## 2. Architecture proposée

Commun : sortie 128x128 (stride 4 par rapport à 512), logit par cellule. Décodeur type U-Net léger (4 niveaux, concat de skips, ~2-5 M de paramètres) + supervision profonde à strides 8 et 16 (cible = masque 128 réduit par moyenne puis seuil 0,5) + tête image-niveau (3 classes : tout-0, tout-1, mixte).

| Méthode | Backbone | Init | Entrée | Notes |
|---|---|---|---|---|
| PixelSeal | `det.convnext` (ConvNeXt-V2-T, 768 ch @ stride 32) + skips des stades 1-3 | poids de l'extracteur | RGB 512 (prétraitement natif de l'extracteur, sans resize 256) **ou** resize 256 puis upsample x2 : tester les deux | agrégation « mean pooling + linear » remplacée par la carte par position |
| TrustMark | ResNet-50 (`layer1..4`, strides 4-32) | poids de l'extracteur | RGB 512 puis variante 245 | échelle du motif liée à la taille du rectangle (voir §5.1) |
| StegaStamp | ConvNeXt-T ImageNet (timm, ex. `convnext_tiny.fb_in22k`) | ImageNet (à pré-télécharger) | RGB + SRM + cartes d'accord par fenêtre/boîte précalculées via ONNX | l'ONNX n'est pas fine-tunable (conversion ONNX→torch possible via onnx2torch, non vérifié) |

Option unifiée : un seul réseau ConvNeXt-T + FiLM de méthode, pré-entraîné sur les 3 synthétiques, puis fine-tune par méthode. À essayer **seulement si** les données réelles par méthode (170) s'avèrent trop pauvres pour 3 réseaux ; sinon 3 réseaux (les statistiques du signal diffèrent, et la tâche elle-même dit « pas de solution unique »).

Stade 2 « boîte » (TrustMark, StegaStamp d'abord, PixelSeal ensuite) :
- Proposition : bbox du masque prédit seuillé (composante principale), plus K=8 variantes jitter.
- Raffinement : descente de coordonnées sur (x0,y0,x1,y1) par pas de 16 puis 4 px, objectif = précision de bits de `decode(crop(boîte))` contre le message connu ; contrainte de ratio (≤2:1 TM, ≤16:1 PS/SS). ~100-200 décodages par image ⇒ 200 k décodages pour 1020 images, quelques minutes sur A100 [HYP].
- Masque final = boîte AND (probabilité du réseau > τ) ; si le réseau dit « forme libre » (transplantation, aucun rectangle plausible : écart boîte/masque > seuil appris en CV), garder le masque du réseau seul.

## 3. Génération de données synthétiques illimitées

### 3.1 Images propres sans Internet sur les nœuds de calcul
Les nœuds de calcul JURECA n'ont probablement pas d'Internet (usage courant des centres HPC : accès via les nœuds de login ou un proxy [LU: pages d'autres centres, rien de spécifique JURECA ; **à vérifier en 1 min**]). Le script d'équipe a déjà téléchargé des datasets HF depuis le login [KIT: hackathon_setup.sh], donc **télécharger tout sur le login dans `$HF_HOME` (dossier partagé `.cache`) avant 18:00**, puis `HF_HUB_OFFLINE=1` sur les nœuds.
Sources, par ordre de priorité :
1. **Les images du train/test eux-mêmes** (aucun téléchargement). Train : les cellules GT=0 sont propres (parfois génératives, donc réalistes). Test : régions non étiquetées, utilisables en pseudo-étiquetage (§6.5, voir risque). Sert aussi de repli si Internet manque. Problème : zones propres irrégulières et petites ; il faut un « remplissage » propre (utiliser les crops GT=0 les plus grands, ou LaMa).
2. **COCO val2017** (5 k images, ~1 Go) + annotations instances (241 Mo) : URLs classiques `images.cocodataset.org/zips/val2017.zip` et `annotations_trainval2017.zip` [MÉMOIRE]. Fournit les masques d'objets (transplantation, ajout, suppression). `pycocotools` est dans requirements.txt, mais c'est une dépendance de videoseal, ce n'est **pas** une preuve que le jeu du concours vient de COCO [HYP]. Ouvrir 20 images du train et regarder le domaine (la tâche cite « un visage » : possible jeu de visages type CelebA-HQ/FFHQ ; ajouter un jeu de visages au besoin).
3. Un jeu de scènes plus grand si nécessaire (ex. une tranche de `ILSVRC/imagenet-1k` ou LAION, accès HF possiblement fermé : non vérifié) : 5 k bases + recombinaison suffisent, ne pas y passer de temps.
Prétraitement : redimensionner/rogner à 512x512 **une fois** avec antialias (les vraies images sont 512x512), enregistrer en PNG sans perte.

### 3.2 Reproduire le pipeline des organisateurs (à confirmer sur le train)
Hypothèses sur le dataset [HYP, à vérifier en 1 h sur les 510 images] :
- (a) Rectangle R tiré avec ratio dans [1/16,16] (PS/SS) ou [1/2,2] (TM), crop de l'image, **embedding sur le crop à sa taille native** (c'est ce que fait `wm.embed(img_any_size)` [KIT]), recollé. TrustMark ne filigrane que le patch entier si ratio ≤ 2 [KIT].
- (b) `full_watermark=true` [KIT] : toute l'image filigranée puis une zone restaurée depuis l'original (transplantation : objet filigrané collé sur une image propre).
- (c) Les éditions génératives ont probablement un composite final « pixels originaux hors masque » ; sinon l'aller-retour VAE d'un pipeline diffusion abîmerait le filigrane partout. **Test : précision de bits de `decode(crop(boîte_GT))` doit être élevée ; si elle est mauvaise même hors masque, le composite n'est pas exact, et il faut simuler un aller-retour VAE dans la synthèse.**
- (d) Outpainting : le 512 final est probablement une image plus grande réduite ou rognée : **le motif filigrane est alors rééchelonné** ; mesurer sur le train si la boîte GT donne une meilleure précision de bits à l'échelle native ou à une échelle ≠ 1.

### 3.3 Les 4 types d'édition (recettes)
Soit B une image propre, W(B,R,m) = B avec le crop R filigrané par `wm.embed(crop, m)`, M_edit un masque d'édition. Étiquette = R ET NON M_edit sauf transplantation.
1. **Suppression/altération d'objet** : M_edit = masque d'objet (COCO ou SAM, dilaté de 4-12 px), contenu = LaMa (rapide, quasi gratuit) ou SD-inpaint avec prompt vide/« background » ; l'« altération » = SD-inpaint avec un prompt d'objet différent.
2. **Ajout d'objet** : coller un objet issu d'une autre image (masque COCO/SAM, option : lissage de bord gaussien 1-3 px, ajustement de luminosité) **ou** SD/FLUX inpaint avec prompt d'objet dans un masque. Étiquette 0 sur l'objet.
3. **Outpainting** : placer W(B) (ou une version rognée/réduite) sur un canevas plus grand d'une à trois marges (25-100 %), remplir par outpainting diffusion, **puis rééchelonner à 512 avec probabilité p_resize** (calibrer sur le train, voir §3.2 d) ; étiquette = rectangle R transformé.
4. **Transplantation** : objet (masque COCO/SAM) pris dans W(B1) pleine image (ou le rectangle R contient l'objet), collé sur B2 propre à une position éventuellement décalée ; étiquette 1 uniquement sur l'objet. Un décalage de position casse l'ancrage du motif pour les méthodes ancrées en position (SS) : inclure avec et sans décalage.
Masques d'édition : mélange appris des statistiques du train (aire, nb de composantes, contact avec le bord) : masques d'objets SAM/COCO, traits/ellipses/rectangles façon LaMa (grands et petits), fines structures (« objet mince »), visages.
Leurres : avec p≈0,15 ajouter un deuxième filigrane **avec un autre message** dans une zone d'étiquette 0 (force le réseau à utiliser l'accord au message, pas la simple présence de bruit). À ablater : si cela dégrade, p=0.
« Édition sur édition » : sur les 510 images réelles, ajouter des trous synthétiques supplémentaires (LaMa/copier-coller) et mettre à jour l'étiquette (étiquette_réelle ET NON nouveau_trou) : multiplie le réel avec un réalisme exact des zones génératives.

### 3.4 Modèles de diffusion sur A100 40 Go (pré-télécharger sur le login) [MÉMOIRE : noms de dépôts HF à vérifier avant téléchargement, certains sont soumis à licence/accès]
| Modèle | Usage | Débit estimé sur 1 A100, 512 px, fp16 [HYP] |
|---|---|---|
| LaMa (`big-lama`) | suppression d'objet, trous quelconques ; non diffusion | ~20-50 img/s |
| SD-2-inpainting (`stabilityai/stable-diffusion-2-inpainting`, natif 512) | suppression/ajout/outpainting ; **choix principal** | ~0,4-0,6 img/s (30 pas) ⇒ ~1,5-2 k/h |
| SDXL-inpainting (`diffusers/stable-diffusion-xl-1.0-inpainting-0.1`) | diversité de texture, ajout d'objets | ~0,2-0,3 img/s à 512 |
| FLUX.1-Fill-dev (12 B, bf16 ~24 Go, licence non commerciale, accès HF fermé) | texture « moderne » proche des éditeurs récents | ~0,1-0,2 img/s |
| SAM / SAM2 (`facebook/sam-vit-huge` ou `sam2`) | masques d'objets automatiques sur les bases | ~1-3 img/s pour 20 masques |
Plan de volume : banque de 6-10 k éditions diffusion (~3-5 heures-GPU) + éditions LaMa/copier-coller illimitées (CPU/GPU léger). On ne sait pas quel générateur les organisateurs ont utilisé : **diversifier** (LaMa + SD2 + SDXL, FLUX si l'accès existe) et laisser le fine-tune sur les 510 réels faire la correction de domaine. Contrainte pratique : `diffusers`/`accelerate` ne sont pas dans requirements.txt, à installer sur le login dans un venv séparé (torch 2.5.1 épinglé : ne pas laisser pip le changer).

### 3.5 Débit d'embedding et stockage [HYP]
PixelSeal/TrustMark/StegaStamp : ~20-60 crops/s chacun sur A100 ⇒ 25-40 k crops filigranés par méthode en ~15-30 min. Banque stockée : crop filigrané PNG + boîte + message ; ~0,3-0,5 Mo l'unité ⇒ 10-20 Go par méthode, acceptable sur `/p/scratch/training2666`. Composite à la volée par DataLoader (le projet donne 128 CPU par tâche sur le gabarit `main-job-template.sh`).

## 4. Recette d'entraînement

### 4.1 Pertes (alignées sur la métrique)
- `L = BCE(logits, y) + λ_L · Lovász_img + λ_I · SoftIoU_img + λ_c · CE(tête image)`, λ_L = 0,5, λ_I = 0,5, λ_c = 0,2 [HYP, à régler sur OOF].
- **Lovász-hinge par image** (pas par batch) appliqué à la classe 1 **et** à la classe 0 (`lovasz(p)` et `lovasz(−p)` moyennés) : reproduit la moyenne des deux IoU.
- **Règle d'union vide** : si G_c est vide pour une classe, la métrique vaut 1 si la prédiction est elle aussi vide, 0 sinon ; un seul faux positif sur une image « tout-1 » fait passer le score de 1 à 0,5. Dans le soft-IoU, utiliser `(I+ε)/(U+ε)` avec ε=1 pour que la classe vide pousse la somme des probabilités vers 0 ; et traiter l'image-niveau en dur par la tête 3 classes (seuil calibré en CV). **Mesurer d'abord** la part d'images « tout-1 » ou « tout-0 » dans le train (5 min) ; si elle est ≈ 0, ne pas dépenser d'effort dessus.
- Cible : `downsample_mask` officiel [KIT: utils.py] sur le 512 ; supervision profonde aux strides 8 et 16 avec la cible réduite (moyenne puis seuil 0,5).
- Optionnel (si les frontières sont floues) : perte de bords type MVSS-Net sur le gradient du masque.

### 4.2 Optimisation [HYP, valeurs de départ]
- AdamW, lr 3e-4 (décodeur) / 1e-4 (backbone), wd 0,05, warmup 500 pas, cosine, bf16 AMP, channels_last, EMA des poids (0,999).
- Pré-entraînement synthétique : batch 32, 512x512, ~25-40 k exemples/époque, 15-25 époques par méthode ≈ 2-3 h sur 1 A100 si ~100-150 img/s [HYP : ConvNeXt-T à 512 ≈ 23 GFLOPs en passe avant].
- Fine-tune réel : par fold, 136 images réelles (+ « édition sur édition » + 50 % de synthétique de rappel), batch 16, 20-40 époques courtes ⇒ ~20-30 min par fold et par méthode.
- Fenêtres : l'entrée est toujours 512x512 entier (pas de patchs aléatoires) car la position du rectangle est informative.

### 4.3 Augmentations
Autorisé : composition synthétique, décalages multiples de 4 px, flips/rot90 **appliqués à l'image ET aux cartes d'accord ET au masque** (à valider en ablation : si l'accord est calculé sur l'image non retournée puis retourné, le motif reste cohérent ; sinon désactiver), leurres de message, bruit gaussien très faible (σ≤0,5/255) en option.
Interdit (détruit le filigrane, absent du test PNG) : rescale/rotation non multiple de 90°, flou, JPEG, changements de couleur forts. **À vérifier** que les images de test n'ont pas de recompression (comparer des statistiques de base train/test en 10 min).

### 4.4 Coût
| Poste | Heures-GPU A100 [HYP] |
|---|---|
| Banque diffusion (SD2 + SDXL + SAM) | 4-6 |
| Embedding des 3 méthodes | 1-2 |
| Pré-entraînement 3 méthodes | 6-9 |
| Fine-tune 5 folds x 3 méthodes | 6-9 |
| Ablations courtes (8-10 runs de 20-30 min) | 4-5 |
| Inférence, recherche de boîte, TTA, pseudo-étiquettes | 3-5 |
| **Total** | **≈ 25-35** (+ marge 30 % pour files d'attente et échecs) |

## 5. Protocole de validation

### 5.1 Trois mesures de 30 min à faire AVANT d'entraîner (elles décident de l'architecture)
1. **Statistiques des 510 masques** : fraction de masques tout-1/tout-0 ; aire de la zone filigranée ; nb de composantes ; rapport masque / enveloppe convexe ; contact avec les bords. Sert à (a) calibrer les générateurs de masques, (b) **étiqueter les types d'édition heuristiquement** pour la validation détaillée : « un seul rectangle plein touchant les bords opposés »≈ outpainting ; « rectangle avec trou(s) intérieur(s) » ≈ suppression/ajout ; « forme compacte non rectangulaire isolée » ≈ transplantation ; `full_watermark=true` aide à isoler la transplantation [KIT].
2. **Existence du signal** : pour chaque méthode, `decode(crop(bbox(masque GT)))` ; précision de bits dans la boîte contre un crop propre de même taille. Si ≈ 0,5 partout pour une méthode : le décodeur ne résiste pas aux éditions/rééchelonnage et il faudra des indices non filigrane.
3. **Fenêtre glissante décalée vs boîte exacte** (StegaStamp et TrustMark) : confirme ou infirme que l'accord ne tient qu'au crop exact (conclusion n°2).

### 5.2 Validation croisée
- 5 folds sur les 510 images, **stratifiés par méthode et par type d'édition heuristique** ; même découpage pour tous les modèles ; tout seuil/post-traitement ajusté uniquement sur les prédictions hors-pli (OOF).
- Score rapporté : mIoU officiel moyen, par méthode, par type d'édition, sur le sous-ensemble « union vide » ; intervalle de confiance par bootstrap sur les images (paires pour comparer deux variantes).
- Les images synthétiques ne servent **jamais** à la validation (domaine différent) ; le seul juge est le réel hors-pli. Le classement public (306 images, ±0,014) sert de contrôle de cohérence, pas de sélection.
- Les 5 modèles de fold forment l'ensemble final (moyenne des probabilités), donc la validation croisée n'est pas un coût perdu.
- Signal d'alerte : OOF ≫ public ⇒ fuite (ex. même image de base dans la banque synthétique et dans un fold de validation : exclure du pré-entraînement les images du train réel qui servent de validation, ou ne pas utiliser le train réel comme base de synthèse avant le gel des folds).

### 5.3 Post-traitement, TTA, ensemble
- Seuil par méthode (OOF) ; plus grande composante + suppression des composantes < 4-8 cellules ; remplissage des trous < 4-8 cellules ; **pas** de rectangle forcé pour les formes libres (décision par détecteur de forme appris en CV).
- Option « seuil par image optimisant le mIoU espéré » : échantillonner des masques à partir des probabilités et choisir τ ∈ [0,3, 0,7] maximisant le mIoU moyen simulé [HYP : surévalue la confiance si les pixels sont corrélés].
- TTA : flips/rot90 si l'ablation les a validés (8 vues) ; sinon identité seule.
- Ensemble : 5 folds x (éventuellement 2 graines) ; gain attendu de l'ordre +0,01 à +0,03 [HYP].
- Pseudo-étiquetage des 1020 images de test (étiquettes issues de l'ensemble, confiance élevée seulement) pour un tour de fine-tune : permis puisque seules les entrées de test sont utilisées, mais **gain incertain et risque d'auto-renforcement** ; à n'engager que si le temps reste après H13 et que le gain OOF-simulé est mesuré en validant sur les folds retenus. Aucune ressource du concours n'est appelée.

## 6. Pronostic honnête [HYP, non mesuré]
- Plancher sans apprentissage : tout-0/tout-1 ≈ 0,25-0,35 si la zone filigranée vaut environ la moitié de l'image (chiffre du rapport existant).
- Segmenteur extracteur-backbone + boîte pour PS : plausible 0,75-0,90 ; TM : 0,75-0,90 (rectangle presque compact) ; SS : plus incertain, 0,55-0,85 selon l'existence du signal (§5.1.2). Moyenne visée : 0,70-0,85. Je ne l'ai pas mesuré ; c'est le premier test réel (H3-H6).

## 7. Plan heure par heure, 2 ingénieurs, 18 h
H0 = samedi 17:00 ; fin du code dimanche 12:00 (H19) ; on réserve H17-H19 de marge. Rôles : **E1 = données** (téléchargements, banques, diffusion, composeur), **E2 = modèle** (entraînement, évaluation, soumission). Rotations de sommeil : E1 dort H9-H13, E2 H13-H17 (4 h chacun ; E2 garde un pipeline qui tourne seul pendant son sommeil).

| Heure | E1 (données) | E2 (modèle/éval) | Jalon |
|---|---|---|---|
| H0-1 | Login : télécharger COCO val+annot, SD2-inpaint, SAM, LaMa, timm convnext_tiny dans `$HF_HOME` ; tester `HF_HUB_OFFLINE` ; lancer `download_models.py` si pas déjà fait | Charger train, tout-0 + tout-1, vérifier l'évaluateur local ; stats des masques (§5.1.1) ; **soumission de sécurité valide** (tout-0 ou meilleur trivial) ; préparer folds stratifiés | J0 : fichier de soumission valide chargé avec `weights_only=True` |
| H1-3 | Script d'embedding des 3 méthodes sur la banque B (5 k bases x boîtes/messages) ; composeur v0 (rectangle + copier-coller COCO + LaMa) | Mesures §5.1.2 et §5.1.3 ; carte d'accord par position (PS, TM) ; baseline fenêtre sur 50 images | J1 : on sait quel signal existe par méthode |
| H3-5 | Composeur v1 : 4 types d'édition, masques SAM, leurres ; lancer génération diffusion (SD2 puis SDXL) sur 1 GPU | Modèle v0 par méthode sur synthétique v0 (3 GPU en parallèle) ; évaluation zéro-coup sur le réel | J2 : premier mIoU réel (même mauvais) |
| H5-8 | Génération diffusion en continu ; intégration dans les composites ; vérifier visuellement 50 exemples contre le train | Fine-tune réel, 5 folds PS et TM ; calibrage seuils ; stade boîte pour TM/SS ; **soumission 1 vraie** (modèle v1) | **J3 (H8) : OOF ≥ 0,60 ? voir arbre §8** |
| H8-10 | (E1 dort H9) fin du pré-traitement, banque diffusion v1 (≥ 6 k) | Pré-entraînement v2 avec diffusion ; ablations : SRM, flips, leurres (runs de 25 min) | J4 : v2 > v1 de 0,02+ (bootstrap) |
| H10-13 | Édition sur édition du réel ; SDXL/FLUX pour diversité | StegaStamp : boîte vérifiée + réseau ; post-traitement ; **soumission 2** | J5 (H13) : OOF par type d'édition publié dans le canal |
| H13-15 | Remplacer E2 : vérifier folds, ensemble, TTA ; pseudo-étiquettes (optionnel) | (dort H13-17) | |
| H15-17 | Inférence finale sur 1020 images avec les 5 folds x 3 méthodes ; boîte vérifiée ; assemblage | (dort) | J6 : soumission candidate finale |
| H17-19 | Marge : contrôles de format, ré-exécution complète depuis un état propre, une seule soumission finale supplémentaire si OOF a progressé | idem | Gel du code H19 = dimanche 12:00 |

Ordre de soumission : au plus 4 soumissions au total (safe, v1, v2, finale) pour limiter la tentation de sur-régler le public.

## 8. Arbre de décision si la base sous-performe (OOF réel)
1. **OOF < 0,45 pour une méthode** : le signal manque. Vérifier §5.1.2 : si la précision de bits dans la boîte GT est ≈ 0,5, pas de signal filigrane : basculer sur indices génériques (texture générée, bords, empreinte de bruit) + prior de rectangle ; ne pas insister sur l'accord au message.
2. **Zéro-coup synthétique → réel < 0,5 mais le signal existe** : écart de domaine : (a) vérifier §3.2 c et d (composite exact ? rééchelonnage ?) ; (b) augmenter la part du réel (fine-tune plus long, « édition sur édition ») ; (c) inclure l'aller-retour VAE dans la synthèse.
3. **Bons gros rectangles, mauvais petits trous/objets fins** : le stade 128 est trop grossier : augmenter la résolution interne (entrée 512 native, stride 2 dans le décodeur puis réduire), perte de bords, SAM pour caler les frontières (priorité basse).
4. **Faux positifs sur images « tout-propre/tout-filigrané »** : calibrer la tête image 3 classes ; en dernier recours un seuil dur sur la somme des probabilités.
5. **Un type d'édition sous-performe (ex. transplantation)** : augmenter sa part dans la synthèse ; contrôler le décalage de position pour SS (§3.3-4).
6. **StegaStamp reste < 0,55 à H10** : abandonner son réseau dédié ; appliquer le réseau PixelSeal/générique + boîte vérifiée, ou, si pas de signal, la meilleure heuristique géométrique ; **ne pas** bloquer les autres méthodes pour lui (le score est la moyenne des trois tiers égaux).
7. **Pipeline lent / GPU saturés par T2-T3** : réduire à un modèle unifié FiLM, 2 folds, pré-entraînement 8 époques ; sauter diffusion (LaMa + copier-coller seulement).
8. **Panique à H15** : soumettre la meilleure combinaison déjà validée OOF ; aucune nouvelle idée après H15.

## 9. Risques et vérifications à faire en premier
- Le dataset réel peut contenir des opérations non anticipées (rééchelonnage d'outpainting, recompression) ; les trois mesures de §5.1 les détectent en ~1,5 h.
- Chiffres de débit et d'heures-GPU : estimés, à remplacer par une mesure sur 100 exemples dès H1.
- Licences/accès : FLUX.1-Fill-dev et certaines versions SD peuvent exiger un jeton HF ; ne pas en dépendre.
- Le partage du nœud avec T2/T3 et les autres équipes : annoncer l'usage GPU sur le bus (issue #2) avant de lancer 4 GPU [CLAUDE.md].
- Je n'ai appelé aucun endpoint du concours, téléchargé aucune donnée ni modifié le dépôt.

## Sources
- WAM (HTML lu) : https://arxiv.org/html/2411.07231 ; code https://github.com/facebookresearch/watermark-anything
- MaskWM : https://arxiv.org/abs/2504.12739 (résumé via recherche) ; code https://github.com/hurunyi/MaskMark
- Banc d'essai filigranes locaux : https://arxiv.org/abs/2609.16832 (résumé)
- TruFor : https://arxiv.org/abs/2212.10957 (résumé)
- Localisation d'images régénérées par diffusion : https://arxiv.org/html/2506.05466v2 (TGIF2, résumé via recherche) ; InpDiffusion https://arxiv.org/pdf/2501.02816
- Tâche/kit : reports/sources/Watermark_Localization.txt, reports/sources/t1/starter_kit/wmloc/{watermarks,utils,data}.py, download_models.py, requirements.txt ; reports/sources/hf-hackathon_setup.sh, hf-README.md
- MVSS-Net, CAT-Net, PSCC-Net, ManTra-Net, SAFIRE, LaMa, SD/SDXL/FLUX inpainting : non relus cette session (MÉMOIRE) ; vérifier les noms de dépôts avant de télécharger.
