# T1 Watermark Localization : méthodes qui sortent du lot (ideas-t1)

Date : sam. 21/11/2026 ~16:30. Reste ~19 h 30 (code gelé dim. 12:00). Je n'ai rien exécuté, téléchargé ni appelé ; tout vient du kit local (watermarks.py, utils.py, data.py, énoncé) et des rapports déjà écrits (t1-decoders, t1-segmentation, t1-watermark-etat-de-lart, strategy, intelligence). Je ne répète pas leurs idées (fenêtres glissantes, z nul par messages aléatoires, Kadane 2D, tomographie, IG, optimisation de masque, réseau d'appui WAM-like, données synthétiques LaMa/SD) : je les cite comme « acquis » et j'ajoute ce qui manque.
Légende : [KIT] lu dans le kit ; [MESURÉ] aucun (rien n'est mesuré) ; [CALC] calculé à la main sur la métrique ; [PAPIER url] ; [HYP] hypothèse. Les gains sont en points de mIoU globaux (moyenne des 3 méthodes) par rapport à une base « fenêtres + segmenteur + seuil » supposée à ~0,70 [HYP]. Heures = temps réel pour 1-2 personnes sauf mention ; GPU = heures-GPU cumulées A100.

## 0. Faits structurels exploitables (tous vérifiés dans le kit ou par calcul)

1. Les embedders du kit SONT ceux du dataset ; on peut fabriquer des données exactes [KIT : « exactly the embedders used to build the dataset »]. Et l'embedder est déterministe pour (image, message) : on peut donc calculer le résidu qu'il « voudrait » ajouter à n'importe quelle image, pour le vrai message ou un autre.
2. Le message de chaque image de test est donné [énoncé], et la méthode aussi : on teste une hypothèse précise (« ce message est-il là ? ») et non une détection ouverte.
3. [CALC] Asymétrie de la règle d'union vide (N = 16 384 cellules) : GT tout-1 et prédiction avec k=100 cellules 0 : IoU0 = 0 (P0 non vide, G0 vide), IoU1 = 0,994 -> score 0,497 ; prédire tout-1 : 1,0. GT avec m=200 cellules propres et prédiction tout-1 : 0,494 ; avec un masque qui trouve la zone propre à IoU0 = 0,5 : 0,74. Donc « tout-1 » est préférable dès que P(GT tout-1) > ~0,33 (voir I5). Même chose symétrique pour « tout-0 » (l'énoncé dit explicitement qu'une image peut n'avoir aucune zone filigranée).
4. [KIT utils.downsample_mask] cellule = 1 si >= 50 % des 16 pixels sont 1 : on peut raisonner en pixels 512 puis appeler la fonction officielle sur un masque pixel ; ça règle automatiquement l'arrondi des bords d'un rectangle non aligné sur 4 px.
5. [KIT] `full_watermark` existe en train : une partie des images a été filigranée en entier à 512 px (géométrie native 512), puis restaurée ; les autres ont un patch filigrané à SA taille (géométrie du rectangle). Deux hypothèses de géométrie à tester par image, au test (pas de drapeau).
6. [KIT] Le wrapper `embed()` rend un uint8 arrondi. Pour toute idée qui exploite le résidu (I1, I2), il faut le résidu en flottant : StegaStamp l'a déjà (`_run(..., ["residual"])` ×255, avant clip), PixelSeal : `out["imgs_w"]` avant `.round()` (code du wrapper à contourner), TrustMark : sortie interne de `tm.encoder` à vérifier [HYP]. Sinon le bruit d'arrondi (±0,5 niveau) est du même ordre que le signal (±1-3 niveaux).

## 1. Les idées (une fiche chacune ; classement en §2)

### I1. DEC : corrélation différentielle de ré-embedding (« trou par résidu attendu »)
1. Idée : pour une image x et son message m, calculer D = R(x,m) − R(x,m') où R(x,m) = embed(x,m) − x (résidu flottant) et m' = ~m (ou la moyenne de K messages aléatoires). Statistique par cellule : c(p) = Σ_{pixels de p} HP(x)·D / normalisation locale, HP = x − flou_σ(x). Là où le filigrane est présent, x = c + R(c,m) et R(x,m) ≈ R(c,m) : HP(x)·D ≈ |R_m|² − R_m·R_m' > 0 ; ailleurs HP(c)·D ≈ 0 (contenu naturel décorrélé du motif). Le contenu commun à R(x,m) et R(x,m') s'annule dans D.
2. Mécanisme : c'est un filtre adapté à résolution PIXEL avec le motif exact, alors que les fenêtres de décodeur ont une résolution de 64 px et que le champ réceptif fuit. Variantes : (a) test d'annulation : décoder embed(x,~m) par fenêtre ; dans une zone déjà filigranée avec m, le second résidu (anti-corrélé) annule en partie le premier, donc marge ~0 ; dans une zone propre l'embedder prend le contrôle (marge pleine) ; (b) test de gain : marge(embed(x,m)) − marge(x). Tous trois sont des sondes « ré-embedding » ; (a) et (b) utilisent le décodeur, donc résistent mieux au contenu mais restent à résolution de fenêtre.
3. Preuve : [HYP]. Le rapport coéquipier (§2.3.4) a esquissé la corrélation de résidu ; ce qui est nouveau ici : le nul différentiel m' (annule la corrélation contenu-embedder), la géométrie (le résidu dépend de la taille du patch, car le U-Net/ONNX travaille à 256/400 px puis rééchelonne) et l'usage en 2e étage une fois le rectangle R grossièrement connu (on calcule embed(crop(R),m), pas embed(image)), ce qui rend l'alignement exact et donne des trous au pixel. Deux géométries testées par image : patch R et image entière 512 (cas full_watermark, fait n°5) ; on garde celle de plus forte somme de c.
4. Gain : 0 à +0,08. Hypothèses : surtout StegaStamp (résidu basse fréquence, fort, flottant disponible) et TrustMark ; PixelSeal (JND) est faible en aplat. Là où il marche, il règle les trous fins (visage, objet mince) que les fenêtres ratent.
5. Faisabilité : 6-8 h (1 pers.), ~2 h-GPU (2 embeds/image + rectangles candidats : ~1 530 × (1 + ~20 candidats) embeds, ms chacun). Dépend : accès au résidu flottant par méthode ; rectangle grossier (I17/Kadane). Casse si : (i) résidu dépendant fortement du contenu (R(x_w,m) ≠ R(c,m)) -> signal noyé ; (ii) éditions qui lissent HP ; (iii) PNG propre mais éditeur appliquant un re-encodage global (le motif atténué partout).
6. Signal d'alerte : AUC pixelique de c(p) < 0,6 sur 20 images train avec le rectangle GT.
7. Test 30 min (20 images train, GT connue, 1 GPU) : calculer c(p) avec le rectangle GT exact par méthode, AUC cellule (dans GT=1 vs GT=0 à l'intérieur de la boîte englobante). Décision : AUC > 0,75 pour une méthode -> garder pour elle ; entre 0,6 et 0,75 -> canal d'entrée du réseau de fusion ; < 0,6 -> tuer.
8. Conformité : calcul sur images de test et embedder public, aucune requête serveur.

### I2. Noise2WM : régression dense du résidu sur paires exactes (pré-entraînement dense)
1. Idée : avec l'embedder on fabrique des paires exactes (c, x_w = c + r) pour des images propres quelconques et des messages aléatoires. Entraîner un U-Net (par méthode) à prédire r (ou le masque d'énergie de r) à partir de x_w seul, éventuellement avec en entrée supplémentaire R(x_w,m) (message-conditionné). Les images à entraîner sont des mosaïques : un rectangle filigrané + zones propres + zones d'un autre contenu (les éditions n'ont pas besoin d'être réalistes pour apprendre l'empreinte de l'embedder).
2. Mécanisme : supervision DENSE au pixel sans aucun masque d'édition réaliste : tous les pixels sont étiquetés par construction, 100 % exact, volume illimité. Le réseau apprend l'empreinte de CET embedder (statistiques spatiales et spectrales du résidu), indépendamment du décodeur. Le coût « domaine » est limité car l'empreinte est la même qu'au test ; l'écart restant est ce qu'a fait l'éditeur génératif (corrigé par fine-tune sur les 510 réels). Sert soit de segmenteur direct (énergie prédite), soit de pré-entraînement du réseau d'appui du rapport segmentation (au lieu de commencer à partir d'ImageNet).
3. Preuve : [HYP]. Analogie [PAPIER url] : WAM entraîne un détecteur dense par pixel sur filigrane synthétique https://arxiv.org/abs/2411.07231 ; ici la différence : l'embedder cible est fixe, donc c'est une régression d'un opérateur connu.
4. Gain : +0,02 à +0,08 comparé à un segmenteur initialisé ImageNet sur 510 images (le pré-entraînement exact manque aux plans actuels qui visent surtout des composites d'éditions). Hypothèse : le motif est assez détectable sans message ; sinon le canal message (I1) porte le signal.
5. Faisabilité : 6-8 h, 6-10 h-GPU (3 méthodes en parallèle, 30-60 k exemples chacune, 256² ou 512² crops). Dépend : images propres (train/test régions propres suffisent ; COCO optionnel), embeds ~20-60/s/GPU [HYP]. Casse si : le motif de PixelSeal est trop faible en aplat (JND) -> sortie plate ; rien d'autre de grave.
6. Signal : après 15 min d'entraînement sur 3 k exemples, AUC pixelique sur les 20 images train < 0,65.
7. Test 30 min : 3 k paires, U-Net 5 couches, 10 min d'entraînement par méthode, AUC cellule sur 50 cellules de train réel (GT=1 vs GT=0). Si AUC > 0,75 pour au moins 2 méthodes -> poursuivre.
8. Conformité : OK (données synthétiques, embedders du kit).

### I3. Détecteur sur le vecteur de logits signés (LDA / régression logistique) au lieu de la moyenne
1. Idée : pour chaque fenêtre, au lieu de s = moyenne de (2m_i−1)·clip(l_i), apprendre une régression logistique (L2 ou LDA) sur le vecteur s_i (256 ou 100 dimensions) -> P(fenêtre filigranée). Étiquette : recouvrement GT >= 90 % vs <= 10 %. 170 images × ~1 400 fenêtres = 240 k exemples par méthode : largement assez pour 100-256 poids.
2. Mécanisme : les bits n'ont pas tous la même fiabilité (décodeur biaisé, certains bits seulement robustes sous petits crops ; StegaStamp sature) ; les erreurs sont corrélées (le décodeur échoue de façon structurée sous mismatch d'échelle). Une pondération par bit et la décorrélation (LDA) augmentent l'AUC à nombre de bits égal ; elle apprend aussi le biais par bit (soustraction de b_i) qui, avec un z nul, sont deux corrections de la même chose. Gratuit : mêmes logits mis en cache.
3. Preuve : [HYP] (classique en détection ; non mesuré ici).
4. Gain : +0,01 à +0,04 (surtout StegaStamp et petites fenêtres où le z moyen est faible).
5. Faisabilité : 1-2 h, 0 GPU supplémentaire (sklearn/torch sur le cache de logits). Dépend du cache de logits du plan de base. Casse : surapprentissage sur 170 images -> L2 fort, validation croisée par image (pas par fenêtre !).
6. Signal : AUC OOF identique à la moyenne simple (écart < 0,01).
7. Test 30 min : AUC OOF fenêtre (5 plis par image) de la moyenne vs logistique, par méthode et par taille de fenêtre. Garder si AUC +0,02 ou plus sur au moins une méthode.
8. Conformité : OK.

### I4. Décision d'IoU espérée maximale, avec échantillonnage du postérieur (et options tout-0/tout-1)
1. Idée : à partir de probabilités calibrées par cellule p(c) (OOF), échantillonner S = 64 « mondes » GT : en tirant des paramètres structurés (rectangle + trous) du postérieur, ou simplement des Bernoulli corrélées par blocs de 4×4 cellules ; pour chaque seuil/ensemble candidat (top-k par p, k balayé, plus les candidats tout-0 et tout-1 et « masque rectangle »), calculer le mIoU simulé avec la règle d'union vide ; choisir le candidat de plus grand mIoU espéré PAR IMAGE.
2. Mécanisme : le seuil unique de 0,5 maximise la précision par cellule, pas l'IoU moyenne des deux classes. Pour la classe minoritaire (petit trou, petit objet transplanté) le seuil optimal est plus bas (rappel prioritaire) et il dépend de l'incertitude de l'image ; la règle d'union vide ajoute un saut de +0,5 sur les cas tout-1/tout-0. Optimisation d'IoU espérée connue : Nowozin 2014 (« Optimal decisions from probabilistic models : the intersection-over-union case », CVPR) [PAPIER https://arxiv.org/abs/1402.1782 — numéro d'arXiv à revérifier].
3. Preuve : [CALC] pour les sauts d'union vide ; le reste [HYP]. Le rapport segmentation propose un « seuil par image optimisant le mIoU espéré » en une ligne, sans l'algorithme.
4. Gain : +0,01 à +0,04 (la part élevée si beaucoup de petits objets ou de tout-1).
5. Faisabilité : 3-4 h (c'est du numpy/torch sur les OOF), 0 GPU. Dépend de la calibration des probabilités (température/isotonique sur OOF). Casse : probabilités mal calibrées -> pire que seuil fixe ; parade : valider par mIoU OOF réel, pas simulé.
6. Signal : mIoU OOF avec règle EV ≤ mIoU avec seuil fixe optimisé.
7. Test 30 min : sur les OOF d'une première version (même un seuillage de moyenne de fenêtres), appliquer (i) seuil fixe optimisé, (ii) top-k EV par image, (iii) (ii)+tout-0/tout-1. Garder si +0,01 en OOF et bootstrap par image sur 510.
8. Conformité : OK.

### I5. Triage d'image à trois voies (tout-0 / tout-1 / mixte) avec règle de rentabilité explicite
1. Idée : classifieur d'image (logistique sur ~10 features : max z de fenêtre, fraction de cellules >τ, taille du plus grand composant, énergie de bords, z de l'image entière, TTA, etc.) donnant P_all1, P_all0, P_mix. Décision par la formule : prédire tout-1 si 0,503·P_all1 > (1−P_all1)·L1, L1 = perte attendue du masque par rapport à tout-1 quand l'image est mixte (≈ 0,25, [CALC] pour m petits), de même pour tout-0.
2. Mécanisme : la moitié du score d'une image mixte quasi-pleine dépend de quelques cellules de la classe minoritaire ; et une image pleine bien appelée vaut 1,0 au lieu de 0,5. Si le train contient, disons, 8 % d'images tout-1 ou tout-0 [HYP à mesurer], le gain maximal est 0,08 × 0,5 = +0,04 si bien appelées ; en cas d'erreur la perte est de 0,25 × (faux positifs).
3. Preuve : [CALC] pour l'asymétrie ; la part d'images pleines est INCONNUE (statistique à mesurer en 5 min sur le train).
4. Gain : 0 à +0,05 ; borne haute selon la fréquence des images pleines.
5. Faisabilité : 2 h, 0 GPU. Dépend de la statistique du train. Casse : fréquence ~0 -> aucun gain (on ne dépense alors que 5 min).
6. Signal : moins de 2 % d'images pleines dans le train.
7. Test 5-30 min : compter les masques tout-0/tout-1 + presque (moins de 20 cellules minoritaires) dans les 510 masques. Si >= 4 % -> faire le classifieur ; sinon seulement une garde.
8. Conformité : OK.

### I6. Classifieur de type d'édition + solveur paramétrique par type (mélange d'experts)
1. Idée : 4 familles de formes à partir du masque GT (heuristique, §5.1.1 du rapport segmentation) : (a) rectangle plein (outpainting), (b) rectangle avec trou(s) (suppression/ajout), (c) forme libre compacte (transplantation), (d) pleine/vide. Un classifieur d'image (features d'évidence + `full_watermark` appris comme étiquette auxiliaire) choisit parmi trois solveurs : (a) rectangle MAP seul (I11) ; (b) rectangle + trous par coupe de graphe (I13) ; (c) segmentation libre du réseau. Ou, mieux, un MDL : on calcule les trois masques et on garde celui de plus haute vraisemblance de l'évidence − pénalité de complexité (nombre de paramètres de forme).
2. Mécanisme : chaque type a un a priori de forme très informatif que le réseau libre n'exploite pas (rectangle exact -> bord au pixel, IoU1 > 0,97 sur les gros rectangles). Sur outpainting le masque est un rectangle quasi parfait : le solveur rectangle bat le réseau par les bords ; sur transplantation il perd, d'où la sélection.
3. Preuve : [HYP]. Le rapport decoders propose déjà un « sélecteur logistique » ; j'ajoute le MDL et la pénalité de complexité par type, et la répartition des types comme mesure préalable.
4. Gain : +0,02 à +0,06 (grande part des images en rectangle plein ? à mesurer).
5. Faisabilité : 4-5 h, 0 GPU. Dépend d'I11, I13 et du réseau. Casse : confusion de type -> perte ; parade : si P(type) < 0,6, retomber sur le masque du réseau.
6. Signal : accuracy OOF du sélecteur < 70 %.
7. Test 30 min : étiqueter les 510 GT par règle, afficher fractions et IoU moyen du « rectangle MAP » vs « réseau » par type (avec l'évidence de la base). Si un type pèse > 25 % et que le solveur dédié gagne de 0,05 sur ce type -> garder.
8. Conformité : OK.

### I7. Détecteur de contenu généré (traces VAE / diffusion), indépendant du filigrane
1. Idée : les éditions génératives (outpainting, suppression, ajout, altération) font passer des pixels par un VAE (SD/FLUX : décodeur avec périodicités de 8 px, spectre haute fréquence atypique) et un bruit de réseau, alors que les pixels d'origine hors masque ne passent pas par le VAE s'ils sont recollés. Entraîner un petit CNN « résidu haute fréquence -> généré ou original » où « généré » = cellules GT=0 à l'intérieur de la boîte englobante du filigrane (trous) ou hors de la zone d'outpainting ; entrées : HP(x) (Laplacien/SRM), spectre local (FFT 32×32), profondeur d'autocorrélation à lag 8. Le canal est INDÉPENDANT du message : utile partout où le décodeur est à zéro (aplats PixelSeal, petits objets).
2. Mécanisme : fournit une 2e vue conditionnellement indépendante de la première (message vs traces d'édition). Les deux vues se combinent par le réseau de fusion ou se corrigent (co-training I9). Données : supervision réelle sur 510 images ; partage entre les trois méthodes (le signal « généré » ne dépend pas du filigrane), soit 3× plus de données pour ce détecteur ; supervision synthétique par aller-retour VAE d'un jeu d'images (SD VAE disponible localement : encode/decode ~100 img/s sur A100).
3. Preuve : [PAPIER https://arxiv.org/html/2506.05466v2 (TGIF2), https://arxiv.org/abs/2212.10957 (TruFor)] : les localiseurs de falsification savent détecter des régions régénérées par diffusion après fine-tune ; non évalué ici sur ces données. L'existence d'une périodicité de 8 px après décodage VAE est [MÉMOIRE] et à mesurer.
4. Gain : +0,02 à +0,07 sur les types « trou » et « outpainting » ; 0 sur transplantation (pas de génération). Hypothèse : le composite recolle l'original hors masque (sinon la trace est partout, sans information de localisation, mais alors c'est aussi le filigrane qui est partout atténué, à voir en E3).
5. Faisabilité : 5-6 h, 4-6 h-GPU. Dépend : modèle VAE (pré-télécharger sur le login) pour la synthèse ; sans VAE on s'entraîne sur le réel seulement. Casse : la trace VAE est effacée par une étape de lissage/recompression des organisateurs ; les images du test pourraient avoir été re-sauvegardées (vérifier PNG).
6. Signal : FFT moyenne inchangée entre pixels « trou » et pixels « filigranés » sur le train.
7. Test 30 min : sur 50 images train, spectre de puissance moyen 2D de HP(x) en blocs 64×64 dans les trous vs dans le filigrane vs hors rectangle ; chercher des pics à f = 1/8, 2/8... Décision : pic détectable (rapport > 1,5) ou AUC d'une régression logistique sur 16 features spectrales > 0,7 -> garder.
8. Conformité : OK (n'utilise que les images publiques et un VAE public).

### I8. Test de structure inter-images : doublons de contenu entre train, test et méthodes
1. Idée : 1 530 images = 3 × 510. Si les 510 images de base sont réutilisées pour les trois méthodes (ou si train et test partagent des bases), alors un voisin le plus proche (embedding basse résolution + hash perceptuel) sur 1 530 images révèle des paires/triplets. Pour une paire (x_A avec méthode A, x_B méthode B) de la même base, la différence x_A − x_B hors éditions est la somme de deux résidus ; mais surtout la valeur de x_A dans une zone où x_B est propre donne une référence quasi exacte du contenu propre : localisation par différence au pixel près.
2. Mécanisme : exploite la structure de la donnée publique (aucune requête, uniquement des images fournies). Gain énorme SI la structure existe.
3. Preuve : [HYP] très incertain ; l'énoncé dit seulement « split equally across the three methods ». Les décomptes (170/340 par méthode) ne prouvent rien.
4. Gain : 0 (probable) ou +0,10 à +0,25 si la structure existe. Probabilité estimée 10 % [HYP].
5. Faisabilité : 30-45 min, 0 GPU (CPU : miniatures 32×32, distance L2/corrélation sur 1 530×1 530). Casse : rien (coût borné).
6. Signal : aucune paire avec corrélation > 0,9 sur les miniatures.
7. Test 30 min : miniatures 32×32 de toutes les images, matrice de corrélation, histogramme du max hors diagonale par split. Décision : s'il y a au moins 5 % d'images avec un voisin à corrélation > 0,95 -> exploiter ; sinon abandonner.
8. Conformité : OK (c'est une analyse des données publiques de la tâche) ; ce n'est pas de l'usage d'un autre équipe ni du serveur. À signaler dans le journal d'équipe par transparence.

### I9. Co-entraînement / auto-apprentissage sur les 1 020 images de test avec certificats
1. Idée : sur le test (message connu), chaque pseudo-masque a un CERTIFICAT vérifiable : z du crop du rectangle prédit (message connu, nul par messages aléatoires) et z « zéro » du crop de la zone prédite propre (doit être à ~0). On ne retient en pseudo-label que les images avec certificats forts des deux côtés (précision élevée, biais de sélection vers les faciles). Deux vues : réseau d'évidence-décodeur (I3/I10) et réseau de traces (I7) ; chacune étiquette pour l'autre (co-training). En plus : AdaBN (recalibrer les statistiques de BatchNorm sur les 1 020 + 510 images : adaptation de domaine gratuite).
2. Mécanisme : 340 images test par méthode >> 170 en train : doubler les données. Le certificat évite la dérive d'auto-renforcement typique du pseudo-étiquetage. Le test n'a pas de labels ; on n'emploie que ses entrées.
3. Preuve : [HYP] ; co-training (Blum-Mitchell) et AdaBN (Li et al. 2016 https://arxiv.org/abs/1603.04779) sont classiques ; non évalués ici.
4. Gain : +0,01 à +0,04 (surtout si le domaine test ≠ train par les types d'éditions).
5. Faisabilité : 4 h pour 1 pers. après que la base tourne ; 2-3 h-GPU. Casse : renforcement de biais, fuite de validation (garder les plis OOF du train étrangers au pseudo-étiquetage : ne pas utiliser les images de validation dans l'AdaBN... en fait AdaBN sans labels est sans risque).
6. Signal : mIoU OOF de la v2 (avec pseudo) < v1 sur les plis de validation.
7. Test 30 min : AdaBN seul : recalculer les statistiques BN sur train+test, comparer l'OOF avant/après. Garder si +0,005 ou plus, ou sans perte. Pseudo-labels : seulement à H15+ si le temps reste.
8. Conformité : OK (transductif sur entrées publiques).

### I10. Calage des frontières sur les bords de l'image (filtre guidé / CRF dense / segments SAM)
1. Idée : les masques d'objets (suppression, ajout, transplantation) suivent des contours d'objets ; la sortie du réseau (stride 4 ou carte floue) ne les suit pas. Raffiner la carte de probabilité p avec un filtre guidé (guide = image RGB, rayon 4-8 px, eps 1e-3 : cinq convolutions de boîte, torch, 1 ms) ou un CRF dense à noyau bilatéral ; option segments : superpixels (SLIC, CPU) ou SAM (vote par segment : moyenne de p dans chaque segment, seuil à 0,5, option de fusion avec le masque rectangle).
2. Mécanisme : sur objets minces/visages, l'IoU de la classe minoritaire est dominée par les bords ; un trou de 20×20 cellules a un périmètre de 80 cellules, soit 20 % de la surface.
3. Preuve : [HYP]. Les filtres guidés améliorent classiquement la segmentation à bords nets (He et al.) ; gain non mesuré ici.
4. Gain : +0,01 à +0,04 sur les images à petits objets.
5. Faisabilité : 2 h (filtre guidé), +3 h (SAM si jamais) ; GPU < 1 h. Casse : lisse les petits objets réels -> perte ; seulement par validation OOF.
6. Signal : mIoU OOF ≤ sans raffinement.
7. Test 30 min : appliquer le filtre guidé aux probabilités OOF d'une base quelconque, balayer rayon et eps ; garder si +0,005 OOF.
8. Conformité : OK.

### I11. A priori bayésien de rectangle + géométrie au pixel (arrondi officiel, grille 8, bord d'image)
1. Idée : estimer sur le train une densité de rectangles par méthode (x0,y0,x1,y1 -> noyau gaussien ou histogramme sur (aire, aspect, position, contact avec les bords)) ; rectangle MAP = argmax_R Σ_{p∈R}(e(p)−τ) + log p(R | méthode), recherché sur grille à 4 px puis affiné à 1 px, avec les contraintes dures de l'énoncé (aspect <= 16:1 pour PixelSeal/StegaStamp, <= 2:1 pour TrustMark, ce qui exclut des familles entières de faux rectangles). Convertir le rectangle pixel en masque par `downsample_mask` pour reproduire l'arrondi. Vérifier en train si les frontières de trous tombent sur une grille de 8 px (alignement latent) : si oui, aligner les prédictions de trous sur 8 px.
2. Mécanisme : sur les gros rectangles, chaque cellule de bord compte ; un rectangle MAP donne des bords nets là où un seuillage de carte floue erre de ±1-2 cellules, soit de l'ordre de 1-3 % d'IoU par image. Les contraintes d'aspect éliminent les solutions aberrantes. Pour TrustMark (<= 2:1 et patch entier) l'a priori est très fort.
3. Preuve : [CALC] sur l'effet de bord (rectangle 64×64 cellules, erreur ±1 cellule par côté : IoU 0,94) ; reste [HYP]. Kadane est déjà proposé ; l'apport : prior appris, MAP avec pénalité de complexité, arrondi pixel, grille 8.
4. Gain : +0,02 à +0,05.
5. Faisabilité : 3 h, 0 GPU. Casse : prior trop serré hors distribution (test avec rectangles plus petits) -> prior large (mélange 0,8 appris + 0,2 uniforme).
6. Signal : IoU du rectangle MAP sur les images à rectangle plein < 0,9 en OOF.
7. Test 30 min : sur le train, avec l'évidence de fenêtres simple, comparer seuillage vs rectangle MAP sur les images dont le GT est (presque) un rectangle ; garder si +0,03.
8. Conformité : OK.

### I12. z robuste par image (auto-nul) en plus du nul par messages aléatoires
1. Idée : normaliser la carte d'évidence de chaque image par ses propres statistiques : médiane et MAD de l'évidence sur les 40 % de cellules les plus basses (supposées propres) + nul aléatoire K messages. Seuil = quantile de cette distribution.
2. Mécanisme : la force du filigrane varie par image (JND, texture, contenu, force de l'édition) ; un seuil global sur z mélange des images où le signal est ×3 plus fort. Le z par image rend le seuil comparable et réduit la dépendance à la texture. Complète le z nul du rapport decoders (qui corrige le biais de position/échelle mais pas la force par image).
3. Preuve : [HYP].
4. Gain : +0,005 à +0,02.
5. Faisabilité : 1 h, 0 GPU. Casse : images presque pleines (le 40 % « propre » est en fait filigrané) -> corriger avec la garde I5.
6. Signal : AUC pixelique inchangée.
7. Test 30 min : comparer le mIoU OOF du seuil global vs par image sur la carte de la base. Garder si +0,005.
8. Conformité : OK.

### I13. MRF / coupe de graphe sur l'évidence avec a priori « rectangle moins trous »
1. Idée : minimiser E(y) = Σ_p U_p(y_p) + λΣ_{pq} V(y_p,y_q) + prior(y ⊂ R) sur la grille 128×128, avec U = −log de la probabilité calibrée d'un réseau ou d'I3, V potts à poids sensible aux bords image (I10) ; coupe de graphe exacte (max-flow sur 16 k noeuds : PyMaxflow si installable sur le login sans toucher à torch ; sinon ICM/mean-field ~50 itérations sur GPU). Contrainte : y = 1 uniquement à l'intérieur d'un rectangle R (choisi par I11) pour les types a/b ; libre pour le type c.
2. Mécanisme : impose la cohérence spatiale (taille minimale des trous, absence de poussière) tout en laissant les trous ; plus fiable qu'une suppression de composantes < 4-8 cellules qui efface les vrais petits trous.
3. Preuve : [HYP] (MRF classique sur cartes d'évidence).
4. Gain : +0,01 à +0,03.
5. Faisabilité : 3-4 h ; le mean-field en torch est plus sûr (aucune dépendance). 0 GPU/léger. Casse : sur-lissage de petits objets.
6. Signal : OOF ≤ post-traitement morphologique simple.
7. Test 30 min : mean-field avec Potts sur les probabilités OOF ; balayer λ ; garder si +0,005 vs morpho.
8. Conformité : OK.

### I14. Empreinte spectrale d'échelle : estimer taille et aspect du rectangle depuis le motif
1. Idée : le résidu est calculé à 256 (PixelSeal/TrustMark) ou 400 (StegaStamp) puis étiré à (w,h) du patch ; le spectre d'énergie du résidu filtré (moyenne locale dans une zone filigranée) porte une coupure à ~ (256/w, 256/h) de Nyquist quand w, h > 256 : anisotropie -> rapport d'aspect et échelle de la zone. Régresser (w,h) par un petit CNN spectral ou estimer la coupure.
2. Mécanisme : donne la taille indépendamment de la position, vérifiable ensuite par décodage de crop (I17) : réduit l'espace de recherche des rectangles de 4 à 2 dimensions. Valide seulement pour w ou h > 256 (rectangles de moins de la moitié de l'image : pas de coupure, motif sous-échantillonné).
3. Preuve : [HYP] très spéculatif.
4. Gain : 0 à +0,02 ; surtout accélérateur.
5. Faisabilité : 4 h, 0 GPU ; casse presque certainement pour les petits patchs.
6. Signal : spectres moyens (1D x et 1D y) identiques pour w/h différents.
7. Test 30 min : 40 images train avec w >= 384 vs w <= 192 : comparer le spectre moyen du résidu (HP) à l'intérieur ; si les coupures diffèrent -> poursuivre, sinon tuer.
8. Conformité : OK.

### I15. Cohérence d'effacement (analysis-by-synthesis par ascension de coordonnées)
1. Idée : maximiser la marge du décodeur sur l'image où l'on EFFACE (flou fort ou remplissage moyen) un ensemble Z de cellules supposées propres : Z* = argmax_Z margin(x ⊙ (1−Z) + flou(x) ⊙ Z) − λ|Z|, avec montée de coordonnées sur des blocs de 16×16 cellules par ordre de gain. Le bon Z élimine le bruit des trous, donc la marge monte ; une cellule filigranée effacée la fait baisser.
2. Mécanisme : évidence causale avec le décodeur dans son cadrage normal (utile pour StegaStamp à flatten/STN). Diffère de l'occlusion du rapport decoders (une seule fenêtre à la fois, marge globale très faible) : ici on optimise un ensemble avec un objectif cumulatif, auto-calibré par la marge globale qui monte.
3. Preuve : [HYP].
4. Gain : 0 à +0,03 (StegaStamp).
5. Faisabilité : 4-5 h, 5 h-GPU (~300 passes/image). Casse : coûteux ; solutions adversariales (effacer fait aussi apparaître des artefacts qui augmentent la marge).
6. Signal : AUC pixelique < 0,65.
7. Test 30 min : 10 images StegaStamp ; 100 itérations ; comparer au masque GT. Garder si IoU de Z^c (filigranée) > 0,5 sans autre information.
8. Conformité : OK.

### I16. Empilement tardif (stacking) de vues hétérogènes + TTA par décalage
1. Idée : modèles de natures différentes (I3 sur fenêtres, I1 DEC, I2 Noise2WM, I7 traces, réseau d'appui) produisent des cartes ; un méta-modèle (logistique par cellule avec voisinage 5×5 en entrée ou petit CNN à 8 canaux) entraîné sur les OOF (plis par image) les combine ; TTA par décalages (multiples de 4 px).
2. Mécanisme : les erreurs des vues sont peu corrélées (message vs empreinte vs traces) ; l'empilement sur OOF est plus sûr qu'une somme pondérée à la main.
3. Preuve : [HYP] ; pratique standard.
4. Gain : +0,01 à +0,03.
5. Faisabilité : 3 h (à la fin) ; 1-2 h-GPU. Casse : fuite si les OOF d'étages amont sont refaits avec d'autres plis : fixer UN découpage dès H1 et le garder partout.
6. Signal : OOF empilé < meilleure vue seule.
7. Test 30 min : empiler deux cartes disponibles ; garder si +0,01.
8. Conformité : OK.

### I17. Recherche de rectangle native avec certificat (rappel : déjà au plan, avec deux améliorations)
Déjà proposé (Exp. 0 des rapports decoders/segmentation). Améliorations : (a) recherche SÉPARABLE : le z du crop dépend surtout de (x0,x1) et (y0,y1) de façon quasi indépendante, donc 2 recherches 1D de 128 positions chacune coordonnées alternées au lieu de 4D (~3-5 passes de coordonnées × 2 × 64 candidats = < 400 décodages/image) ; (b) certificat à deux côtés : z(R) élevé ET z(fenêtres de même taille hors R) ~ 0, et z du rectangle élargi (+16 px) plus faible que z(R) (le rectangle parfait est un maximum de z : une dérivée de z par rapport aux bords donne le sens de la correction) ; (c) hypothèse de géométrie image entière 512 à tester en plus (fait n°5). Gain +0,03 à +0,10 si l'échelle native améliore la précision de bits (Exp. 0). Risque et test : voir Exp. 0 des rapports ; ajouter à E3 ci-dessous.

## 2. Classement

Composite = G × P × F̄ / 5, avec F̄ = (F + C)/2 (F = faisabilité en temps réel, C = coût GPU, 5 = bon marché) ; G = gain attendu, P = probabilité de réussite, O = originalité (affichée mais non utilisée). Échelles 1-5. Effort lu dans F̄.

| Rang | Idée | G | P | F | C | O | Composite | Gain mIoU [HYP] | Type |
|---|---|---|---|---|---|---|---|---|---|
| 1 | I5 triage tout-0/tout-1 par règle de rentabilité | 3 | 4 | 5 | 5 | 3 | 12,0 | 0 à +0,05 | détail (métrique) |
| 2 | I4 décision d'IoU espérée par image (postérieur) | 3 | 4 | 4 | 5 | 3 | 10,8 | +0,01 à +0,04 | détail (métrique) |
| 2 | I11 prior bayésien de rectangle, pixel/arrondi/grille 8 | 3 | 4 | 4 | 5 | 3 | 10,8 | +0,02 à +0,05 | détail (a priori) |
| 4 | I17 recherche de rectangle native + certificat | 4 | 3 | 3 | 4 | 3 | 8,4 | +0,03 à +0,10 | méthode |
| 5 | I3 LDA/logistique sur logits signés | 2 | 4 | 5 | 5 | 3 | 8,0 | +0,01 à +0,04 | détail (calibration) |
| 6 | I6 type d'édition + experts (MDL) | 3 | 3 | 3 | 5 | 3 | 7,2 | +0,02 à +0,06 | méthode |
| 7 | I16 stacking de vues + TTA | 2 | 4 | 4 | 4 | 2 | 6,4 | +0,01 à +0,03 | détail (ensemble) |
| 8 | I12 z robuste par image | 2 | 3 | 5 | 5 | 3 | 6,0 | +0,005 à +0,02 | détail (calibration) |
| 9 | I7 traces génératives (VAE/diffusion) | 3 | 3 | 3 | 3 | 4 | 5,4 | +0,02 à +0,07 | pari |
| 10 | I8 doublons inter-images (structure du jeu) | 5 | 1 | 5 | 5 | 4 | 5,0 | 0 ou +0,10 à +0,25 | pari (loterie 10 min) |
| 11 | I1 DEC ré-embedding différentiel | 4 | 2 | 2 | 4 | 5 | 4,8 | 0 à +0,08 | pari |
| 11 | I10 calage frontières (filtre guidé/SAM) | 2 | 3 | 4 | 4 | 3 | 4,8 | +0,01 à +0,04 | détail (post-traitement) |
| 11 | I13 MRF/mean-field rectangle moins trous | 2 | 3 | 3 | 5 | 3 | 4,8 | +0,01 à +0,03 | détail (post-traitement) |
| 14 | I9 co-training/AdaBN sur le test | 2 | 3 | 3 | 4 | 3 | 4,2 | +0,01 à +0,04 | semi-supervisé |
| 15 | I2 Noise2WM (régression dense du résidu) | 4 | 2 | 2 | 3 | 4 | 4,0 | +0,02 à +0,08 | pari |
| 16 | I15 effacement par ascension de coordonnées | 2 | 2 | 3 | 3 | 4 | 2,4 | 0 à +0,03 | pari |
| 17 | I14 empreinte spectrale d'échelle | 2 | 1 | 2 | 5 | 5 | 1,4 | 0 à +0,02 | pari |

Remarque honnête : les gains ne s'additionnent pas (I4/I5/I12 se chevauchent ; I11/I6/I13 aussi). Le cumul prudent des détails (I3 + I4/I5 + I11 + I12) est de +0,04 à +0,10, ce qui est du même ordre qu'une grosse méthode. Les probabilités P sont mes jugements, sans mesure.
Idées « détails astucieux » (>= 6) : I3, I4, I5, I10, I11, I12, I13, I16 (huit) ; plus la mesure E1 des statistiques du train.

## 3. Portefeuille en trois étages

### Étage 1 : base sûre (< 2 h) — objectif : soumission valide + plancher
- H0-0:30 : charger train+test, mesurer E1 (statistiques des 510 masques : fraction tout-1/tout-0/presque, types de forme, boîtes englobantes, `full_watermark`, frontières mod 8) ; calcul de l'évaluateur local avec `miou_image` officiel.
- H0:30-1:30 : tout-0/tout-1 par méthode ; décodage de l'image entière + seuil par méthode ; règle I5 minimale (si bit-acc de l'image entière > seuil -> tout-1 ; sinon tout-0) ; contrôle de format (1020 noms uniques, [1020,128,128], uint8, `weights_only=True`, < 20 Mo).
- H1:30-2 : première soumission de sécurité. Go : fichier valide chargé et mIoU OOF > 0,30.

### Étage 2 : méthode principale (H2 à H12) — décodeur + prior + décision
1. H2-4 : fenêtres multi-échelles + cache de logits + z nul (plan decoders) ; I3 (LDA) et I12 (z par image) en 1-2 h ; AUC par échelle.
2. H4-6 : tomographie/évidence 128² -> I11 (rectangle MAP, aspect, pixel, grille 8) -> I13 (mean-field) pour trous ; I4 + I5 par-dessus. Soumission 1 réelle à H6. Go/no-go H6 : OOF >= 0,55, sinon basculer sur « réseau d'appui » (plan segmentation) en priorité.
3. H6-10 : réseau d'appui (segmentation) alimenté par les cartes ; I6 (type d'édition et MDL), I10 (filtre guidé) ; I16 (stacking). Go/no-go H10 : OOF >= 0,68 ; sinon figer l'étage 2 simple.
4. H10-14 : I9 AdaBN + pseudo-labels certifiés (si temps) ; calibrage final ; soumission 2 à H12.

### Étage 3 : paris à haut rendement (1 à 2 personnes-heures par pari ; en parallèle dès H1)
- Pari A (le plus gros potentiel) : I1 DEC + I2 Noise2WM + I17 (échelle native). Départage par E4/E3 en 30 min ; si AUC pixelique > 0,75 sur ≥ 1 méthode, y mettre une personne de H3 à H10 ; sinon tuer à H2.
- Pari B (loterie 30 min) : I8 doublons inter-images ; coût fixe 30 min, gain potentiel géant. À lancer dès H0:30 par la personne qui attend le GPU.
- Option C : I7 traces génératives, seulement si E5 montre un pic spectral ; à intégrer comme canal de fusion à H8.

Heure par heure (H0 = sam. 17:00, code gelé dim. 12:00 = H19) :
| H | Jalon | Go | No-go |
|---|---|---|---|
| H0:30 | E1 faite | stats exploitables | — |
| H1:30 | E8 doublons, E3 (rectangle natif) | E8 : un voisin > 0,95 dans ≥ 5 % -> exploiter | sinon abandonner E8 |
| H2 | soumission de sécurité ; E4 (DEC) | AUC > 0,75 -> équipe pari A | AUC < 0,6 -> tuer I1 ; passer à I2 si E2 bon |
| H4 | cache de logits + z nul | AUC par échelle > 0,65 | échelle < 0,6 supprimée |
| H6 | soumission 1 réelle | OOF >= 0,55 | sinon bascule réseau d'appui |
| H10 | réseau + I6/I10/I16 | OOF >= 0,68 | figer la base |
| H12 | soumission 2 | OOF > soumission 1 + 0,02 | rester sur 1 |
| H15 | plus aucune nouvelle idée | — | — |
| H17-19 | marge : format, relance depuis un état propre, une dernière soumission | — | — |

## 4. Expériences à lancer en premier (30 min max chacune, par ordre d'exécution)

| N° | Expérience | Mesure | Décide |
|---|---|---|---|
| E1 | Stats des 510 masques (CPU, 10 min) | fraction tout-1/tout-0/presque ; types ; histogrammes de boîte ; frontières mod 8 ; `full_watermark` | I5, I6, I11, intérêt de l'étage 2 |
| E8 | Miniatures 32², matrice de corrélation sur 1 530 images (CPU) | voisins > 0,95 | I8 |
| E3 | Bit-acc du crop de la boîte GT à l'échelle native vs image entière vs fenêtres décalées, aussi en géométrie 512 pour `full_watermark` | écart de bit-acc | I17, fait n°5, composite exact |
| E4 | DEC sur 20 images train avec rectangle GT, par méthode (GPU 1) | AUC pixelique de c(p) | I1 |
| E6 | AUC OOF fenêtre : moyenne vs logistique (par image) | écart d'AUC | I3 |
| E5 | Spectre local de HP(x) en trous vs filigrane vs hors-rectangle (CPU/GPU 1) | pic 1/8 ; AUC | I7 |
| E2 | Noise2WM 10 min, 3 k paires | AUC pixelique | I2 |
| E7 | Règle EV-IoU et tout-0/tout-1 sur les OOF d'une base | mIoU OOF +/− | I4, I5 |

## 5. Hypothèses non vérifiées dont dépend le plan
1. Le motif est mis à l'échelle du rectangle (l'embedder travaille à 256/400 puis rééchelonne) : déduit du code StegaStamp du kit et de ma mémoire de TrustMark/videoseal. Il conditionne I1, I14, I17.
2. Le résidu flottant est accessible pour les trois embedders (sûr pour StegaStamp [KIT], probable PixelSeal, incertain TrustMark).
3. R(x_w, m) ≈ R(c, m) (le résidu dépend peu du contenu déjà filigrané) : conditionne I1.
4. Les éditions recollent l'original hors masque (sinon le filigrane est dégradé partout et I7 change de rôle).
5. La fréquence des images tout-0/tout-1 dans le dataset (I5) : inconnue.
6. Le dataset a une structure de bases partagées (I8) : probabilité ~10 % [HYP].
7. Les chiffres de gain sont des fourchettes de jugement, non mesurées. Le numéro d'arXiv de Nowozin (1402.1782) est à revérifier avant de le citer.
8. Les débits GPU (embeds/s, passes/s) sont des estimations non mesurées.
9. Je n'ai pas lu watermarks.py de videoseal ni de trustmark (seulement les wrappers du kit) : les sorties internes (résidu flottant, `imgs_w`) sont à confirmer dans le code au premier lancement.

## 6. Conformité
Toutes les méthodes utilisent uniquement : les images et messages fournis (train/test), les embedders et décodeurs du kit, des modèles publics (VAE, SAM, LaMa) et la structure connue de la métrique. Aucune requête au serveur d'évaluation autre que les soumissions normales ; aucune tentative d'extraire les labels du test via le classement. I8 compare des images publiques entre elles (analyse du jeu de données fourni), sans usage des travaux d'une autre équipe. Le seuil et les modèles se règlent sur les plis OOF du train, pas sur le classement public (30 %).

## Sources
- Énoncé et kit : /Users/titouandestors/Hack European Championship/EAC/reports/sources/Watermark_Localization.txt ; .../reports/sources/t1/starter_kit/wmloc/{watermarks,utils,data}.py ; README.txt.
- Rapports : research/t1-decoders.md, research/t1-segmentation.md, reports/t1-watermark-etat-de-lart.md, research/intelligence.md.
- WAM https://arxiv.org/abs/2411.07231 ; TruFor https://arxiv.org/abs/2212.10957 ; TGIF2 https://arxiv.org/html/2506.05466v2 ; AdaBN https://arxiv.org/abs/1603.04779 ; Nowozin IoU optimal decisions (numéro à revérifier) ; PixelSeal https://arxiv.org/abs/2512.16874 ; TrustMark https://arxiv.org/abs/2311.18297 ; StegaStamp https://arxiv.org/abs/1904.05343.
