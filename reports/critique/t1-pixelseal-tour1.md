# T1 PixelSeal : critique adverse du localisateur dense (`loc.py`, `locnet.py`), tour 1
Décision débloquée : quoi corriger avant de lire les 5 variantes de `loc1.sbatch`, et quoi lancer ensuite pour dépasser 0,573 sur la réserve.
Légende : [MESURÉ] chiffre déjà sorti d'une commande ; [DÉDUIT du code] lu ligne à ligne ; [À VÉRIFIER] hypothèse avec son test.
Termes : *mIoU* = moyenne, par image, du recouvrement (intersection / union) de la classe propre et de la classe filigranée ;
*AUC* = probabilité qu'un pixel filigrané ait un score plus haut qu'un pixel propre (0,5 = hasard), indépendante du seuil ;
*champ récepteur* = taille de la zone d'image qu'un point de la carte « voit » ; *JND* = carte « juste perceptible » qui règle l'amplitude
du filigrane pixel par pixel ; *raccourci* = indice faux que le réseau apprend parce qu'il est corrélé à l'étiquette dans les données fabriquées.

## a) Bugs et défauts

1. **Raccourci certain dans les « objets » fabriqués** (`loc.py:52`, `:60`, `:89`) [DÉDUIT du code]. Le fond d'un exemple « objet » est une
   image partielle TrustMark ou StegaStamp, et le masque est SA forme d'objet. Or dans cette image, le filigrane TM/SS est exactement DANS
   cette forme. L'étiquette PixelSeal coïncide donc avec un second résidu, 1,3 niveau (TM) ou 6,5 niveaux (SS) [MESURÉ, faits t1-reel §4],
   soit 2 à 8 fois plus fort que PixelSeal (0,8). 15 % des exemples fabriqués apprennent « résidu SS/TM ⇒ filigrané », indice absent du test
   PixelSeal. Test de 5 min (inférence seule) : passer le modèle `A_base` sur 20 partielles-objets TM/SS hors réserve SANS PixelSeal ;
   si l'IoU entre prédiction et forme dépasse 0,2, le raccourci est confirmé.
   Correctif : formes d'objets COCO (`annotations/instances_val2017.json`, à vérifier dans `shared_data/coco`) posées sur leur propre image
   COCO, ou à défaut objets TM seulement (résidu plus faible), jamais SS.
2. **`predict.py:28` ignore `agree` et `res` de chaque run** [DÉDUIT du code]. `Loc(...)` est construit avec `agree=1` et un seul `--res` et
   `--ppinit` pour tous les runs. `C_sansbits` (agree 0, ppinit 0) et `D_res512` (res 512) seraient prédits avec la mauvaise architecture,
   sans erreur au chargement (mêmes poids). Correctif : écrire `vars(a)` dans `run_*/args.json` à l'entraînement et le relire ici.
3. **`loc.py:195` : `pred_train.npz` contient des logits vus à l'entraînement** (rôle 2, ~102 images) [DÉDUIT du code]. Le livrable exigé
   est hors-pli. Le tableau `role` est sauvé, mais l'orchestrateur qui mélange les méthodes ne doit lire que `role != 2`.
   Ne passer que par `predict.py --folds`.
4. **Choix de variante sur la réserve** (`loc.py:187-191`) [DÉDUIT du code]. Chaque évaluation imprime la réserve ; choisir parmi
   5 variantes × 8 étapes sur 30 à 34 images, c'est régler sur la réserve (bruit ±0,03). Classer les variantes sur le PLI
   (`res["pli"]`), lire la réserve une seule fois à la fin. Note : `i % 5 == 0` sur 170 PixelSeal donne 34 images selon le brief, 30 selon
   la mesure annoncée : [À VÉRIFIER] (`role==0` imprimé ligne 69).
5. **Mémorisation du réel** [DÉDUIT du code] : 4 vraies images par lot × 4000 pas sur ~102 images, soit ~157 passages chacune.
   L'écart `appris` − `pli` mesurera la mémorisation ; au-delà de 0,15, réduire `--real` à 2 ou arrêter à 2000 pas.
6. **Défauts mineurs** [DÉDUIT du code] : `loc.py:125-126` le décalage par bords répétés crée des bandes de 8 à 16 px sans motif valable
   mais étiquetées comme leur bord (jusqu'à 4 cases 128 de faux positifs à l'entraînement) : remplacer par `reflect` ou masquer la perte
   dans la bande. `loc.py:145` `> 0.5` contre `>= 0.5` ligne 141 (sans effet pratique). `loc1.sbatch:20-21` : B et E partagent le GPU 1,
   donc E et B auront moins d'itérations par seconde, mais même nombre de pas : comparaison juste, durée seulement.
7. **Ce qui est juste** [DÉDUIT du code] : polarité (masque 255 = filigrané, `PG` officiel, `lg > 0` = filigrané) ; grille 512 → 128 par la
   règle officielle (`:141` = `utils.downsample_mask`) ; cases 64 (res 256, pas 4 puis ×2) alignées sur les blocs 8 px de l'image, et
   décalages réels par multiples de 8 px qui respectent cette grille ; le tronc reçoit exactement `preprocess` (`x*2-1`, sans norme finale,
   `convnext.py:134`) ; réserve et pli exclus des fonds, trous et formes (`:50`). Pas de fuite trouvée.

## b) Fidélité des exemples fabriqués

- **Trous des pleines** : les vrais trous sont des taches lisses aléatoires, sans lien avec le contenu (`planche_train.png`), donc
  la recette « trous réels déformés sur un autre fond » est fidèle [MESURÉ visuellement]. Bonne nouvelle : aucun indice de contenu à exploiter.
- **Objets** : la recette « cadre image entière puis découpe » est confirmée [MESURÉ, diag §6], mais le raccourci du §a1 la ruine.
- **Rectangles** : recette fidèle (filigrane au cadre du rectangle). Écart possible : dans le réel le rectangle est une image source
  ENTIÈRE réduite puis entourée d'outpainting ; sa densité de détails peut différer du pourtour. Le fabriqué n'a aucune différence de
  contenu, donc le réseau n'apprend pas cet indice, gratuit dans le réel [À VÉRIFIER : AUC d'une carte d'énergie passe-haut sur les 43 rectangles réels].
- **Fonds** : 30 % de COCO (photos JPEG à blocs 8×8), alors que le test est fait d'images générées lisses (planche) ; et les fonds
  « réels » portent un résidu TM ou SS partout. Les deux ajoutent un bruit absent du test PixelSeal. Préférer les fonds TM (résidu faible).
- **Proportions** : 50 % pleines / 20 % rectangles / 30 % objets, proche du réel (moitié-moitié, partielles 43/42) [MESURÉ].

## c) La localité n'est pas forcée

- [DÉDUIT du code] `locnet.py:19-20` : la carte la plus fine (64×64) = étage 1 (champ récepteur ~150 px à 512) + l'étage 4 global
  remonté. Le réseau peut tout lire depuis le global. `ppinit` (`:22-25`) ajoute en plus les votes de la tête d'origine, prouvés non locaux
  (0,788 dans le trou contre 0,799 hors du trou) [MESURÉ diag §4].
- **La cible des bits est impossible localement** (`loc.py:146`) : demander les 256 bits justes dans chaque case de 8 px alors qu'un carré
  de 128 px ne donne que 0,59 de bits justes [MESURÉ diag §5]. La seule façon de baisser cette perte : décoder globalement puis « éteindre »
  les cases propres, c'est-à-dire refaire la segmentation dans la tête des bits. La perte des bits devient une perte de masque bruitée.
  La variante C (`bitw 0`) dira si elle sert ; si C ≥ A − 0,02 sur le pli, l'abandonner.
- **Pour un détecteur vraiment local (32-64 px)** : ne pas décoder, détecter la PRÉSENCE du résidu. (1) Entraîner sur des rognages
  aléatoires de 96 à 160 px à 512, étiquette = carte du masque ; le champ récepteur est borné par construction, et l'inférence se fait
  en fenêtre glissante. (2) Ou couper la pyramide aux étages 1 et 2 (supprimer `fs[2]`, `fs[3]` du chemin descendant). (3) Entrée en
  fp32 sans réduction : 2 couches 3×3 sans pas sur la luminance passe-haut (style SRNet), plus le canal JND du §d.

## d) Signaux physiques inexploités

Le modèle du filigrane, lu dans `wam.py:168-197`, `jnd.py:80-113`, `pixelseal_card.yaml` [DÉDUIT du code] :
`x_w = arrondi( x + 0,2 · H(x) · agrandi×2(u) )`, avec `u` sortie tanh (|u| ≤ 1) d'un U-Net en luminance à 256, et `H` la carte JND
calculée à 512 sur la luminance de l'image propre (`heatmaps` puis `/255`).
1. **Amplitude prévisible** : `A(p) = 0,2·H(p)` (en niveaux) se calcule sur l'image observée (le résidu change à peine `H`).
   Près du gris moyen `H` tend vers 0 (masquage de luminance, `jnd.py:67`), donc `A < 0,5` et l'arrondi efface tout : c'est l'origine
   probable des 39 % de pixels inchangés [MESURÉ diag §2]. Dans ces zones aucune méthode ne peut voir le filigrane : il faut les remplir
   par le contexte (morphologie) et les retirer de la perte. Donner `A` en canal d'entrée et normaliser le passe-haut par `A`.
2. **Gris exact** : l'image propre est entière ; ajouter la même valeur δ aux trois canaux puis arrondir donne le MÊME entier sur R, V, B.
   La chrominance (R−V, V−B) n'est donc jamais touchée : le passe-haut de la chrominance est un témoin de la texture propre, à soustraire
   de celui de la luminance. Indice local gratuit.
3. **Piste hors des réponses évidentes : filtre adapté différentiel avec messages témoins.** Le filtre adapté a échoué (AUC 0,48-0,50)
   [MESURÉ diag §3] parce que le gabarit `r̂ = embed_float(x, m) − x` dépend surtout de l'image : sa corrélation avec la texture propre crée
   un biais aussi grand dans les zones propres. Mais deux messages sur la même image ne se corrèlent qu'à 0,35 : la part propre au message
   (0,65) est indépendante de l'image propre. Statistique locale :
   `T(p) = < passe-haut(x), r̂_m − moyenne_k r̂_mk >` sur une fenêtre de 32 ou 64 px, avec 8 messages témoins aléatoires `mk`, divisée par `A`.
   Dans une zone propre l'espérance est nulle ; dans une zone filigranée elle vaut ~0,65·0,78 de l'énergie du vrai résidu.
   Ordre de grandeur [DÉDUIT] : rapport signal/bruit ≈ √N · σ_résidu / σ_image ≈ 64 × 0,04 × 0,5 ≈ 1,3 par bloc de 64 px.
   Faible seul, mais c'est un canal vraiment local et calé sur le vrai message.
   **Test de 5 min** (nœud de login, 20 vraies images PixelSeal hors réserve, 9 `embed` par image) : AUC de `T` par bloc de 32 et de 64 px
   contre le vrai masque. AUC > 0,70 à 64 px : l'ajouter en canal du réseau. AUC < 0,60 : abandonner la piste.

## e) Post-traitement par population, sans régler sur 30 images

- Population connue presque sans erreur par les bits justes de l'image entière, seuil 0,75 (pleines 0,98, partielles 0,53) [MESURÉ].
- **Le « billet de loterie » est presque gratuit** [DÉDUIT du score] : dans une pleine avec un trou h = 6 %, tout prédire filigrané donne
  (1−h)/2 ≈ 0,47 ; prédire une tache propre de 6 % au MAUVAIS endroit donne ≈ 0,44 (−0,03) ; la placer à moitié juste donne ≈ 0,62.
  Même calcul pour les partielles (tout à 0 ≈ 0,49, tache fausse ≈ 0,47). Donc : toujours prédire au moins une région de la classe
  minoritaire, la composante connexe la plus probable, de taille = médiane de la population (pleines : 6 % propre ; partielles : 3 % filigrané).
- Seuil par image qui maximise le mIoU ESPÉRÉ sous les probabilités du réseau (IoU souple), au lieu d'un seuil global : aucun réglage.
- Partielles quasi rectangulaires (rectangularité > 0,9) : remplacer la tache par le rectangle voisin qui maximise les bits justes du rognage,
  seulement si ceux-ci dépassent 0,65 (rectangles exacts PS 0,69 en moyenne, petites boîtes illisibles) [MESURÉ t1-reel §2].
- Deux nombres à régler au plus (taille minimale par population), sur les prédictions hors-pli des 4 plis (~136 images), jamais sur la réserve.

## Expériences suivantes (5 au plus, classées)

| # | Expérience | Coût | Chiffre qui décide |
|---|---|---|---|
| 1 | Post-traitement par population : tache minoritaire forcée + seuil par image au mIoU espéré, sur les `pred_train` hors-pli existants (v2dec puis `loc`) | 20 min CPU, aucun entraînement | gain ≥ +0,03 sur les 4 plis hors réserve, puis lecture unique de la réserve |
| 2 | Test du raccourci objet (inférence sur 20 objets TM/SS sans PixelSeal), puis relance de A avec formes COCO ou objets TM seulement | 5 min + 1 run | IoU forme/prédiction > 0,2 ⇒ raccourci ; la relance doit gagner ≥ 0,03 sur les partielles du pli |
| 3 | Filtre adapté différentiel à messages témoins, normalisé par `A` (§d3) | 5 min login | AUC par bloc de 64 px > 0,70 ⇒ canal d'entrée ; < 0,60 ⇒ abandon |
| 4 | Détecteur local : rognages de 96 à 160 px, pyramide coupée à l'étage 2, `bitw 0`, canaux `A` et chrominance | 1 job (4 GPU, 4 tailles de rognage) | pleines du pli ≥ +0,04 sur `A_base` |
| 5 | Corriger `predict.py` (args par run), puis ensemble hors-pli F1-F4 de la meilleure variante choisie sur le PLI | 1 job + 10 min | ensemble ≥ meilleur run seul + 0,02 sur la réserve ; sinon garder le run seul |
