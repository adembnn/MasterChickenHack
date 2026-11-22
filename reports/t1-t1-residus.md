# T1 · session t1-residus : ce que donne la connaissance du tampon et du message, sans gros entraînement
Nuit du 21 au 22/11/2026. Décision débloquée : quoi reprendre dans l'assemblage (rectangles TrustMark calés au pixel, trous des pleines TrustMark), et quelles pistes fermer.
**Bilan TrustMark, base réseau `rangP-reel-mixr` : hors réserve 0,6533 → 0,7187 (rectangles) → 0,7313 (+ pleines) ; réserve 0,6223 → 0,7256 → 0,7394.**
Scripts : `E:\EAC-lab\t1s\t1-residus\` (copie `h200b:~/eac/t1/t1-residus/`). Fichiers livrés : `h200b:/tmp/eac-t1-partage/t1-residus/`.
[MESURÉ] = chiffre collé d'une sortie de commande sur les vraies images du train. [FABRIQUÉ] = mesuré sur des images que j'ai tamponnées moi-même.

Mots (une fois) : **tampon** (embedder) = le réseau qui ajoute le filigrane ; **décodeur** = le réseau qui relit le message ; **résidu** = image filigranée
moins image propre ; **gabarit** = le résidu que le tampon produirait pour notre message ; **cadre** = le rectangle exact sur lequel le tampon a été
appliqué (il redimensionne ce rectangle à sa taille de travail, donc le motif est calé dessus) ; **z** = écart au hasard en écarts-types :
`z = Σ(2m−1)·l / √Σl²` sur les logits `l` du décodeur et les bits `m` du message (hasard : z ≈ 0 ± 1 ; maximum 10 pour 100 bits) ;
**R3** (« re-tamponnage ») = énergie de ce que le tampon ajouterait au rognage avec le VRAI message, divisée par la même énergie avec des messages au hasard ;
**AUC** = chance qu'une case filigranée ait un score plus haut qu'une case propre (0,5 = hasard) ; **réserve** = images du train d'index % 5 == 0, jamais utilisées pour régler.

## 1. Résultat principal : rectangles TrustMark trouvés par le décodeur puis calés au pixel
Un quart des images (les partielles « rectangle », issues de l'outpainting) portent un filigrane calé sur un rectangle. Méthode en trois temps :
1. **Tri** : bits justes de l'image entière < 0,75 → image « partielle » [MESURÉ : 1 pleine sur 85 et 3 partielles sur 85 mal triées].
2. **Recherche de boîte** (`bs.py`) : ~13 000 rognages par image (tailles en progression ×1,3, pas = moitié de la taille), on garde le z le plus haut,
   puis montée de coordonnées (on bouge un bord à la fois tant que z monte). On garde la boîte si z > 6,5.
3. **Calage au pixel** (`refine.py`) : autour de la boîte, on cherche le rectangle qui minimise R3, puis on le rétrécit de 3-4 px par bord.

Sorties collées :
```
ROUTEUR (bits justes image entière < 0,75 = partielle) : pleines prises pour partielles 1/85 ; partielles prises pour pleines 3/85
rectangles n=43 | z trouvé : médiane 9.7 [q10 7.3 ; q90 9.8] | IoU boîte : moyenne 0.742, médiane 0.787 | mIoU officiel si masque = boîte : 0.859 (réserve 0.827, n=13) ; si tout-0 : 0.460
objets     n=42 | z trouvé : médiane 4.2 [q10 3.7 ; q90 6.1] | mIoU officiel si masque = boîte : 0.416 (réserve 0.411, n=8) ; si tout-0 : 0.441
REFINE trustmark train n=41 : rectangles vrais n=40 | R3 au cadre affiné : médiane 0.52 ; R3 < 0,7 : 31 | R3 à la boîte du décodeur : médiane 0.99
mIoU officiel des rectangles (masque = boîte) : décodeur seul 0.886 -> affiné R3 puis rétréci de [-3 -3  3  3] (réglé hors réserve) 0.943 | réserve : 0.887 -> 0.970 (n=11)
```
Fusion avec le réseau (`fuse.py` : le masque du réseau `rangP-reel-mixr` est remplacé par le rectangle quand z > 6,5).
Avec les boîtes du décodeur seules (temps 1 et 2) :
```
RÉSERVE                    n= 39  avant 0.6223  après 0.7011  écart +0.0788  (erreur type de l'écart 0.0268 ; images modifiées 11, améliorées 10, dégradées 1)
hors réserve               n=131  avant 0.6533  après 0.6962  écart +0.0428  (erreur type de l'écart 0.0122 ; images modifiées 30, améliorées 24, dégradées 6)
```
Avec le calage au pixel (temps 3, cadre rétréci de 3,5 px, gardé si R3 < 0,7 ; sinon boîte du décodeur) : version livrée.
```
FUSE trustmark base=pred_train_rangP-reel-mixr_trustmark.npz | rectangles posés sur le train : 41 (dont affinés R3 : 31), seuil z > 6.5
  RÉSERVE                    n= 39  avant 0.6223  après 0.7246  écart +0.1023  (erreur type de l'écart 0.0319 ; images modifiées 11, améliorées 11, dégradées 0)
  réserve, partielles        n= 21  avant 0.6321  après 0.8221  écart +0.1900  (erreur type de l'écart 0.0522 ; images modifiées 11, améliorées 11, dégradées 0)
  hors réserve               n=131  avant 0.6533  après 0.7066  écart +0.0532  (erreur type de l'écart 0.0139 ; images modifiées 30, améliorées 25, dégradées 5)
  test : 340 images, rectangles posés 77 (dont affinés R3 : 50)
```
**Lecture corrigée par la critique adverse** (`reports/critique/t1-residus-tour1.md`, chiffres recalculés par un relecteur séparé : exacts) :
le gain à attendre au test est celui du hors réserve, **+0,05 à +0,065 sur TrustMark** (≈ +0,02 sur la moyenne des trois), pas +0,10. La réserve est
favorable par hasard : ses 11 rectangles sont tous calés au pixel, contre 20 sur 30 hors réserve et 50 sur 77 au test ; un rectangle calé rapporte
+0,34, une boîte non calée +0,04 seulement. Rien n'est réglé sur la réserve (rétrécissement 3 à 4 px, seuil R3 0,5 à 0,7 : réserve 0,719 à 0,725).
Version 2 (`fuse2.py`), après critique : tri pleine/partielle sans la vérité, rattrapage des cadres non calés (`refine2.py`), et quand le cadre
reste non calé on pose la boîte du décodeur rétrécie de 3 px (règle choisie hors réserve parmi 7) :
```
FUSE2 trustmark base=pred_train_rangP-reel-mixr_trustmark.npz boîtes=boxes_trustmark_train_g13k_r3c.json | avant : hors réserve 0.6533, réserve 0.6223
  A boîte si non calé                calés 36, replis  4 | HORS RÉSERVE 0.7168 (+0.0635 ± 0.0131 ; dégradées 1) | réserve 0.7256 (+0.1033)
  B réseau si non calé               calés 36, replis  0 | HORS RÉSERVE 0.7112 (+0.0579 ± 0.0127 ; dégradées 0) | réserve 0.7256 (+0.1033)
  E boîte rétrécie 3 px si non calé  calés 36, replis  4 | HORS RÉSERVE 0.7187 (+0.0653 ± 0.0134 ; dégradées 1) | réserve 0.7256 (+0.1033)
```
Les 3 rectangles non trouvés font moins de 35 px et sont illisibles même à la vraie boîte. Points faibles relevés par la critique : les z des objets
montent jusqu'à 6,4 et 7,6 (1 objet reçoit un rectangle au train, −0,60 sur cette image) ; au test, 2 boîtes posées couvrent plus de 60 % de l'image.

## 2. Deux faits sur les données, utiles à toutes les sessions
- **Le masque d'un rectangle = le cadre du filigrane rétréci de 3 à 4 px par bord** [MESURÉ sur 31 rectangles : écart médian cadre − masque (−3, −3, +3, +3),
  indépendant de la taille]. R3 vaut 0,875 à la boîte du masque et 0,41 à la boîte agrandie de 4 px [16 rectangles]. Pour fabriquer des rectangles
  fidèles : tamponner un rognage, ne recoller que l'intérieur rétréci de 3-4 px. Même marge pour StegaStamp (R3 0,756 au masque, 0,622 à +4 px).
- **Le tampon reconnaît son propre filigrane** : re-tamponner une image déjà filigranée avec le même message ajoute 2 à 3 fois moins d'énergie
  [FABRIQUÉ, image entière : R3 = 0,28 TrustMark / 0,62 PixelSeal / 0,59 StegaStamp dans la zone filigranée, 0,95-1,05 ailleurs]. Ce rapport résiste au
  JPEG 70, à un bruit de 3 niveaux et à un changement de taille (reste 0,27-0,30) mais exige le bon cadre : à l'échelle d'un rectangle de ~130 px,
  0,37 au cadre exact, 0,64 décalé de 1 px, 1,0 à 4 px [FABRIQUÉ]. C'est ce qui permet le calage au pixel.

## 3. Netteté des décodeurs autour de la vraie boîte (rectangles réels, z moyen) [MESURÉ]
| Perturbation de la vraie boîte | TrustMark (max 10) | PixelSeal (max 16) | StegaStamp (max 10) |
|---|---|---|---|
| exacte | 9,52 | 10,74 | 8,07 |
| décalée de 4 px | 9,45 | 9,95 | 3,48 |
| décalée de 8 px | 9,04 | 8,19 | 0,40 |
| agrandie de 8 px par bord | 6,65 | 8,03 | 4,33 |
| rétrécie de 8 px par bord | 5,02 | 9,61 | 2,78 |
Conséquences : TrustMark et PixelSeal se cherchent à la grille mais ne se calent qu'à ±6 px (même erreur sur rectangles fabriqués, et avec 5 critères :
z, marge, log-vraisemblance, milieu de plateau, médiane de 8 montées : IoU 0,81 à 0,87). StegaStamp est précis mais introuvable à l'aveugle : z tombe à
2,9 pour un décalage de 5 % de la largeur (taille tolérée à ±20 %) ; il lui faut un point de départ (carte d'un réseau), ce que fait `boxnet.py`.
Astuce StegaStamp : le graphe ONNX du kit recalcule l'encodeur à chaque décodage ; `rlib.ss_decoder_only(wm)` écrit un graphe « décodeur seul » (logits identiques à 1e-5).

## 4. Les pistes R1 à R5 du brief, une par une
| Piste | Verdict | Chiffre |
|---|---|---|
| R1 filtre adapté (corrélation image × gabarit), cadre image entière | ne marche pas sur le réel | AUC pleines 0,50-0,57 (16 images TrustMark, 300 variantes : 10 débruiteurs dont l'effaceur TrustMark, pondération, gabarit itéré, 3 lissages) ; meilleur mIoU réserve 0,32 |
| R1 bis corrélation croisée pour trouver un rectangle StegaStamp | ne marche pas | bonne position 35 % avec le gabarit idéal, 5 % avec un gabarit sur fond gris (20 rectangles) |
| R2 résidu estimé sans message (effaceur TrustMark, flous, médian) | ne marche pas | compris dans R1 : effaceur AUC 0,49 |
| R3 re-tamponnage | **marche pour les rectangles** (calage au pixel) ; moyen pour les objets et les pleines (signal réel, pas de gain de score) | rectangles : §1 ; objets et pleines : §5 |
| R4 inversion différentiable | non codée | remplacée par R3, qui en est la forme directe |
| R5 vote par position / z exact des logits | **marche comme critère de boîte** | §1 et §3 ; routeur pleines/partielles : 4 erreurs sur 170 |
Pourquoi R1 échoue alors que le contrôle de bon sens passait [FABRIQUÉ : cosinus entre le gabarit recalculé sur l'image filigranée et le vrai résidu
0,76 PixelSeal / 0,79 TrustMark / 0,83 StegaStamp ; mon gabarit sans arrondi reproduit `wm.embed` à 0,96-0,99] : (a) le gabarit suit le contenu de l'image
(deux messages sur la même image : cosinus 0,35 à 0,60), donc il se corrèle à l'image elle-même, filigranée ou non (StegaStamp : corrélation 0,3 à 0,6
dedans comme dehors) ; (b) le cadre réel n'est pas l'image entière (pleines : toile à marges inégales de 0 à ~60 px ; rectangles : +3-4 px).

## 5. Au-delà du brief : objets (pas abouti) et trous des pleines (petit gain mesuré)
- **Carte locale de R3 pour les partielles « objet »** (TrustMark, 12 images) : le filigrane de l'objet est au cadre d'une toile décalée de 0 à ~130 px.
  Avec la toile trouvée grâce au vrai masque (plafond) : rapport 0,57 dedans / 1,06 dehors, AUC moyenne par image 0,815, mIoU 0,52 (tout-0 : 0,456).
  À l'aveugle (toile cherchée par le creux de la carte) : AUC 0,54. Fermé : le signal existe mais la toile ne se trouve pas sans le masque.
- **Carte locale de R3 pour les trous des pleines** [MESURÉ]. Premier essai, même marge des quatre côtés (`r3auc.py`, 20 pleines par filigrane, 0 à 24 px) :
  AUC des trous 0,50 TrustMark / 0,58 PixelSeal / 0,63 StegaStamp : rien. Erreur de cadre : t1-assemblage a vu que les marges sont INÉGALES par côté.
  Second essai, 4 marges cherchées séparément (`pleine.py`, grille 0 à 40 px puis descente au pixel), 82 pleines TrustMark du train :
  ```
  toile trouvée (R3 < 0,6) : 39/42 ; R3 médian 0.34 ; marges médianes [ 6. 10. 16.  8.], somme haut+bas médiane 28, gauche+droite 28
  lissage   10 : AUC des trous moyenne 0.763, médiane 0.812, > 0,8 : 22/42      (lissage 3 : 0.706 ; lissage 24 : 0.752)
  ```
  Fusion avec le réseau, re-tamponnage seul : règles simples (seuil, « ou ») -0,003 à -0,12 ; régression logistique par case : pleines réserve +0,007 à +0,009.
  Mesures des cartes (`pz.py`, `pleine_ps.py`) :
  ```
  PZ train n=85 : lissage 10 : AUC des trous moyenne 0.786, médiane 0.804 | toile trouvée seulement : moyenne 0.809, médiane 0.829 | z médian dedans -2.0, trou -0.6
  RESULT pixelseal RÉEL n=6+6 | R3 à la toile trouvée : médiane 0.74 / 0.68 | AUC des trous moyenne 0.74 / 0.69 (pleines fabriquées témoins : 0.85)
  ```
  Lecture : (a) la toile réelle des pleines est un CARRÉ de côté 512 + s (s : q10 5 px, médiane ~32, q90 ~63, max ~84 ; sommes des marges verticale et
  horizontale corrélées à 0,96) dont l'image est un rognage placé au hasard ; 19 % des pleines sont au cadre exact de l'image. Transmis à t1-reel, qui
  l'a repris dans son générateur. (b) Au bon cadre le tampon reconnaît son filigrane (0,34 contre 0,9 à 1,2 au cadre image) et la carte voit les trous
  (AUC ~0,8), mais il le « reconnaît » en partie jusque dans les trous (z -0,6 et non 0) : seul, ce signal ne rapporte que +0,004 à +0,017.
  **Second signal qui le complète** (`rm.py`) : au cadre de la toile, l'effaceur de TrustMark estime le filigrane présent r^ = toile − effaceur(toile), qu'on
  corrèle au motif attendu t0 = tampon(image nettoyée, m) : c'est le filtre adapté R1/R2 refait au bon cadre (seul : AUC des trous 0,74 ; |r^|² : 0,67).
  Variantes mesurées (`pinv.py`) : vrai message contre message inversé 0,78 (égal à la carte de base) ; corrélation signée vrai / inversé 0,57.
  Fusion finale à deux signaux (`pfuse3.py`, régression logistique par case apprise sur les pleines hors réserve, prédictions hors-pli) :
  ```
  PFUSE3 base=pred_train_boxr3v2-rangP-reel-mixr_trustmark.npz | pleines TrustMark n=85 (toile fiable 75)
    pleines hors réserve (hors-pli) : 0.6023 -> 0.6270 (+0.0247 ± 0.0057 ; améliorées 39, dégradées 14)
    pleines RÉSERVE                 : 0.6109 -> 0.6408 (+0.0299 ± 0.0194 ; n=18 ; améliorées 8, dégradées 6)
    TrustMark entier (170 images, fichier d'entrée -> sortie) : hors réserve 0.7187 -> 0.7313 | RÉSERVE 0.7256 -> 0.7394 (n=39)
    test : 176 images classées pleines, 138 avec toile fiable et masque refait ; part propre prédite : avant 0.109, après 0.110 (train, vérité : 0.091)
  ```
  Ajouter les variantes à la fusion n'est pas stable (hors-pli +0,030 mais réserve -0,004) : écartées. Accepter les toiles moins nettes (R3 < 0,7 ou 0,8
  au lieu de 0,6) ne change rien de mesurable (hors-pli +0,030, réserve +0,021). **Le gain dépend de la base** : sur le meilleur réseau de t1-reel (`f_s1`,
  hors-pli vérifié), t1-assemblage mesure avec ce même script +0,008 hors réserve et +0,009 en réserve sur les pleines (`f_s1a` : +0,007 et +0,016) :
  ce réseau trouve déjà une partie des trous que les cartes révèlent. Intégré dans son candidat 24 (réserve 0,7536, TrustMark 0,7708).
  PixelSeal : toile trouvée 9 fois sur 12, AUC des trous 0,69 à 0,74 (0,85 même sur fabriqué), pas d'effaceur : non poursuivi. StegaStamp, objets : non faits.
- **Affiner la boîte avec le décodeur seul** : plafond ±6 px (§3).
- StegaStamp et PixelSeal par ma propre recherche de boîte : arrêtés, deux autres sessions le font (`boxnet.py`, `boxsearch.py`).

## 6. Fichiers livrés
- `boxes_trustmark_{train,test}_g13k.json` : par image partielle, `box` = y0, x0, y1, x1 et `z`. `..._r3.json` : + `frame` (cadre au pixel) et `r3`.
- `pred_{train,test}_boxr3v2-rangP-reel-mixr_trustmark.npz` et `..._boxr3v2-partage_trustmark.npz` : format commun (`names`, `logits` 128x128, > 0 = filigrané),
  = prédictions du réseau avec les rectangles calés au pixel, version 2 (à prendre). Le train est hors-pli comme sa base.
  (`..._boxr3-...` : version 1 ; `..._box-...` : sans calage.) `boxes_trustmark_{train,test}_g13k_r3c.json` : cadres après rattrapage.
- **À prendre** : `pred_{train,test}_boxr3v2pl-rangP-reel-mixr_trustmark.npz` = rectangles calés + trous des pleines (train hors-pli).
- Cartes des pleines (pour refaire la fusion sur une autre base ou servir de canal d'entrée) : `maps_trustmark_{train,test}_{pz,rm}.npz`,
  `maps_trustmark_train_pleine_s?of2.npz`, `maps_trustmark_test_pleine_s?of4.npz` (clé `frame` = toile par image ; `r3` < 0,6 = fiable : 75/85 et 138/176).
- Cartes du premier filtre adapté au cadre image : non livrées (AUC 0,5).

## 7. Limites
- Réserve partielles TrustMark : 21 images, donc ±0,045 sur l'écart ; l'écart hors réserve (+0,043 ± 0,012 sur 131 images) confirme le sens.
- Le calage R3 trouve 31 cadres sur 40 au premier passage (train) et 50 sur 77 (test) ; les autres gardent la boîte du décodeur. Un rattrapage
  (`refine2.py`, recherche proportionnelle à la taille de la boîte) récupère les grands rectangles où le décodeur se trompe de 20 à 60 px
  (exemples : R3 0,86 → 0,24 ; 0,80 → 0,23, cadre = masque ± 3 px) ; au test il porte les cadres calés de 50 à 70 sur 75 rectangles posés ; la version livrée l'inclut.
- R1 n'a été balayé en grand que sur 16 images TrustMark ; PixelSeal et StegaStamp sont couverts par le diagnostic (6 images chacun) et par t1-pixelseal (AUC 0,48-0,50).
- Machines : Jülich est tombé trois fois (code MFA) ; tout a fini sur les deux H200 partagées, en petits lots.
