# T1 · PixelSeal : localiser le filigrane (session t1-pixelseal)

État au 21/11, 23 h 30. Décision débloquée : quoi soumettre pour les 340 images PixelSeal du test, et où il reste (ou non) du score.
Scripts : `E:\EAC-lab\t1s\t1-pixelseal\` ; copies sur Jülich `adem/t1-pixelseal/` et sur `h200b:~/eac/t1/t1-pixelseal/`.

## Résultat
Score = mIoU officiel (`wmloc.utils.miou_image` : par image, moyenne du recouvrement intersection/union de la classe propre et de la
classe filigranée). Réserve = les 30 images PixelSeal du train d'index % 5 == 0, jamais vues ni utilisées pour régler.
« Hors-pli » = chaque image est prédite par un modèle qui ne l'a pas vue (4 modèles, un par cinquième) ; « réglage » = les 140 autres images.

| Méthode | Réglage (140) | Réserve | pleines | partielles |
|---|---|---|---|---|
| Tout à 1 si le décodeur relit > 70 % des bits, sinon tout à 0 | | 0,452 | 0,437 | 0,478 |
| Meilleure carte sans entraînement (occlusion 64 px, exp1) | | 0,51 | | |
| seg v2 (réseau précédent) | | 0,573 | 0,596 | 0,534 |
| Décodeur réentraîné en localisateur, ensemble de 4 plis, seuil 0 | 0,638 | 0,622 | 0,647 | 0,578 |
| + un seuil par population | 0,646 | 0,626 | 0,647 | 0,589 |
| + part minimale de la classe rare | 0,650 | 0,628 | 0,652 | 0,586 |
| + calage des partielles sur les contours SAM | 0,664 | 0,625 | 0,652 | 0,579 |
| + rectangle trouvé par décodage (v3) | 0,680 | 0,641 | 0,652 | 0,622 |
| **idem avec la grille fine de rectangles (livré, v4)** | **0,683** | **0,640** | **0,652** | **0,619** |

Sortie de `python post.py --train pred_train_G.npz --test pred_test_G.npz ...` (h200c puis rejouée à l'identique sur le PC) :
```
0. brut (seuil 0)                               réglage 0.6380 | RÉSERVE 0.6217 pleines 0.6469 partielles 0.5780
1. + un seuil par population                    réglage 0.6458 | RÉSERVE 0.6258 pleines 0.6469 partielles 0.5893
2. + part minimale de la classe rare            réglage 0.6502 | RÉSERVE 0.6278 pleines 0.6522 partielles 0.5858
3. + calage des partielles sur les contours SAM réglage 0.6636 | RÉSERVE 0.6254 pleines 0.6522 partielles 0.5791
4. + rectangle trouvé par décodage              réglage 0.6799 | RÉSERVE 0.6411 pleines 0.6522 partielles 0.6220
```
Le bruit sur 30 images vaut ±0,03 : sur la réserve seule, seul le saut 0,573 → 0,62-0,64 est prouvé. Les étapes du post-traitement
sont prouvées sur les 140 images de réglage (0,638 → 0,680, six nombres réglés) et, pour SAM, en moitiés croisées (voir plus bas).
But fixé (« nettement au-dessus de 0,51 ») : atteint, +0,13. Livrables : `pred_train_pixelseal.npz` (170, hors-pli) et
`pred_test_pixelseal.npz` (340), clés `names`, `logits` (> 0 = filigrané), dans `h200b:~/eac/t1/t1-pixelseal/` et `/tmp/eac-t1-partage/`
(suffixe `_v4`). Sur le test : 161 partielles, dont 37 rendues par un rectangle et 40 par des masques SAM. Poids : Jülich `run_G1..G4/net.pt`. Refaire : `predict.py` → `boxsearch.py` → `sam_masks.py` → `post.py`.

## La méthode livrée
1. **Réseau.** On garde les couches du détecteur PixelSeal (un ConvNeXt, déjà sensible au motif), entrée en 512 px, et on ajoute une
   pyramide (couches qui remontent des cartes grossières vers une carte fine 128x128). Il apprend « filigrané ou propre » case par case,
   sans recevoir le message. Exemples fabriqués à la volée avec le vrai encodeur sur des fonds réels du hackathon (images TrustMark et
   StegaStamp, propres pour PixelSeal), trois recettes vérifiées sur le réel : pleine (tout filigraner, recoller l'original dans un trou),
   rectangle (filigrané dans son propre cadre), objet (filigrane au cadre de l'image, gardé dans la forme d'un objet).
2. **Population.** « Pleine » si le décodeur relit plus de 75 % des bits sur l'image entière (juste à 96,5 %). Un seuil par population,
   et une part minimale de la classe rare (3 % de cases propres dans une pleine, 1 % de cases filigranées dans une partielle).
3. **Objets des partielles : calage sur SAM.** SAM est un modèle de segmentation générique et libre, qui découpe l'image en masques
   d'objets sans rien savoir du filigrane. On ajoute des masques un par un tant que le recouvrement ATTENDU sous la carte du réseau
   monte ; on rend l'union si ce recouvrement attendu dépasse 0,4, sinon on garde la carte.
4. **Rectangles : boîte par décodage**, sans entraînement. On essaie ~3800 boîtes, on rogne chacune, on la remet à 256 px et on demande
   au décodeur d'origine à quel point elle relit le message connu ; on affine les quatre bords sur la marge brute (moyenne des logits
   comptés dans le sens du bit attendu). On rend la boîte si elle relit ≥ 85 % des bits et l'image entière au moins 0,35 de moins.

## Ce que les mesures ont établi
- **bfloat16** (calcul sur 16 bits) n'efface rien : bits justes 0,9835 en fp32, 0,9823 en bfloat16. Hypothèse du tour 2 réfutée.
- **Résidu** (ce que l'encodeur ajoute) : 1,16 niveau sur 255, 39 % des pixels inchangés, identique sur les trois couleurs. Il dépend
  de l'image bien plus que du message (corrélation 0,01 entre deux images, même message). D'où l'échec du filtre adapté (corréler
  l'image avec le motif attendu), même en version différentielle : AUC 0,50 (AUC = probabilité qu'une case filigranée ait un score
  plus haut qu'une case propre ; 0,5 = hasard).
- **Le décodeur d'origine ne localise pas** : avec un trou propre de 128 px, ses votes par case sont justes à 0,788 dans le trou
  contre 0,799 dehors. Un carré filigrané seul est relu à 0,52 (64 px), 0,59 (128 px), 0,80 (256 px).
- **Sur les pleines, le réel (0,65) vaut le fabriqué (0,67)** : pas d'écart de domaine, c'est la finesse de détection qui plafonne.
- **Objets partiels = masques SAM**, pour les trois filigranes. Plafond (meilleure union en connaissant la réponse) : PixelSeal 0,89 /
  0,99 / 0,98 selon la taille (< 3 %, 3-6 %, > 6 % de l'image), TrustMark 0,96 / 0,97 / 0,94, StegaStamp 0,96 / 0,98 / 0,93.
  Les rectangles ne sont jamais des masques SAM (0,51 à 0,68), les trous des pleines non plus (voir limites).

## Par taille de la zone rare (170 images hors-pli, ensemble seul, seuil 0)
Zone rare = partie propre d'une pleine, partie filigranée d'une partielle. « Touchée » = la case la plus sûre tombe dans la vraie zone.
```
part rare            |  n | mIoU  | IoU classe rare | touchée
PLEINES    0.00-0.01 | 12 | 0.491 | 0.002 | 0.08      PARTIELLES 0.00-0.01 | 17 | 0.495 | 0.001 | 0.00
PLEINES    0.01-0.03 | 16 | 0.564 | 0.157 | 0.38      PARTIELLES 0.01-0.03 | 19 | 0.590 | 0.199 | 0.32
PLEINES    0.03-0.06 | 15 | 0.589 | 0.226 | 0.67      PARTIELLES 0.03-0.06 | 10 | 0.610 | 0.263 | 0.60
PLEINES    0.06-0.15 | 23 | 0.734 | 0.525 | 0.91      PARTIELLES 0.06-0.15 | 20 | 0.668 | 0.402 | 0.95
PLEINES    0.15-1.00 | 19 | 0.717 | 0.599 | 1.00      PARTIELLES 0.15-1.00 | 19 | 0.770 | 0.659 | 1.00
```

## Essais, mesures, verdicts
| Essai | Mesure | Verdict |
|---|---|---|
| Prédire les 256 bits par case + accord avec le message | réserve 0,604 avec, 0,598 sans | inutile |
| Aucune vraie image PixelSeal à l'entraînement | 0,604 / pli 0,644 (avec : 0,604 / 0,658) | le réel aide peu |
| Tronc en 512 au lieu de 256 | fabriqué 0,680 contre 0,660 | gardé |
| Branche pleine résolution (type stéganalyse) | pleines fabriquées 0,684 contre 0,683 | aucun gain |
| Modèle spécialisé pleines | réel pleines 0,64, comme le général | inutile |
| Ensemble de 4 modèles | 0,6217 contre 0,6218 pour un seul | aucun gain |
| Marge de toile 8-16 px (piste t1-reel) | marge du décodeur 10,66 / 10,45 / 10,41 à 0 / 8 / 16 px | non concluant, mis en augmentation |
| Seuil adaptatif par image | +0,004 (pleines), 0 (partielles), moitiés croisées | négligeable |
| Aiguillage vers un modèle spécialisé partielles | sans SAM : réserve 0,645 (36 images de réglage) ; avec SAM : 0,6813 / 0,6414 contre 0,6799 / 0,6411 | inutile une fois SAM en place |
| Inférence zoomée (tuiles 256 → 512, maximum) | partielles 0,591 contre 0,639 ; zoom seul 0,485 | écartée |
| Calage SAM des partielles | 59 images : 0,630 → 0,678 ; moitiés croisées +0,041 / +0,049 ; réserve (15) 0,586 → 0,579 | gardé |
| Classifieur de masques SAM (régression logistique) | 0,665 contre 0,678 pour la règle gloutonne | écarté |
| Vérifier un masque par le décodeur (on l'efface, on lit la chute) | AUC 0,66 (objets), 0,80 (trous) ; carte du réseau 0,91 et 0,965 | écarté |
| Grille fine de rectangles (côtés de 48 et 72 px) | 13 boîtes rendues au lieu de 10, 13 vrais rectangles, IoU 0,85 | gardé (v4) |
| Calage SAM sur TrustMark / StegaStamp (cartes t1-reel) | hors réserve +0,010 / +0,008 ; réserve −0,001 / +0,002 | gain faible avec ma règle |

## Limites : pourquoi on ne monte pas plus haut
- **Trou des pleines** (plus gros poste d'erreur). Ce sont des taches libres : union de masques SAM pour 2 images sur 85 (IoU ≥ 0,8),
  même en dilatant ; rectangle pour 2 %. Le réseau classe bien les régions (AUC 0,965) mais dessine flou (16 à 32 px) et rate les
  trous de moins de 3 %. Un seuil parfait par image ne rendrait que +0,03. Planche : `E:\EAC-lab\t1s\t1-pixelseal\planche_pred_D_a.jpg`.
- **Rectangles.** Sur 43, la boîte exacte n'est lisible (≥ 85 % des bits) que pour 13, tous retrouvés. Les 21 dont un côté fait moins
  de 32 px relisent 0,533 : le motif, calculé à 256 px puis réduit, a disparu. Aucun indice visuel : détail fin dedans / anneau = −0,05,
  saut au bord 0,84 (1 = pas de couture).
- **Objets.** Le plafond SAM est à 0,98 et la règle gloutonne à 0,87 sur les zones de plus de 6 % : il reste de la marge dans le choix
  des masques, que ni le classifieur ni le décodeur n'ont su prendre.
- Supposé, non mesuré : qu'un entraînement beaucoup plus long ne déplace pas le plafond des pleines (fabriqué 0,69 → 0,71 de 2500 à
  4500 pas, réel inchangé).

## Critiques indépendantes
Tour 1 (`reports/critique/t1-pixelseal-tour1.md`) : raccourci dans les objets fabriqués sur fonds StegaStamp (corrigé : TrustMark seul),
`predict.py` qui ignorait l'architecture de chaque modèle (corrigé), cible des bits impossible localement (confirmé, retirée).
Tour 2 (`...-tour2.md`) : part minimale en cases éparses (corrigé : tache d'un seul tenant), population vraie utilisée dans la recherche
de rectangle sur le train (corrigé). Non corrigé : seuils réglés sur un modèle seul et appliqués à la moyenne de 4 au test ; la réserve,
prédite elle aussi par la moyenne de 4, mesure bien ce qui est soumis.

## Infrastructure
Connexion Jülich tombée trois fois ; H200 partagées saturées (SAM : 31 à 100 s par image contre 4,3 s sur le PC, RTX 5060) ; depuis
20 h 15 un seul job Jülich pour tout le compte, lancé par l'orchestrateur. Les masques SAM ont donc été calculés sur le PC.
