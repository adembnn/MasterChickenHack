# T1 PixelSeal : critique adverse du localisateur, tour 2
Décision débloquée : où mettre les ~15 h restantes (post-traitement, diagnostic, un ou deux jobs) pour passer 0,636 sur la réserve.
Légende : [MESURÉ] sortie de commande ou planche regardée ; [DÉDUIT du code] lu ligne à ligne ; [À VÉRIFIER] hypothèse + test.
Termes : *classe rare* = la plus petite des deux classes d'une image (trou propre d'une pleine, zone filigranée d'une partielle) ;
*AUC anneau* = probabilité qu'une case de la classe rare ait un score plus « rare » qu'une case de l'anneau de 32 px qui l'entoure
(mesure locale, sans seuil) ; *étalonnage* = correspondance entre la valeur d'un logit et la probabilité réelle.

## 0. Ce que la planche montre et que le tour 1 avait faux [MESURÉ visuellement, planche_pred_D_a/b]
- **Les trous des pleines ne sont pas toujours des taches aléatoires.** Sur 18 pleines, au moins 4 trous suivent le contenu :
  le poulain blanc (trou = l'objet exact), le ciel au-dessus de l'horizon (ski), un rectangle au-dessus des toilettes, une bande
  rectangulaire devant Big Ben. La recette « vrai trou déformé posé sur un autre fond » (`loc.py:111-113`) détruit cet alignement :
  le réseau n'apprend jamais « le trou épouse un objet ». Les pleines ont donc aussi une sous-population objet/rectangle.
- Sur le ski et l'ours, le ciel (amplitude 0,2·JND presque nulle, gris sur la colonne 5) est prédit propre en bloc : le réseau a
  appris « zone plate = propre ». C'est juste quand le trou est le ciel, faux sinon. Dans une zone plate il n'y a pas de signal.
- Partielles : les grandes formes (zèbre, tour, garçon) sont trouvées en tache ronde qui déborde sur le fond ; le rectangle de ciel
  de l'archer est trouvé en tache arrondie (un a priori « rectangle » aurait donné ~0,9) ; les zones à 1-6 % (bus, cerf-volant,
  horloge, fillette, bateau) sont absentes de la probabilité : rien à recoller, il n'y a pas de signal brut à affiner.

## a) Petits trous et petites zones (< 3 %, 39 % des images, IoU de la classe rare 0,02-0,16)
Plafond en jeu : 39 % × (0,5 − 0,05)/2 ≈ +0,09 si on montait l'IoU rare à 0,5. C'est le seul gisement de taille.
Mon avis : **surtout physique, en partie apprentissage**. Arguments [MESURÉ] : 96 px ne donnent que 0,52 bit juste (diag §5) ;
les petits rectangles sont illisibles même avec la boîte exacte (0,55) ; sur fabriqué, où les données sont infinies, petit trou 0,61
contre 0,77 pour les grands, et la branche pleine résolution n'y change rien (0,684 / 0,683). Mais détecter la PRÉSENCE est plus facile
que décoder 256 bits, et deux défauts d'apprentissage poussent vers la classe majoritaire [DÉDUIT du code] :
(1) le contexte global de la pyramide (`locnet.py:23-24`) dit « pleine ⇒ tout filigrané » et écrase le signal local ;
(2) `loc.py:157` : BCE moyenne + Dice, sans Lovász (`--lovasz 0`), donc 30 cases rares sur 16 384 pèsent peu.
**Test de 10 min qui tranche** (inférence seule, modèle D, fabriqué plein) : trous disques de 1, 2, 3, 5, 8, 12 % à position
aléatoire, 64 images par taille ; mesurer (i) l'AUC anneau, (ii) le mIoU avec taille oracle (composante connexe de taille exacte
autour du minimum lissé), (iii) les deux séparés par amplitude moyenne A dans le trou (tiers bas / haut).
- AUC anneau ≥ 0,80 à 2 % ⇒ l'information est là, c'est le seuil et la forme : post-traitement par composante connexe (§d3) et Lovász.
- AUC anneau ≤ 0,65 à 2 % et qui monte avec A ⇒ limite physique : arrêter d'investir, ne garder que le billet de loterie.
À changer si c'est l'apprentissage : Lovász 0,5 ; tirage des trous biaisé vers 0,5-4 % ; trous alignés sur des objets (masques SAM ou
COCO du fond lui-même) ; masquer la perte où A < 0,5 niveau (étiquette indécidable, `loc.py:154`).

**Inférence zoomée du vérificateur (tuiles 256→512 et 128→512, fusion par maximum)** : physiquement peu plausible. Agrandir
n'ajoute aucun pixel d'information ; ça ne change que l'échelle vue par le réseau.
- Objets : le motif est au cadre de l'image (u à 256, agrandi ×2 à 512). Le réseau à 512 le voit déjà à son échelle d'entraînement ;
  zoomer ×2 ou ×4 lui montre un motif 2 à 4 fois plus gros qu'il n'a jamais vu ⇒ plutôt une perte [DÉDUIT].
- Rectangles : le motif est calculé à 256 puis RÉDUIT à la taille du rectangle. Un rectangle de 90 px porte un motif comprimé ×2,8,
  dont les hautes fréquences sont perdues à la réduction puis à l'arrondi 8 bits. Zoomer le remet à l'échelle, mais `boxsearch.py`
  fait déjà exactement cela (boîte exacte remise à 256) et lit 0,55 : l'information n'y est plus [MESURÉ]. Gain possible seulement
  entre ~150 et 250 px, déjà couverts par la recherche de boîte. La fusion par maximum ajoute en plus des faux positifs à chaque tuile.
Verdict : coût 20 min d'inférence, je le classe dernier ; s'il est lancé, juger sur l'IoU rare des < 3 % hors-pli, pas sur le mIoU global.

## b) Objets : coller au contour avec l'image
- **Sélection de masques SAM par mIoU espéré, sans réglage** : générer les masques automatiques SAM (plusieurs granularités) ; pour chaque
  candidat M, calculer le mIoU espéré sous la carte de probabilité p (IoU « souple » des deux classes) ; garder le meilleur candidat
  SEULEMENT s'il bat le mIoU espéré de la prédiction brute seuillée. Un seul critère, pas de seuil appris. Sert aussi aux pleines
  dont le trou est un objet (poulain, ciel).
- **Étalonnage d'abord** : p est sous-confiante (planche) ; le mIoU espéré n'a de sens que si p est étalonnée. Appliquer une mise
  à l'échelle de température par population, apprise sur les logits hors-pli (1 paramètre, robuste).
- **Plafond oracle SAM 0,71** : incohérent avec la planche pour les grands objets (zèbre, tour, poulain : un masque SAM devrait dépasser
  0,9). Cohérent seulement si la moyenne est tirée vers le bas par les objets < 3 % (bus, horloge) que SAM ne propose pas à la bonne
  granularité, ou par des masques de vérité = union de plusieurs objets (les deux chiens). [À VÉRIFIER] : oracle par taille d'objet
  (< 3 %, 3-15 %, > 15 %) ; si > 15 % donne ≥ 0,85, SAM vaut la peine sur ces images-là.
- CRF dense ou filtre guidé : effet de bord de 8-16 px, gain faible (contours flous à 16-32 px), plus fragile que la sélection SAM.

## c) Mieux utiliser le message connu et l'embedder
- **Piste hors des réponses évidentes : décodage masqué des candidats SAM (« boxsearch pour objets »).** L'objet porte le filigrane au
  cadre image : remplacer l'extérieur de M par une version floutée de l'image (σ ≈ 1,5 px, détruit le motif sans casser la géométrie ni
  créer de bord net), puis lire les bits avec le décodeur d'origine à 512. Si M contient l'objet filigrané, les bits montent (vraie forme
  sur le réel : 0,65 sur les grands objets [MESURÉ diag §6]) ; un masque faux reste vers 0,5. Test 10 min sur ~40 objets hors réserve :
  bits(vrai masque) contre bits(meilleur faux candidat SAM de même aire) ; si le vrai masque gagne ≥ 70 % des duels, en faire le
  critère de choix des masques SAM (indépendant du réseau, donc complémentaire). Inutile sous ~8 % d'aire (§a).
- Même idée, en négatif, pour les trous-objets des pleines : flouter l'intérieur d'un candidat ; un trou vrai ne fait presque pas
  baisser la marge globale. Signal faible (3-8 % de la marge) : à mesurer dans le même script, sans y croire d'avance.
- Gradient de la marge, filtre adapté, double filigrane : même biais de texture que le filtre adapté enterré (AUC 0,50) ; ne pas relancer.

## d) Bugs et fuites
1. **Étalonnage croisé dans le post-traitement** [DÉDUIT du code] : `predict.py:36-39` donne aux images hors-pli les logits d'UN modèle,
   mais au test (et à la réserve) la MOYENNE de 4 modèles. `post.py` règle `t_full`, `t_part`, `q_*` sur des logits mono-modèle et les
   applique à des logits moyennés, plus petits et autrement répartis. Les seuils transférés sont faux d'une quantité inconnue.
   Correctif : prédire le test avec le seul modèle qui a le même entraînement que le réglage, ou régler sur des logits moyennés
   « hors-pli » (impossible ici) ; le plus simple : seuils en quantiles ou température par population, pas en valeur absolue.
2. **Part minimale de la classe rare** (`post.py:48-50`) : arithmétique juste (décalage du logit au quantile ; la case au quantile
   exact tombe du côté propre à cause du `> 0` strict, négligeable). Mais ce sont des cases ÉPARSES (les q % de plus bas logits
   n'importe où), pas une composante connexe : avec `q_full` jusqu'à 0,08, on sème des cases propres sur les zones plates.
   Le billet de loterie du tour 1 était « une tache connexe de taille médiane autour du minimum lissé ». À réécrire ainsi.
3. **Sur-ajustement des 5 paramètres** : 4 réglés (2 seuils sur 25 valeurs, 2 parts sur 6-7) sur ~70 images par population ; gains
   +0,008/+0,004/+0,013 sur le réglage, tous sous le bruit (±0,02 à n=140) ; réserve +0,014 contre +0,025 au réglage. Optimisme
   modéré, pas de catastrophe. `tau` = 0,85 et l'écart 0,35 ont été choisis en lisant `boxsearch.py:78-84`, dont les tranches mêlent
   la réserve (ligne 81 sans filtre `res`) : légère fuite de choix.
4. `boxsearch.py:62-63` : sur le train, la recherche ne se fait que sur les VRAIES partielles (`full_watermark`), au test sur les
   partielles devinées (`ga ≤ 0,75`). Le 0,914 ignore donc les 2 % d'erreurs de population. Mineur.
5. `post.py:59` : sans tableau `role`, `ok` vaut vrai partout ; si `--train` reçoit un fichier prédit sans `--folds`, le réglage
   se fait en silence sur des logits appris. Ajouter un `assert` sur la provenance.
6. Toujours présent depuis le tour 1 : `loc.py:131-132` décalage en bords répétés (bandes étiquetées sans motif).
7. `loc.py:113` : l'intersection de deux trous peut être vide ⇒ exemple « plein » sans trou ; sans gravité, mais il gonfle la classe
   « tout filigrané » au moment où le problème est justement de trouver les petits trous.

## e) Plan pour ~15 h
- Aiguillage par population (général pour les pleines, spécialisé pour les partielles) : +0,018 sur 66 images non apprises, même
  sens que la réserve (0,598 contre 0,565) et le pli (0,674 contre 0,63). Deux mesures indépendantes concordent : l'adopter tout de
  suite (`post.py --train_part/--test_part` existe), puis RE-régler le post-traitement sur ces logits mélangés.
- Pas d'entraînement plus long ni de tronc plus gros : le spécialiste à 4500 pas gagne sur fabriqué (0,71) et rien sur le réel (0,64),
  l'ensemble de 4 plis n'apporte rien. Le goulot est l'écart fabriqué/réel et la classe rare, pas la capacité. Pas de distillation.
- Un seul job d'entraînement, après le test du §a : spécialiste pleines avec trous alignés sur des objets/rectangles du fond, Lovász 0,5,
  tirage des trous vers 0,5-4 %, perte masquée où A < 0,5 niveau. Décidé sur le pli, réserve lue une fois.

| # | Expérience | Coût | Chiffre qui décide |
|---|---|---|---|
| 1 | Aiguillage par population + post-traitement réécrit : température/seuils par quantile réglés sur les logits du même type que le test (bug d1), billet de loterie en composante connexe (d2) | 30 min CPU, aucun entraînement | ≥ +0,02 hors-pli sur le brut aiguillé ; réserve lue une fois |
| 2 | Test physique des petites zones : AUC anneau et mIoU à taille oracle par taille de trou (1-12 %) et par amplitude A, modèle D, fabriqué | 10 min GPU, inférence | AUC anneau à 2 % ≥ 0,80 ⇒ lancer #4 ; ≤ 0,65 ⇒ abandonner les < 3 % |
| 3 | SAM : masques candidats choisis par mIoU espéré (p étalonnée), départagés par décodage masqué (extérieur flouté) ; partielles objets et trous-objets des pleines | 1 h (SAM déjà installé) | décodage : vrai masque gagne ≥ 70 % des duels ; sélection : ≥ +0,03 sur les objets hors-pli > 3 % |
| 4 | Job unique : spécialiste pleines avec trous alignés sur le contenu, Lovász 0,5, trous 0,5-4 %, perte masquée si A < 0,5 | 1 job (4 GPU, 4 graines ou variantes) | pleines du pli ≥ +0,03 sur le général ; IoU rare des trous < 3 % ≥ 0,25 |
| 5 | Inférence zoomée du vérificateur, restreinte aux rectangles devinés de 150-250 px | 20 min inférence | IoU rare des partielles < 3 % hors-pli ≥ +0,05 ; sinon enterrée |
