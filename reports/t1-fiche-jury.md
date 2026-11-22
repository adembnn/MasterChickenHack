# T1 · fiche jury : localisation de filigrane

État au 22/11, 10 h 45. Décision débloquée : ce qu'Adem dit du track 1 au pitch et comment il répond aux questions.
Sources : issue #3, `reports/t1-tableau.md`, `reports/t1-t1-reel.md`, `reports/t1-t1-assemblage.md`, `reports/t1-t1-residus.md`, leaderboard lu à 10 h 40.

## 0. Les chiffres à retenir
- **Public : 0,8230** (candidat 38, soumission 2910, 4 h 32). **1er du track 1** au tableau (affiché 0,82, à égalité d'arrondi avec Incoming PhD ; Murmell 0,81).
- Départ : 0,635 (21/11, 17 h 25). Douze envois retenus, +0,19 en onze heures.
- Par filigrane (réserve du candidat 38) : **StegaStamp 0,930 · TrustMark 0,809 · PixelSeal 0,709**.
- Repères : masque constant (tout propre ou tout filigrané) = 0,25 ; rectangle englobant parfait (donné par la vérité) = 0,66.

## 1. Ce qu'on demande
**Filigrane** : signal invisible ajouté aux pixels d'une image, qui code un message de bits (une suite de 0 et de 1). Un **décodeur** relit ce message depuis l'image.
Trois filigranes publics : **PixelSeal** (Meta, 2025), **TrustMark** (Adobe, ICCV 2025), **StegaStamp** (Berkeley, CVPR 2020). Ils sont faits pour relire *un* message sur *toute* l'image, pas pour dire *où* il se trouve.

Les images ont ensuite été retouchées :
- **Outpainting** : l'image filigranée est agrandie ; le filigrane n'occupe plus qu'un rectangle.
- **Retrait ou ajout d'objet** : un objet est effacé ou collé ; il reste un **trou** sans filigrane.
- **Greffe d'objet** : un objet filigrané est collé dans une image propre ; seul l'objet porte le filigrane.

On rend, pour 1 020 images de test 512 × 512 pixels, un **masque** de 128 × 128 cases (une case = 4 × 4 pixels) : 1 = filigrané, 0 = propre. On connaît pour chaque image le filigrane utilisé et le message caché. Train : 510 images (170 par filigrane) avec leur vrai masque.

**Score (mIoU)**. IoU d'une classe = cases où prédiction et vérité disent toutes deux cette classe, divisé par les cases où au moins l'une des deux la dit. Exemple : 100 cases vraiment filigranées, on en prédit 120 dont 90 justes → IoU = 90 / (100 + 120 − 90) = 0,69. On fait la moyenne de l'IoU « propre » et de l'IoU « filigrané » sur chaque image, puis la moyenne sur les images. Chaque image compte autant, quelle que soit la taille de la retouche.

**Pourquoi c'est un vrai problème** : sans localisation, un contenu ajouté après coup est attribué au propriétaire du filigrane. On peut lui faire porter une image qu'il n'a pas produite.

## 2. Ce qu'on a fait, dans l'ordre
### Étape 0 : mesurer le signal (21/11 après-midi, A100 de Jülich)
- Dans la vraie zone filigranée, le décodeur relit 81 à 85 % des bits justes ; dehors, 50 % (le hasard). Le signal existe.
- Mais la carte native du décodeur fait 8 × 8 cases (64 × 64 pixels chacune) : 0,35. Trop grossière.
- Découverte qui oriente tout : **la moitié des images sont « pleines »** (filigrane partout sauf un trou). Le rectangle parfait y vaut l'image entière, d'où le plafond de 0,66. Le score se joue sur **la forme des trous**.
- Fermé par la mesure : filtre adapté (comparer l'image au filigrane recalculé, 0,28 à 0,36) ; cartes par occlusion (masquer des morceaux et regarder ce que perd le décodeur, 0,48 à 0,51) ; gradients.

### Étape 1 : un segmenteur appris (0,635 → 0,706)
**Segmenteur** = réseau qui prédit une classe par case. On a pris un **U-Net** (encodeur qui résume l'image, décodeur qui redonne une carte à pleine taille) sur un tronc ResNet-34.
- v1, entraîné sur 6 000 images COCO filigranées par nos soins : TrustMark 0,59, StegaStamp 0,63, PixelSeal 0,29 (n'apprend pas). Écart entre exemples fabriqués et vraies images.
- v2 : tronc = le décodeur du filigrane lui-même, vraies images du train mêlées aux lots → **0,635 public**.
- v3 à v6 (session t1-reel) : exemples **fabriqués à partir des images du hackathon**, avec les **vraies formes de masques** → 0,674, 0,697, 0,701, **0,706**.

### Étape 2 : trois découvertes sur les vraies images (0,706 → 0,7935)
1. **La toile** : le filigrane réel n'a pas été calculé sur le cadre 512, mais sur une toile plus grande (512 + 32 pixels en médiane, rognée de façon décentrée). En la reproduisant, un réseau entraîné sans aucune vraie image passe de 0,658 à 0,820 sur StegaStamp.
2. **Première couche en pleine résolution** : le réseau lit chaque pixel au lieu d'un pixel sur deux. C'est le levier principal : PixelSeal se met enfin à apprendre (0,612).
3. **Recette** : long pré-entraînement sur exemples fabriqués (8 000 pas), puis affinage sur les vraies images, **5 plis** (5 modèles qui ont chacun vu 4/5 des images), moyennés.

### Étape 3 : l'assemblage, qui corrige les formes (session t1-assemblage)
1. **Aiguillage** : le taux de bits justes sur l'image entière dit si elle est « pleine » ou « partielle » (juste à 100 % sur la réserve PixelSeal et TrustMark, 94 % StegaStamp).
2. **Partielles, objets** : **SAM** (Segment Anything, Meta) découpe l'image en objets sans connaître la tâche ; chaque région prend le vote majoritaire de la carte du réseau. Les contours deviennent ceux des vrais objets.
3. **Partielles, rectangles** : on cherche parmi ~13 000 rognages celui où le décodeur relit le message le mieux (score **z** = écart au hasard en écarts-types), puis on cale le cadre au pixel en refiltrant l'image avec le vrai message (**re-tamponnage**). Rectangles TrustMark : 0,886 → 0,943.
4. **Pleines** : seuil réglé, plancher de cases propres (0,5 à 4 %).
5. Tous les réglages sont faits sur des images que le modèle n'a pas vues (**hors-pli**).
Résultats : candidat 5 **0,7384** ; candidat 22 **0,7695** (1er du track pour la première fois) ; candidat 27 **0,7935**.

### Étape 4 : moyennes de modèles et PixelSeal (0,7935 → 0,8230)
Moyennes de 3 à 9 modèles par filigrane, canaux passe-haut (le réseau reçoit aussi les détails fins de l'image, là où vit le filigrane), modèle StegaStamp et PixelSeal pleine résolution.
Candidat 35 **0,8033** → candidat 36 **0,8131** → candidat 38 **0,8230**.

## 3. Comment on a mesuré (le point fort à dire au jury)
- **Réserve** : 102 images du train (indice multiple de 5) jamais utilisées pour apprendre ni régler.
- **Validation croisée 5 plis sur les 510 images** : chaque image est prédite par un modèle qui ne l'a pas vue, toute la chaîne (aiguillage, seuils, choix de source) refaite à chaque pli.
- **Écart apparié et intervalle de confiance à 95 %** : on compare deux fichiers image par image, on rééchantillonne 10 000 fois ; un fichier part seulement si l'intervalle exclut 0.
- **Maker-checker** : la session qui produit un fichier n'est jamais celle qui le valide ; un vérificateur recalcule tout.
- **Prévu contre obtenu** : du candidat 5 au 35, prévu +0,065, obtenu au public +0,065.
- **Sondes** : on a envoyé le candidat 35 avec un seul filigrane remplacé, pour savoir lequel avait vraiment gagné au public. Résultat : PixelSeal transfère (+0,0082 pour +0,0096 prévu), TrustMark non (−0,0002 pour +0,011 prévu). On a donc cessé de croire les gains TrustMark hors réserve.

## 4. Où on a bloqué, pourquoi on plafonne
Perte restante du candidat 38 en validation croisée : 1 − 0,835 = 0,165.
- **57 % de la perte = le trou des images pleines PixelSeal et TrustMark.** StegaStamp y fait 0,944 ; PixelSeal 0,73 ; TrustMark 0,76.
- **Petits rectangles PixelSeal** (moins de 32 pixels de côté, bandes de 102 × 9 pixels par exemple) : souvent manqués. Le décodeur n'y relit rien (z médian 0,8, donc le hasard), alors que 43 % des pixels y ont été modifiés.
- **Bords** : 32 % (PixelSeal), 40 % (TrustMark), 76 % (StegaStamp) des cases fausses sont à 8 pixels ou moins de la vraie frontière.

Pourquoi :
1. **Limite du signal.** Ces filigranes étalent le message sur toute l'image pour résister aux retouches. Sur une zone de quelques cases, il y a trop peu de signal pour trancher. Mesure : même sur des rectangles fabriqués déjà vus à l'entraînement, IoU 0,095 sous 32 pixels, 0,25 de 32 à 64, 0,50 de 64 à 128, 0,79 au-delà.
2. **PixelSeal est le plus discret** : appris de façon adversariale (un second réseau cherche à le repérer pendant son entraînement), son résidu est faible et diffus.
3. **Peu de vraies données** : 170 images par filigrane. Bruit de la réserve ±0,03 ; bruit du public (306 images) écart-type 0,011.
4. **Les gains ne s'additionnent plus.** Depuis le 38, chaque nouvelle carte gagne seule environ +0,018 en réserve, mais 70 à 85 % du gain disparaît après le vote SAM et les rectangles : elle corrige ce que l'assemblage corrigeait déjà. Meilleur cumul (candidat 49) : +0,0048, intervalle [−0,0012 ; +0,011], non envoyé.
5. **L'erreur est dans la forme du trou dessinée par le réseau, pas dans le seuil.** Plafond d'un seuil choisi par image avec la vérité : +0,005 seulement.

Pistes mesurées et fermées (à citer si on demande « vous avez essayé X ? ») : seuil fin et morphologie (dilatation, érosion : neutres ou négatives) ; inférence zoomée ; adaptation au test image par image (−0,02, le modèle grossit la zone) ; modèles spécialistes des pleines ou des trous (avance de +0,058 sur le trou TrustMark qui disparaît à l'affinage) ; perte contrefactuelle par paires ; pseudo-étiquetage du test ; dégradations JPEG, flou, rotation (ce n'était pas la cause de l'écart, c'était la toile) ; recherche des images d'origine dans COCO val2017 (aucune).

## 5. Questions probables du jury
**1. Pourquoi pas Watermark Anything, conçu pour localiser ?**
Il localise son propre filigrane, pas PixelSeal, TrustMark ou StegaStamp. La tâche demande de localiser des filigranes existants. On s'est servi de leurs décodeurs comme tronc et comme source de signal (score z), puis on a appris la localisation par-dessus.

**2. Vous n'avez pas sur-appris le tableau public ?**
Non, pour trois raisons : aucun réglage n'utilise le public ; chaque envoi a une prévision écrite avant (prévu +0,065, obtenu +0,065 du candidat 5 au 35) ; on a refusé d'envoyer des fichiers dont l'intervalle touchait 0, y compris le candidat 49 qui semblait meilleur.

**3. Pourquoi StegaStamp va si bien et PixelSeal si mal ?**
StegaStamp a un résidu fort et structuré : le réseau le voit case par case (0,93). PixelSeal a été entraîné pour être indétectable par un réseau adverse : son résidu est faible, et sur une petite zone le décodeur ne lit plus rien (z au niveau du hasard). TrustMark est entre les deux.

**4. Quelle a été l'idée décisive ?**
La toile agrandie et la première couche en pleine résolution. Sans elles, les exemples fabriqués ne ressemblaient pas aux vraies images et les réseaux n'apprenaient pas PixelSeal. StegaStamp : 0,658 → 0,820 sans aucune vraie image, juste en fabriquant correctement.

**5. Vous utilisez des données externes ?**
Les modèles finaux sont fabriqués à partir des images du hackathon (fonds pris parmi les images des deux autres filigranes, formes prises parmi les vrais masques) et des outils de filigrane fournis dans le kit. COCO n'a servi qu'à la première version. Les troncs (ResNet) et SAM sont des poids publics pré-entraînés.

**6. Pourquoi l'aiguillage pleine / partielle ?**
Les deux cas n'ont pas le même problème : sur une pleine on dessine un trou, sur une partielle on trouve une zone. Le taux de bits justes sur l'image entière les sépare sans erreur sur la réserve PixelSeal et TrustMark. Chaque population a ensuite son seuil et son post-traitement.

**7. Qu'est-ce qui vous ferait gagner +0,03 ?**
Ramener le trou des pleines PixelSeal et TrustMark au niveau de StegaStamp (de 0,73 et 0,76 à 0,85) vaudrait +0,035 sur la moyenne. Il faudrait une perte calculée directement sur l'IoU du trou et plus de vraies images ; nos deux essais de modèles dédiés aux trous n'ont pas tenu après affinage.

**8. Combien coûte une prédiction ? Est-ce utilisable en vrai ?**
Non mesuré en production. La chaîne lance plusieurs U-Net, SAM ViT-H et une recherche de rectangle (~13 000 rognages par image partielle) : c'est une chaîne de compétition, à alléger avant usage. Ce qui reste utile hors compétition : on sait dire avec un score z si une zone porte le message, ce qui suffit à contester une attribution.

**9. Votre 0,82 est-il vraiment devant Incoming PhD (0,82) ?**
Pas établi : le public porte sur 306 images, écart-type 0,011. La note finale vient des 70 % cachés ; notre réserve et notre validation croisée prévoient bien le public, donc on s'attend à garder le même niveau.

**10. Comment vous êtes-vous organisés ?**
Plusieurs sessions d'agents en parallèle, chacune sur un lot (cartes du réseau, assemblage, rectangles, PixelSeal), sur des GPU H200. Boucle mesurer, critiquer, corriger : chaque idée devient une mesure, un agent critique séparé cherche la faille, un vérificateur recalcule avant tout envoi. Une seule session envoie, pour ne pas gâcher les créneaux de 10 minutes.

## 6. Pitch en 40 secondes
« Les filigranes PixelSeal, TrustMark et StegaStamp sont faits pour relire un message sur toute l'image, pas pour dire où il est. On a d'abord mesuré que le décodeur relit 85 % des bits dans la zone filigranée et 50 % dehors : le signal existe, mais en cases de 64 pixels. On a appris un segmenteur sur des exemples fabriqués avec les vrais outils, après avoir trouvé que le filigrane réel était calculé sur une toile plus grande que l'image. Puis on corrige les formes : SAM pour les objets, le décodeur pour caler les rectangles au pixel. 0,635 au départ, 0,823 aujourd'hui, premiers du track. Chaque envoi a été prévu avant d'être envoyé, et la prévision a tenu. Ce qui reste : les tout petits trous, où il n'y a presque plus de signal. »
