# T3 unlearning : ce que la communauté a trouvé, ce qu'on n'a pas essayé

Veille du 22 nov. 2026, 04h30. Sources lues : les writeups Kaggle du NeurIPS 2023 Machine Unlearning Challenge
(3e, 5e, 7e, 8e et 10e places), le billet de Ken Ziyu Liu (Stanford, « Machine Unlearning in 2024 »),
et deux fils r/MachineLearning sur SSD (2308.07707) et JiT (2402.01401).

## Le point qui change tout

La métrique du NeurIPS 2023 est de la même famille que la nôtre : on est noté sur la ressemblance avec des
modèles **réentraînés**, pas sur la simple chute de précision. Enseignements chiffrés de cette compétition
(gold = 0,19, vainqueur = 0,12, baseline gradient ascent = 0,06) :

- les ingrédients présents chez tous les gagnants sont : **bruit sur les poids**, **ré-initialisation de
  paramètres**, **régularisation KL**, et **réentraînement sur le jeu de rétention** ;
- le 5e place écrit noir sur blanc que **softmax labels / KL loss, SCRUB, Relax Loss, class weights,
  label smoothing** ont été essayés et **n'ont pas marché** sur cette métrique.

Notre S0 et notre S1 sont exactement dans la famille « distillation douce depuis orig ». C'est la famille qui
ne marche pas. Ça explique nos scores de 0,00 à 0,03 sans qu'il y ait de bug : le modèle reste trop proche de
orig, donc KL(orig‖sub) est faible et la prédiction reste celle de orig, or le gold, lui, se trompe.

## Les trois recettes qui ont réellement gagné

### 1. Ré-initialisation stochastique + cycles oubli/mémorisation (8e, score 0,102)
Tiré du writeup de l'équipe de Kyungpook National University (junhyunb.github.io).
- À chaque cycle, choisir **6 couches au hasard** dans un vivier qui **exclut la couche FC et les projections
  de raccourci** (elles portent l'information de classe et de résolution), puis les ré-initialiser.
- Réentraîner 1 à 2 époques avec une perte qui préserve la connaissance : **MSE sur les sorties de orig**.
  Le MSE bat la cross-entropy (0,068 contre 0,065) et écrase L1 (0,033).
- Faire **3 à 4 cycles** en augmentant progressivement la part ré-initialisée. C'est le facteur qui apporte
  le plus : 1 cycle 0,068, 2 cycles 0,084, 3 cycles 0,086.
- Bruit gaussien σ = 0,01 sur les images : petit gain. Flip vertical, recadrage, cutout : **catastrophe**
  (0,025 et 0,00001).
- Le choix **aléatoire** des couches bat le choix par information de Fisher (0,062 contre 0,049), parce que
  le tirage aléatoire reproduit la diversité des modèles réentraînés.

### 2. Poids transposés avant réentraînement (5e place)
Avant de réentraîner, on échange les deux dernières dimensions de chaque convolution :
`module.weight.swapaxes(2, 3)`. Cela détruit la mémoire des images vues tout en gardant les filtres utiles,
et converge beaucoup mieux qu'un départ aléatoire. Applicable chez nous en permutant les matrices Q/K/V et
les projections des blocs ViT, ou plus simplement en permutant les lignes de la tête et les colonnes d'entrée.

### 3. Étiquettes aléatoires + masque de saillance (10e, SalUn)
Masque de saillance calculé par gradient ascendant sur l'oubli, puis descente de gradient avec étiquettes
aléatoires en ne modifiant que les poids du masque. C'est l'approche qui a le mieux survécu aux ablations
dans cette compétition. Nous n'avons jamais essayé le masque : notre version applique le changement à tout
le réseau, ce qui détruit la rétention.

Complément utile, la recette KDE citée dans le benchmark Deep Unlearn : ré-initialiser **première et dernière
couches** d'un élève, puis le distiller vers le maître sur des vues et sur le jeu de rétention.

## Ce que ça donne chez nous, où il n'y a AUCUN jeu de rétention

Contrainte : on n'a que les 3500 images à oublier, sans étiquettes, et l'énoncé interdit les images externes
pour entraîner. On remplace donc le « réentraînement sur retain » par deux substituts :

- les mêmes images à oublier, vues autrement, servent de jeu de travail ;
- l'ancrage KL vers orig se fait sur des **mélanges** (mixup, cutmix) des images à oublier, ce qui est
  licite puisque aucune image nouvelle n'entre dans l'entraînement.

Les trois variantes à lancer, chacune en trois graines, puis une soumission par variante :

- **V1 ré-initialisation** : ré-initialiser 6 blocs tirés au hasard (jamais la tête), 3 cycles, 1 à 2 époques
  par cycle, cible de perte cale sur les moments supposés du gold (0,55 et 1,17 pour une précision gold
  de 0,85), ancre KL sur les mélanges.
- **V2 transposition** : permuter les matrices des blocs (Q/K/V et projections) et la tête, puis fine-tuner
  4 époques sur les mêmes cibles.
- **V3 masque** : gradient ascendant sur l'oubli pour obtenir le masque de saillance, puis étiquettes
  aléatoires voisines (classes proches par cosinus) sur les seuls poids du masque.

Diagnostic à mesurer avant de soumettre, pour chaque candidat : moyenne et écart-type de la perte sur les
images exactes (à comparer à 0,55 et 1,17), part d'argmax différents, et KL vers orig sur des copies
retouchées. La porte de 70 % de précision n'est pas mesurable localement, donc on garde la variante la plus
conservatrice comme filet.

## Ce qu'il ne faut plus essayer

SCRUB, Relax Loss, pondération de classes, label smoothing, distillation KL seule, augmentation forte
(flip, recadrage, cutout) : tous documentés comme perdants sur cette famille de métriques, et notre propre
journal le confirme pour les quatre premiers.

## Matériel

Les H200 libres sont `eac-h200-9` (10.96.0.91) et `eac-h200-10` (10.96.0.80), accessibles avec
`/clients/ami/.ssh/eac_h200_all_config` via `ProxyJump eac-h200-6`. Un entraînement de 1500 pas sur
ViT-B/16 tient en moins de six minutes sur une carte.
