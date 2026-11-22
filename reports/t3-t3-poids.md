# T3 · session t3-poids : méthodes dans l'espace des poids (21/11, état final de la série de mesures)

Décision débloquée : la famille « agir sur les poids » vaut-elle mieux que la distillation vers des cibles fabriquées, et quel fichier envoyer.
Tout est mesuré sur la H200, en mode évaluation et en fp16 comme le serveur, avec le tableau de bord `dash.py` de t3-methodes. Aucune image externe n'entraîne un candidat.

## Termes
- **Vecteur de tâche** : différence de poids `plus − orig`, où `plus` est `orig` sur-entraîné sur le forget set. **Négation** : `orig − α·(plus − orig)`.
- **Pseudo-étiquette** : classe prédite par `orig` (les vraies étiquettes ne sont pas fournies).
- **m, s** : moyenne et écart-type de l'entropie croisée (loss) avec la pseudo-étiquette, sur les images exactes.
- **Réserve c** : 698 des 3 500 images (20 %, découpage 40/40/20 de `split.pt`), non utilisées pour apprendre le vecteur (appris sur les 2 802 autres, a+b). Ce sont des images vues par `orig`, des mêmes classes. Elle sert à mesurer la dérive.
- **KL** : divergence de Kullback-Leibler entre la sortie d'`orig` et celle du candidat, softmax à température 2 (celle du score de rétention).
- **Sélectif** : l'oubli touche les images traitées (a+b) et pas la réserve c.

## 1. Résultat principal
La négation du vecteur de tâche appris par entropie croisée + AdamW sur les images exactes est sélective, et elle l'est beaucoup plus quand on ne l'applique qu'aux blocs hauts du transformeur.
Score réel du premier fichier, `NEGce_all_a2` (modèle entier, α=2) : **0,0219** (soumission 1330). La distillation à oubli comparable (`S1_A85_l30`) : 0,0114.

Frontière mesurée (vecteur appris sur a+b, 3 époques, AdamW 2e-5 ; sorties `logs/run1.log`, `run2.log`, `run6.log`) :

| Blocs touchés | α | Exactes a+b : m / s / classes changées | KL réserve c | Réserve c sous flou fort : précision (orig 0,867) | Réserve c sous couleur forte : KL |
|---|---|---|---|---|---|
| tout le modèle | 2 | 0,211 / 1,465 / 3,2 % | 0,031 | 0,847 | 0,117 |
| tout le modèle | 3 | 0,413 / 2,065 / 7,0 % | 0,121 | 0,808 | 0,310 |
| tout le modèle | 4 | 0,823 / 2,597 / 17,1 % | 0,373 | 0,742 | 0,630 |
| 0 à 5 + embedding | 4 | 0,183 / 0,983 / 4,8 % | 0,128 | 0,797 | 0,405 |
| 6 à 11 + tête | 4 | 0,356 / 2,110 / 4,6 % | 0,020 | 0,842 | 0,067 |
| 6 à 11 + tête | 6 | 0,620 / 2,909 / 8,0 % | 0,066 | 0,824 | 0,155 |
| 6 à 11 + tête | 8 | 0,925 / 3,528 / 12,1 % | 0,163 | 0,797 | 0,278 |
| 8 à 11 + tête | 6 | 0,461 / 2,494 / 6,0 % | 0,025 | 0,835 | 0,083 |
| 9 à 11 + tête | 8 | 0,493 / 2,591 / 6,5 % | 0,033 | 0,818 | 0,105 |
| 9 à 11 + tête | 12 | 0,861 / 3,531 / 10,6 % | 0,109 | 0,781 | 0,235 |
| 10 à 11 + tête | 8 | 0,296 / 1,869 / 4,0 % | 0,018 | 0,824 | 0,073 |
| 11 + tête | 16 | 0,360 / 1,956 / 6,2 % | 0,062 | 0,787 | 0,175 |
| tête seule | 16 | 0,007 / 0,044 / 0 % | 0,013 | 0,824 | 0,055 |

Lecture : à oubli égal (m ≈ 0,4), limiter la négation aux blocs 6 à 11 divise la dérive par 6 (KL 0,020 contre 0,121). Les blocs 0 à 5 et la tête n'apportent que de la dérive.
Au-delà de m ≈ 0,6 la dérive croît plus vite que l'oubli : la négation pure n'atteint pas m = 1,5 avec une dérive contenue.
Forme de l'oubli : queue lourde (s vaut 4 à 6 fois m). Peu d'images deviennent très fausses, la plupart gardent la sortie d'`orig` (probabilité max à T=2 : 0,92 à 0,94 contre 0,985).

## 2. Ce qui a été écarté, avec la mesure
| Idée | Mesure | Verdict |
|---|---|---|
| Vecteur appris avec une perte de marge (poids égal par image, SGD) | cosinus 0,72 avec le vecteur appris sur des vues fortement augmentées (0,96 dans la tête) ; négation α=1 : KL(a+b) 0,52, KL(c) 0,40 | non sélectif : c'est une direction de classe, pas d'image |
| Même vecteur, orthogonalisé contre celui des vues augmentées | α=1 : KL(a+b) 0,093, KL(c) 0,056 | non sélectif |
| « exact − augmenté » (monter sur les exactes, descendre sur les vues) | α=1 : KL(a+b) 0,109, KL(c) 0,079 | non sélectif |
| Ablation d'un bloc ou d'une branche (gain LayerScale × 0 ou × 0,5) | la marge des copies légères chute autant que celle des exactes pour les 24 branches | aucune branche propre aux images exactes |
| Importance de Fisher (gradient de la marge au carré), exactes contre vues augmentées | rapport 0,8 à 1,0 dans tous les groupes, corrélation des logarithmes 0,996, recouvrement du top 1 % : 0,89 | la mémorisation n'est pas localisée dans des poids repérables ainsi |
| NPO ou montée de gradient avec ancre, un seul optimiseur | loss d'oubli inchangée après 200 pas (1,386) | le gradient d'oubli est saturé (probabilité ≈ 1) et écrasé par l'ancre ; corrigé par deux optimiseurs séparés (§4) |
| Effacer puis réparer : bruit gaussien relatif (0,3 / 0,6 / 1,0 × l'écart quadratique du tenseur) sur les blocs 8-11, puis 500 pas de distillation d'`orig` sur mélanges et vues fortes | loss des images visées 0,000 / 0,001 / 0,003 (aucun oubli) ; KL(c) 0,0014 / 0,0075 / 0,0485 ; précision de c sous flou fort 0,858 / 0,838 / 0,805 | aucun oubli, seulement de la dérive : la mémorisation survit au bruit et la réparation la restaure |
| Effacer puis réparer : bloc 11 réinitialisé, 800 pas de réparation (blocs 10-11) | loss des images visées 0,001 ; KL(c) 0,032 ; flou fort 0,828 | idem |
| Vecteur « marge » appris avec AdamW au lieu de SGD | norme 4,0 (3 fois celle du vecteur d'entropie croisée), cosinus 0,29 avec lui ; négation α=1 : KL(a+b) 0,191, KL(c) 0,082 | non sélectif : la sélectivité vient de la pondération de l'entropie croisée (poids 1 − p par image), pas de l'optimiseur |
| Atténuation ciblée (type Selective Synaptic Dampening) : multiplier par Fisher_ref / Fisher_exact les poids dont l'importance sur les exactes dépasse 10, 30 ou 100 fois celle sur vues augmentées et mélanges | 1 231, 520 et 152 poids concernés sur 85 millions (0,001 %), presque tous dans le bloc 8 ; loss des images visées 0,000, dérive nulle | inopérant : aucun poids n'est propre aux images exactes |

Ces mesures recoupent le papier « Finding DoRI » du labo organisateur : la mémorisation est répartie, pas locale.

## 3. Mesure sur le gold (sans donnée externe)
Dans le forget set, une régression logistique validée hors échantillon sur les caractéristiques du bloc 8 retrouve la pseudo-étiquette à 66,5 % (loss 1,45, écart-type 2,56) ;
les 10 plus proches voisins partagent la classe à 22 %. `orig`, quand il se trompe sous changement de couleur fort, a une loss de 6,2.
Lecture (supposée, non mesurée) : un modèle qui n'a pas vu ces images se trompe souvent et avec confiance ; m_gold entre 1,5 et 2,5, s_gold proche de 3. La forme « queue lourde » de la négation va dans ce sens.
Les scores réels (issue #5) suivent m : 0,0010 (m 0,012), 0,0114 (0,087), 0,0219 (0,229, négation), 0,0241 (0,301), 0,0273 (0,376).

## 4. Somme de deux familles et version itérée
- Somme des deltas « cibles adoucies » de t3-methodes (`PGab_T85`) et de la négation (blocs 6-11), mesurée sur a+b / c (`logs/run8.log`) :
  PG seul : m 0,327, s 0,840, probabilité max T=2 0,804, KL(c) 0,107. PG + 2·négation : 0,475 / 1,276 / 0,767 / 0,150. PG + 4·négation : 0,676 / 1,878 / 0,724 / 0,222.
  Les deux deltas s'additionnent presque linéairement, sur l'oubli et sur la dérive. À m égal, la négation seule dérive 5 fois moins que les cibles adoucies (0,020 contre 0,107).
- Somme avec le meilleur delta de l'équipe (`PGab_T85_f20_l814c`, 0,1487 réel pour sa version complète ; `logs/sum3.log`) : seul m 2,045 / s 4,185 / KL(c) 0,148 ;
  + 4·négation(6-11) : 2,266 / 4,523 / KL(c) 0,287. Loss +9 %, dérive doublée : non rentable, aucun fichier livré.
- Version itérée sur les blocs 8-11 (NPO = perte bornée qui cesse de pousser une image une fois oubliée ; ancre KL sur mélanges et vues fortes).
  Il faut deux optimiseurs AdamW séparés (oubli / ancre) : dans un seul, le gradient d'oubli, saturé, est écrasé par l'ancre (loss d'oubli inchangée après 200 pas).
  | Pas d'oubli / pas d'ancre | Exactes : m / s / classes changées | Copies légères | KL(c) / accord | Précision de c sous flou fort |
  |---|---|---|---|---|
  | 3e-5 / 3e-5, 1 600 pas | 0,430 / 1,961 / 5,8 % | 0,217 / 1,198 | 0,0026 / 1,000 | 0,854 |
  | 6e-5 / 3e-5, seuil 0,8 | 0,622 / 3,008 / 6,1 % | 0,457 / 2,263 | 0,027 / 0,994 | 0,834 |
  | 6e-5 / 3e-5, seuil 1,2 | 0,848 / 3,411 / 7,9 % | 0,610 / 2,582 | 0,041 / 0,987 | 0,814 |
  | 6e-5 / 3e-5, fin | 1,242 / 3,828 / 12,6 % | 0,932 / 3,021 | 0,129 / 0,963 | 0,791 |
  | 1e-4 / 2e-5 (NPO) | 1,062 / 4,137 / 10,1 % | 0,971 / 3,673 | 0,255 / 0,928 | 0,789 |
  | 1e-4 / 2e-5 (montée de l'entropie croisée) | 3,293 / 7,649 | – | 1,280 / 0,759 | 0,648 |
  À oubli égal, le NPO à petit pas dérive 3 à 8 fois moins que la négation en un pas. Mais les bascules ciblées de t3-methodes font m 2,05 / s 4,19 pour KL(c) 0,148 : à dérive égale, 1,6 fois plus de loss.

## 4 bis. Score exact dans le simulateur (monde B de t3-simulateur, 3 golds, `logs/simB.log`, `simB2.log`)
Monde B : orig pré-entraîné puis affiné, 9 classes retirées en entier, gold de loss 3,73 / 5,35 (D = 9,1, l'échelle que demandent nos scores réels). Conditions du réel : orig + images sans étiquettes.

| Blocs, α | m / s (exactes) | Score de loss | Score de prédiction | Rétention | Final |
|---|---|---|---|---|---|
| tout, 0,5 | 0,03 / 0,56 | 0,06 | 0,13 | 0,988 | 0,008 |
| tout, 1 | 0,10 / 0,90 | 0,11 | 0,31 | 0,942 | 0,031 |
| tout, 1,5 | 0,36 / 1,50 | 0,21 | 0,46 | 0,822 | 0,079 |
| tout, 2 | 1,15 / 2,54 | 0,41 | 0,47 | 0,612 | 0,118 |
| 6-11, 2 | 0,20 / 1,06 | 0,14 | 0,43 | 0,875 | 0,053 |
| 6-11, 3 | 0,76 / 1,99 | 0,31 | 0,50 | 0,713 | 0,109 |
| 6-11, 4 | 1,74 / 3,00 | 0,52 | 0,43 | 0,553 | 0,125 |
| 8-11, 4 | 1,01 / 2,16 | 0,35 | 0,46 | 0,660 | 0,106 |
| tout, 4 ; 6-11, 8 ; 9-11, 8 | ≥ 5 / ≥ 2,9 | – | 0 | < 0,31 | 0 (porte des 70 %) |

Dans le même monde : distillation par quantiles calée sur l'envoi 1197 : 0,0146 ; oracle (cibles = vrais golds) : 0,508.
Lecture : maximum plat pour m entre 0,8 et 1,7 ; le score de prédiction atteint 0,43 à 0,50 dès qu'une queue d'images fausses existe ; la rétention descend à 0,55-0,7 et reste rentable ; au-delà la porte coupe tout.
Étalonnage : `NEGce_all_a2` réel (m 0,229, s 1,489) = 0,0219 ; le simulateur donne environ 0,05 au même m. Le réel paie 2 à 2,5 fois moins.
Limites : orig du monde B pré-entraîné (le vrai est entraîné de zéro), 9 classes retirées en entier (hypothèse non confirmée sur le réel) ; l'échelle de α ne se transpose pas, seule la cible sur m et s se transpose.

## 4 ter. Cibles et sources par image livrées à t3-methodes (`h200:~/eac/t3/t3-poids/targets/`, scripts `oof.py`, `adv.py`, `agree.py`)
- Sonde hors échantillon (régression logistique sur les caractéristiques d'`orig`, 5 plis × 3 tirages, chaque image prédite par des sondes qui ne l'ont pas vue). Précision contre la pseudo-étiquette selon le bloc : 6 → 0,345 ; 8 → 0,693 ; 10 → 0,991.
  `oofA_b8` : loss 1,15 / 2,10 (T=1), proche du gold « entraîné de zéro » du simulateur (0,70 ; 1,35 / 2,45). Réserve : toutes les images ont été vues par `orig` ; le niveau dépend de la profondeur, ce n'est pas une preuve que ces erreurs sont celles du gold.
- Plus petite perturbation adversariale (descente de la marge, norme L-infini) : médiane de bascule 0,875/255 ; la classe atteinte est la 2ᵉ classe d'`orig` dans 98,6 % des cas (pas de classe nouvelle) ; elle donne une mesure de fragilité par image.
- Accord sur la mauvaise classe (3 500 images ; référence par classe 0,12 à 0,15) : sonde / vues dégradées 0,223 ; sonde / adversarial 0,262 ; vues dégradées / adversarial 0,353. Au moins 2 sources d'accord : 54,1 % des images (`consensus_class.npy`).
- Accord sur les images difficiles (corrélation de rang) : vues dégradées / fragilité 0,44 ; fragilité / marge d'`orig` 0,43 ; la sonde est presque indépendante des trois autres (0,12 / −0,01 / 0,15).

## 5. Candidats livrés (H200 `/home/adem/eac/t3/t3-poids/candidates/`)
Tous : chargement strict, float32, 343,6 Mo, sortie fp16 finie (`chk.py`). Mesures sur les 3 500 images.

| Fichier | Recette | Exactes : m / s / classes changées | Copies légères | Prob. max T=2 | Score réel |
|---|---|---|---|---|---|
| `NEGce_all_a2.pt` | négation, modèle entier, α=2 | 0,229 / 1,489 / 4,1 % | 0,157 / 0,991 / 3,6 % | 0,919 | 0,0219 |
| `NEGce_all_hi_a4.pt` | négation, blocs 6-11 + tête, α=4 | 0,373 / 2,128 / 5,0 % | 0,235 / 1,442 / 4,2 % | 0,932 | à soumettre |
| `NEGce_all_top_a8.pt` | négation, blocs 9-11 + tête, α=8 | 0,523 / 2,613 / 6,9 % | 0,351 / 1,840 / 6,1 % | 0,915 | à soumettre |
| `PG85_NEGhi4_all.pt` | cibles adoucies PG_T85 + négation blocs 6-11, α=4 | 0,670 / 1,856 / 16,8 % | 0,541 / 1,471 / 14,8 % | 0,699 | à soumettre |
| `PG70_NEGhi4_all.pt` | cibles adoucies PG_T70 + négation blocs 6-11, α=4 | 0,839 / 1,595 / 25,9 % | 0,734 / 1,300 / 23,2 % | 0,508 | à soumettre |
| `NEGce_all_hi_a6.pt` | négation, blocs 6-11, α=6 | 0,640 / 2,890 / 8,4 % | 0,458 / 2,147 / 7,9 % | 0,887 | à soumettre |
| `NEGce_all_t8_a8.pt` | négation, blocs 8-11, α=8 | 0,714 / 3,126 / 8,9 % | 0,507 / 2,324 / 8,4 % | 0,895 | à soumettre (conseillé en premier) |
| `NEGce_all_top_a12.pt` | négation, blocs 9-11, α=12 | 0,926 / 3,526 / 12,5 % | 0,710 / 2,730 / 11,8 % | 0,863 | à soumettre |
| `NEGce_all_hi_a8.pt` | négation, blocs 6-11, α=8 | 0,955 / 3,452 / 14,0 % | 0,750 / 2,731 / 13,5 % | 0,829 | à soumettre (conseillé en second) |

Repères réels : `PG_T85_f20_l48c` (m 1,066, s 2,236) = 0,0735 ; `PG_T85_f20_l814c` (m 1,82, s 3,87, 20 % d'images basculées à loss 8-14) = 0,1487, meilleur de l'équipe au 21/11 21 h 30. Le vérificateur prédit 0,026 à 0,031 pour `hi_a8` / `t8_a8` (score de prédiction 0,16 contre 0,40 pour la famille PG) : ils n'ont pas été soumis.
Verdict sur la famille : sélective et peu coûteuse en dérive à faible oubli, dominée par les bascules ciblées au niveau de loss que le score récompense (m ≈ 2, s ≈ 4). Elle redevient utile si la rétention redevient le facteur limitant.
Ce que chaque envoi apprend : ces fichiers ont un écart-type de 3,1 à 3,5 pour un m sous celui de `PG_T85_f20_l48c`. Score ≥ 0,07 : l'écart-type est payé jusqu'à 3 au moins et la famille « poids », qui dérive moins, devient la base. Score sous 0,05 : s dépasse celui du gold, ou la classe fausse choisie par le modèle n'est pas celle du gold ; la négation reste alors un complément des cibles de t3-methodes (les deux sommes).

## 6. Limites (ce qui est supposé)
- La réserve c est faite d'images vues par `orig`, très sûres. Le jeu de contrôle caché est fait d'images jamais vues, moins sûres : la KL y sera plus grande que sur c (la critique estime 2 à 4 fois, par le calcul). Les colonnes « flou fort » et « couleur forte » de c donnent l'ordre de grandeur haut.
- c ne couvre que les 45 classes du forget set ; la dérive sur les 55 autres classes n'est mesurée nulle part.
- Les vecteurs appris sur les 3 500 images ont une norme 14 % plus grande que ceux appris sur a+b (54 pas par époque contre 36) : le même α y est un peu plus fort.
- Le simulateur n'offre que le monde B (orig pré-entraîné) ; le monde « entraîné de zéro » n'avait pas ses golds sur la H200.
- Signal ou bruit d'optimiseur : cosinus 0,76 à 0,77 entre vecteurs de graines différentes ; la négation de la moyenne de 3 graines donne le même oubli et la même dérive (0,361 / 2,069 / KL 0,0201 contre 0,356 / 2,110 / 0,0198). Le vecteur est du signal.

## 7. Fichiers
Code : `tracks/t3-unlearning/t3-poids/` (`tv.py` vecteur de tâche, `mkdelta.py` opérations par bloc, `loc.py` Fisher et ablation, `unl.py` pertes composables, `ssd.py` atténuation ciblée, `chk.py` contrôles de fichier).
Critique du tour 1 : `reports/critique/t3-poids-tour1.md`. Journaux : H200 `~/eac/t3/t3-poids/logs/`.
