# Critique T3, session « t3-nonmembres », tour 1

Décision débloquée : peut-on donner aux sessions d'entraînement une cible chiffrée (loss moyenne, écart-type, précision, distribution par image)
tirée des images générées, et peut-on s'en servir à l'entraînement ?
Vérifié sur les fichiers `E:\EAC-lab\t3\t3-nonmembres\meas\logits_*.npz` (numpy, CPU), code `code\*.py`, 3 planches regardées (i2i 0,70, IP-Adapter 0,6, FLUX texte-vers-image).

## 1. Failles de raisonnement ou de mesure

**F1. La « précision 1,000 » sur le forget set est une tautologie.** `prep.py` prend comme étiquette la prédiction d'orig (l'énoncé ne donne pas d'étiquettes).
La précision d'orig sur ses propres pseudo-étiquettes vaut 1 par construction. Toutes les mesures sur images générées utilisent aussi cette pseudo-étiquette.
Sans gravité si orig a 100 % sur ses données d'entraînement (probable), mais à écrire : ce n'est pas une mesure.

**F2. L'étiquette de la source ne survit pas toujours à la génération, et les classes sont très fines.** Les planches montrent des familles de classes voisines :
c5, c6, c7, c8 sont toutes des lanternes ; c19 et c21 deux plantes grasses proches ; c23 et c25 deux ponts ; 9-11 théières, 13-17 champignons (notes).
La classe semble définie par des détails (matière, couleur, style). Preuve chiffrée : sur le forget set, la seule augmentation `couleur_.4.4.4.1` fait tomber orig
à 0,538 de précision (loss 3,0 / 4,3), alors que flip, bruit léger et JPEG q50 ne changent rien. En texte-vers-image, la planche montre des changements de classe
réels (c6 : une lanterne devient des théières ; c12 : une tasse de thé devient un café noir ; c8 : une lanterne de pierre devient un clocheton).
Donc une partie des « erreurs » mesurées sont des erreurs du générateur, pas d'orig. Ce biais grandit avec la force et domine en texte-vers-image (0,33).
- Mesure : les erreurs tombent dans une des 45 classes oubliées dans 68 % des cas (i2i 0,70 et IP-Adapter), contre 45 % attendus au hasard ;
  73 % des erreurs i2i 0,70 tombent dans le 2e à 6e choix d'orig sur l'image source (la note « 30 à 42 % en 2e choix » sous-estime cette structure).
  Les confusions se font entre classes sœurs, toutes deux dans le forget set.
- Conséquence sur le gold : le gold a perdu les images d'entraînement des deux classes sœurs à la fois. Orig, lui, a appris la frontière entre elles avec ces
  images. « Orig sur non-membre » surestime donc probablement la précision du gold sur ces familles : la fourchette 0,62–0,80 est plutôt à lire par le bas.

**F3. Le choix de la force est circulaire, et la calibration du simulateur transfère mal.**
- Entre i2i 0,50 (0,97 / 0,09 / 0,55) et i2i 0,70 (0,80 / 0,88 / 2,15) la loss moyenne est multipliée par 10 : la cible dépend presque entièrement de la force,
  et la fourchette annoncée (0,9–2,1 / 2,1–3,5) revient à « entre i2i 0,70 et IP-Adapter ». Rien hors du simulateur ne fixe ce choix.
- Le simulateur a des classes ImageNet grossières (SD-1.4) ; la vraie tâche a des classes sœurs définies par la couleur et le style. Ce que le générateur
  « dégrade à lui seul » (mesure b de la calibration) n'a aucune raison d'être le même : SDXL change la palette et le rendu, ce qui coûte peu sur « bretzel »
  et beaucoup sur « lanterne verte contre lanterne blanche ».
- La calibration en cours compare M sur variation(membre) à M sur vrai non-membre. Elle ne teste pas la chaîne réellement utilisée : orig_sim sur
  variation(forget_sim) comparé à gold_sim sur forget_sim (voir expérience 1).

**F4. Les images générées ne disent rien de *quelles* images le gold rate.** Corrélation de rang (Spearman) de la loss par image :
i2i 0,70 contre IP-Adapter 0,18 ; i2i 0,70 contre FLUX t2i 0,15 ; IP-Adapter contre t2i 0,13 ; corrélation par classe de la précision 0,11 à 0,31.
Seul i2i 0,50 contre FLUX i2i 0,60 atteint 0,50 (mêmes images presque intactes). La loss d'une image générée dépend surtout du tirage du générateur.
Toute « cible par image » tirée d'un seul générateur est du bruit ; seules les statistiques globales (moyenne, écart-type, proportion d'erreurs) transfèrent.

**F5. Deux cibles différentes sont mélangées.** Le score de loss compare la moyenne et l'écart-type de L *par modèle gold* (puis moyennés) : un seul modèle,
bimodal, sûr de lui même quand il se trompe. Le score de prédiction compare à la *moyenne des distributions* des golds : une distribution plus douce
(simulateur : 73,5 % d'accord entre deux golds, probabilité max moyenne à T=2 de 0,64). Orig sur une image générée imite un modèle seul ; la moyenne sur plusieurs
générateurs imite mieux p_gold (ensemble i2i 0,70 + IP-Adapter + t2i : précision 0,81, probabilité max T=2 0,55, vraie classe 0,51). Il faut publier les deux.

**F6. Pondération.** Le sous-ensemble prend 20 images par classe, le forget set va de 46 à 130 par classe. Repondéré par effectif : i2i 0,70 passe de
0,88 / 2,15 à 0,96 / 2,30, IP-Adapter de 2,06 / 3,49 à 2,13 / 3,49. Écart faible mais systématique ; à corriger dans les chiffres annoncés.

## 2. Ce que l'estimation ne dit pas encore (et que les autres sessions attendent)
- **Copies modifiées (la moitié du score).** Déjà mesurable dans les npz : moyenne des alt0–alt3 sur i2i 0,70 = 1,09 / 2,45, IP-Adapter 2,14 / 3,48,
  contre 0,88 / 2,15 et 2,06 / 3,49 sans modification. Donner une cible séparée « exactes » et « copies ».
- **Forme de la queue.** Un écart-type de 2 à 3,5 avec une moyenne de 1 à 2 exige 15 à 25 % d'images à loss > 5 (i2i 0,70 : 6 % au-dessus de 5, 54 % sous 0,01 ;
  IP-Adapter : 18 % et 35 %). C'est la consigne utile : « X % d'images confiamment fausses, le reste quasi sûr », pas « baisser la loss partout ».
- **Vers où vont les erreurs.** Le fichier `_pT2_moyen_par_classe` contient une matrice classe → distribution moyenne : c'est la cible de prédiction la
  plus robuste (par classe, vers les classes sœurs), alors que la cible par image n'existe pas (F4).
- **Sensibilité du score à l'erreur sur la cible.** On ne sait pas si viser 1,0 au lieu de 1,8 coûte 5 % ou 60 % du score de loss.
- **Recoupement avec le leaderboard.** Sous la pseudo-référence « ensemble de 3 générateurs », un adoucissement d'orig à température effective 4 aurait un score
  de prédiction de 0,60 ; à 32, encore 0,20. Le vrai adoucissement global a donné 0,0039 au total : le facteur limitant est donc le score de loss
  (loss 0,14 face à une cible de 1 à 2), ce qui est cohérent avec l'estimation mais ne la valide pas.

## 3. Expériences proposées (la plus utile d'abord)
1. **Test direct de la chaîne dans le simulateur** (tranche F3). orig_sim sur variation_s(forget_sim) pour s ∈ {0,5 ; 0,6 ; 0,7 ; IP-Adapter} comparé à
   gold_sim sur forget_sim : écart de moyenne, d'écart-type, KL par image, et score complet que donnerait « orig_sim sur variation » pris comme sous-modèle.
   Ajouter gold_sim sur variation_s(forget_sim) pour isoler la part du générateur. Coût : ~1 h de H200 (900 images × 4 forces), à annoncer sur le bus.
2. **Sensibilité du score à une cible fausse** (tranche « la fourchette suffit-elle ? »). Dans le simulateur, sur les logits existants des golds et d'orig_sim,
   fabriquer des sous-modèles synthétiques (mélange orig / gold, erreurs sûres injectées sur k % des images) et tracer le score en fonction de l'écart à la
   vraie (moyenne, écart-type). Coût : minutes CPU, aucun GPU.
3. **Cible par classe plutôt que par image** (hors évidence ; tranche F4). Construire p_cible(classe) depuis la matrice des confusions générées entre classes
   sœurs, puis vérifier dans le simulateur si cette cible par classe donne un meilleur score de prédiction qu'une cible par image tirée d'un générateur, et
   qu'un adoucissement uniforme. Coût : minutes CPU sur logits existants (+ ceux de l'expérience 1).
4. **Mesure du biais propre au générateur dans le vrai domaine** (tranche F2 sans simulateur). Faire passer les images i2i 0,70 de façon *aller-retour*
   (générer depuis l'image générée vers une seconde variation) et regarder la baisse de précision supplémentaire ; ou faire étiqueter 100 paires source /
   générée par un humain (« même classe ? ») sur les familles lanternes et plantes. Donne la part des erreurs qui sont des changements de classe réels.
   Coût : 15 min de GPU ou 20 min d'un humain.
5. **Une soumission dédiée qui mesure la cible** (coûte un créneau ; décision d'équipe). Partir du meilleur candidat (loss 0,376, 9,3 % de classes changées)
   et ne changer qu'un levier : porter la proportion d'images confiamment fausses à ~20 % sans toucher le reste. Si le score monte nettement, l'écart-type
   cible est bien vers 2–3,5 ; sinon la fourchette est trop haute. C'est la seule mesure qui touche le vrai gold.

## 4. Images générées comme données d'ancrage ou cibles d'entraînement
- L'énoncé dit : « The unlearning method has to work from the original model and the forget set alone ». SDXL, FLUX, Florence-2 et IP-Adapter apportent des
  données externes (leurs poids ont été appris sur des milliards d'images). S'en servir *pour entraîner* le modèle soumis (images d'ancrage, cibles de
  distillation) est en dehors de la lettre de la règle, même si les images partent du forget set. Risque réel : le pitch et l'éventuelle relecture du code
  pour les 6 premières équipes.
- Usage défendable : **mesure seulement** (choisir une force, une température, un nombre d'époques, une cible de loss), comme on choisirait un
  hyperparamètre. Les méthodes soumises doivent rester reproductibles avec orig + forget set + transformations sans modèle externe (recadrage, couleur,
  mélange d'images du forget set entre elles).
- Recommandation : ne pas entraîner sur ces images ; si l'équipe veut le faire, poser la question aux organisateurs (issue `decision`), réponse écrite avant usage.
