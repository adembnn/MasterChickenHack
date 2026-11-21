# T3, critique : l'hypothèse des 9 classes entières (tour « classes entières »)

Décision débloquée : envoyer ou non les candidats « 9 classes basculées en entier, classes interdites en sortie », et quoi tester à la place.
Critique séparé, n'a écrit aucune méthode. Aucun calcul lourd, aucun appel au serveur. Calculs : quelques lignes de Python sur les chiffres des rapports.

Termes. **m, s** : moyenne et écart-type de la loss (entropie croisée sur la vraie classe, à T=1) sur les images notées. **S_loss** = 1 − (|m_sub − m_gold| + |s_sub − s_gold|) / (m_gold + s_gold), puisque `orig` a m ≈ s ≈ 0. **S_pred** = 1 − KL(p_gold‖p_sub) / KL(p_gold‖p_orig), sorties à T=2. **KL(p‖q)** : divergence de Kullback-Leibler, coût de prédire q quand la référence est p ; elle punit fort q ≈ 0 là où p a de la masse. **H9** : « le gold n'a jamais vu les 9 classes à 129-130 images ».

## Verdict en 5 lignes
1. H9 est réfutée par `Z9_b0` = 0 combiné à `G10` > 0, et ce raisonnement ne dépend ni de la loss visée ni de la forme exacte de S_pred (§1).
2. La relecture « Z9 mettait la loss très au-dessus du plausible » contredit vos propres mesures : Z9 a m = 3,21 sur le lot, soit environ 9,6 en moyenne sur les 1 169 images, c'est-à-dire la cible même des nouveaux candidats (10 ± 4,5).
3. Les candidats « 9 classes interdites en sortie » reproduisent Z9 et y ajoutent des bascules. À ne pas envoyer : la rétention perd environ 37 % (mesuré par G10) et la marge de porte est consommée.
4. Le levier le plus probable vers 0,40 est S_pred, pas S_loss : à f60, S_loss vaut déjà 0,89 à 0,94 sous tous les golds compatibles. Il reste S_pred × rétention ≈ 0,32 à 0,33.
5. Expérience n° 1 : répartir la masse fausse sur plusieurs classes voisines au lieu d'une seule, sans toucher à la loss vraie (§6).

## 1. H9 face à tous les scores
### Ce que H9 prédit pour le gold (images exactes)
Mélange : 33,4 % des images à loss L, 13 % d'erreurs ordinaires à 5 ± 2 et 53,6 % justes à 0,3 ± 0,4.
| Loss L sur les 9 classes | m_gold | s_gold |
|---|---|---|
| 9 ± 2 | 3,82 | 4,21 |
| 10 ± 4,5 (votre cible) | 4,15 | 5,18 |
| 14 ± 3 | 5,49 | 6,50 |
La cassure mesurée à s_gold ≈ 4,6 se retrouve exactement avec vos deux paires de loss (20 % : ratio prédit 1,149 contre 1,148 observé ; 30 % : 1,02 contre 1,02). Elle accepte H9 avec L ≈ 9 à 10 et peu de dispersion, mais rejette L = 14 (s = 6,5).

### La pente : elle ne tranche pas
S_loss calculé pour la série, et ce qu'il reste pour S_pred × rétention (score / S_loss) :
| Gold (m / s) | f30 | f40 | f50 | f60 | f70 prédit (S_loss) |
|---|---|---|---|---|---|
| H9, L = 9 : 3,82 / 4,21 | 0,83 → 0,235 | 0,89 → 0,27 | 0,95 → 0,29 | 0,92 → 0,33 | 0,88, en baisse |
| H9, L = 10 ± 4,5 : 4,15 / 5,18 | 0,73 → 0,27 | 0,82 → 0,29 | 0,87 → 0,32 | 0,91 → 0,33 | 0,85, en baisse |
| Gold qui connaît tout : 4,6 / 4,6 | 0,75 → 0,26 | 0,83 → 0,29 | 0,89 → 0,31 | 0,94 → 0,32 | 0,96, en hausse |
Les trois golds rendent la série de 30 à 60 %, avec un S_pred × rétention qui monte de 0,24 à 0,33 à mesure que les bascules couvrent les erreurs du gold. La pente n'apporte donc pas de preuve.
Limite : m et s sont mesurés sur les images exactes. Sur les copies, vos bascules ne passent qu'à 80-85 % (`PGho_f70` : 5,43 sur les exactes, 4,48 sur les copies), d'où un m effectif plus bas.

### Ce qui réfute H9 : Z9
- `Z9_b0` met à zéro les lignes de tête des 9 classes. Sur ces images, sa sortie est celle d'`orig` sans la vraie classe y, renormalisée ; sur les 66 % restantes, elle est identique à `orig`.
- Pour une image donnée : KL(g‖Z9) − KL(g‖orig) = p_g(y)·log(p_o(y)/p_Z9(y)) + (1 − p_g(y))·log(1 − p_o(y)). À T=2, la marge médiane d'`orig` est de 12 logits, d'où p_o(y) ≈ 0,99 et p_Z9(y) ≈ 0,003.
  Différence : −4,9 si p_g(y) = 0 ; −0,6 à 0,4 ; 0 à 0,46 ; +2,6 à 0,7.
- Si p_g(y) ≈ 0 (H9), Z9 est **strictement meilleur qu'`orig`** sur chacune des 1 169 images, quelle que soit la classe fausse du gold, que S_pred soit un rapport de moyennes ou une moyenne de rapports. Donc S_pred(Z9) > 0.
- Sous H9, S_loss(Z9) vaut environ 0,85 (d ≈ 1 à 1,5 pour d_orig ≈ 8 à 9). Il ne serait nul que si m_gold + s_gold ≤ 4, ce qui annulerait aussi f60 (0,30).
- La rétention ne s'annule pas (G10 : ×0,63) et la porte tient (G10, G10_f60, G10_f70 > 0).
- Conclusion : Z9 = 0 impose S_pred = 0 sur les deux groupes. Il faut donc qu'en moyenne p_gold(y) ≳ 0,45 à T=2 sur ces images : **le gold reconnaît ces classes**, et le jeu d'entraînement a plus de 130 images par classe.
- Le 130 est sans doute un plafond de tirage par classe dans la construction du lot (classes partielles de quelques images à 104, 9 au plafond), pas la taille d'une classe.
- Tension qui reste (et rend H9 séduisante) : le gold se tromperait sur 40 à 50 % du lot alors que la validation d'`orig` est ≥ 0,78. Une explication sans H9 : le gold est plus faible qu'`orig` (moins de données), et ses erreurs augmentent avec la part de la classe retirée. Cette version graduée est testable sans risque (expérience 2).
- Ce qui sauverait H9 : un défaut du fichier Z9. Contrôle local de 5 minutes : logits des 9 classes à 0 exactement, toutes les autres lignes identiques à `orig`, et 0 en fp16. Faites-le avant de clore le dossier.
- Observation qui la réfuterait encore, sans envoi : f70 (en cours). Sous H9 avec L ≈ 9-10, S_loss baisse de f60 à f70, donc f70 ≤ f60. Si f70 ≥ f60 + 0,015, H9 perd aussi la pente (prédiction : G10_f70 / 0,63 ≈ 0,32).

## 2. Si H9 était vraie : la meilleure cible
- La sortie qui minimise l'espérance de KL(p_gold‖q) est q = E[p_gold] : la moyenne de plusieurs golds, à T=2, est **douce** et partagée entre les 2 à 4 classes les plus proches, parce que chaque gold choisit sa voisine.
- Estimation de cette moyenne : la sortie d'`orig` à T=2 sans la vraie classe (c'est exactement Z9), éventuellement adoucie d'une température réglée sur la réserve. Une seule classe fausse quasi certaine coûte cher en KL dès que la voisine visée n'est pas celle du gold.
- Loss sur la vraie classe : 9 à 10, peu dispersée. Avec s_gold ≈ 4,6, un L = 10 ± 4,5 donne s = 5,2 et dépasse la cassure.
- Copies : même cible. Une classe jamais vue reste fausse sur toute copie. Il faut entraîner avec des copies fortes (recadrage fort, flou, couleur), puisque les copies cachées ont été durcies le 18/11.
- Interdire les classes en sortie n'est pas nécessaire pour atteindre cette cible, et c'est ce qui coûte en rétention (§3).

## 3. Classes partielles, jeu de contrôle, porte
- Le contrôle et la validation viennent de « la même distribution que l'entraînement » d'`orig`, qui contient ces 9 classes. Rien n'indique qu'elles en soient exclues. G10 montre que le contrôle couvre au moins les classes absentes du lot, à environ 1 % par classe.
- La rétention compare à `orig`, pas au gold. Rendre 9 classes impossibles coûte donc environ ×0,63 (G10 l'a mesuré), même si H9 était vraie.
- Porte : ces 9 classes pèsent environ 9 points de validation. La marge mesurée est de 8 à 10 points (G10_f60 et G10_f70 passent). « f70 + 9 classes interdites » est à la limite, et sur 70 % de validation cachée, un échec vaut 0 au classement final.
- Classes partielles : la rétention ne voit que des images hors lot. Ce qui compte est la dérive des images de contrôle de la même classe. Les bascules doivent rester propres à l'image (ancre sur des vues non vues de la même classe, déjà dans la recette).
  À mesurer : la KL de réserve par classe en fonction de la part de la classe dans le lot. Si elle monte pour les classes à 80-130 images, plafonner leur part de bascules.
- Ne jamais toucher aux 55 classes absentes : 10 % d'images de contrôle faussées coûtent environ 29 % du score (mesuré).

## 4. Ce que fait probablement une équipe à 0,40
- À f60, S_loss vaut 0,89 à 0,94 sous tout gold compatible. Le reste (S_pred × rétention ≈ 0,32 à 0,33) contient l'écart : 0,40 / 0,92 ≈ 0,43, donc **+30 % sur S_pred × rétention**.
- Les pistes probables, de la plus à la moins probable :
  - (a) La masse fausse est répartie sur plusieurs voisines selon `orig`, et non posée sur une seule classe.
  - (b) Les images non basculées ont une confiance proche de celle d'un gold moyenné. `PGho` (non basculées laissées tranchées) a perdu 10 %, donc l'adoucissement compte.
  - (c) La part de bascules suit la probabilité d'erreur du gold par image et par classe, au lieu d'un tirage uniforme.
  - (d) Le comportement passe aux copies (entraînement sur copies fortes), alors que vous perdez 15 à 20 % de m sur vos copies.
- Peu probable : une équipe qui interdit des classes, puisque la rétention plafonne alors à environ 0,63.

## 5. Robustesse sur les 70 % cachés
- La partie publique compte environ 1 050 images à oublier. s dépend de quelques valeurs très hautes, donc s_gold public ≠ s_gold caché à ±0,3 près. La cassure « 4,6 » est estimée sur la partie publique : viser s ≈ 4,4 à 4,6, pas pile 4,6.
- Choix de f : de 50 à 70 %, le score bouge de 0,03 au plus. Prendre le palier où la courbe est plate, pas le pic public (le meilleur de plusieurs scores bruités est biaisé vers le haut).
- La porte sur la validation cachée : tout fichier à moins de 8 points de marge (classes interdites, f ≥ 80 %) risque un 0 sur le classement final, même s'il passe en public.
- Question à poser aux organisateurs en personne (à Adem, pas par le serveur) : le classement final prend-il le meilleur envoi en score caché, ou l'envoi qui est le meilleur en public ? Dans le premier cas, des envois variés et sûrs sont gratuits. Dans le second, il faut finir sur le fichier le plus robuste.
- Sur-ajusté au public : la loss 8-14 contre 12-18 (gain de +0,004 à 30 %, du bruit), et f70 contre f60 si l'écart est < 0,01.

## 6. Expériences (au plus 5, la plus crédible d'abord)
1. **Masse fausse répartie.** f60, mêmes images et même loss vraie, mais hors de y la cible est la sortie d'`orig` à T=2 renormalisée sans y (températures 1, 1,5 et 2), au lieu d'une seule voisine. m et s sont inchangés, donc seul S_pred bouge. Adopter si ≥ +0,01.
2. **Part de bascules graduée par classe, à m et s constants.** Les 9 classes à 80-90 % et les partielles en proportion de leur effectif dans le lot, contre le tirage uniforme. Aucune ligne de tête touchée. Un gain franc (≥ +0,02) dit que l'erreur du gold croît avec la part retirée ; un résultat neutre ferme le dossier des classes entières.
3. **Lire f70 contre la prédiction (≈ 0,32).** Si f70 ≤ f60, s'arrêter à 60-70 % et ne pas tenter 80 %.
4. **Passage aux copies.** Ajouter des copies fortes des images basculées à l'entraînement, cible identique. Mesurer localement le m des copies : passer de 83 % à 95 % de celui des exactes, puis envoyer.
5. **Contrôle du fichier Z9 (local, sans envoi)**, puis abandon définitif de tout candidat « classes interdites ».
