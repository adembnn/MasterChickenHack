# Critique T3, session t3-methodes, tour 1 (21/11, ~21 h)

Décision débloquée : quels réglages de la méthode « pseudo-gold » (PG) changer et quels fichiers soumettre dans les prochaines heures.
Critique indépendante, lecture seule. Marques : **[mesuré]** = chiffre du tableau `reports/t3-tableau.md` ; **[code]** = lu dans `tracks/t3-unlearning/t3-methodes/` ;
**[supposé]** = mon raisonnement, non vérifié.

Termes : *KL(p‖q)* = divergence de Kullback-Leibler, coût de remplacer la distribution p par q ; KL(gold‖sub) est « couvrante » : elle punit fort un sub qui met
une probabilité quasi nulle là où le gold en met. *Membre* = image vue à l'entraînement d'un modèle. *Interpolation à α* = poids orig + α × (poids candidat − orig).

## 0. Ce que le code fait vraiment (vérifié)
- `mk_tgt.py` [code] : la « bascule » ne pousse pas la masse vers la classe voisine `cstar` (calculée puis jamais utilisée). Elle baisse seulement le logit de la
  pseudo-étiquette pour fixer la loss T=1 ; la masse restante suit q2 au carré. L'argmax devient bien `cstar`, mais la masse est étalée. Ce n'est pas un bug, c'est
  plutôt mieux pour la note de prédiction ; corrigez seulement la description.
- La loss imposée est linéaire de 6 (image la plus instable) à 2 : distribution uniforme sur [2, 6], moyenne 4 [code].
- Cible T=1 et T=2 issues d'un seul vecteur z = 2·log q2 [code] : cohérent (p1 ∝ p2²). Le défaut `make_q` du premier agent est bien corrigé.
- Sous-réalisation forte [supposé à partir du code + mesuré] : avec 15 % de bascules à loss moyenne 4, la cible vaut environ 0,63 de moyenne et 1,5 d'écart-type.
  Le modèle entraîné atteint 0,30 / 0,80 [mesuré]. Le modèle n'atteint donc que la moitié de la queue visée. Le tableau ne montre pas les moments de la cible
  (`meta['ce_mean']`, `ce_std`) ni `flip_ce` / `flip_touche` que `dash.py` calcule déjà : ajoutez-les, c'est le premier diagnostic à regarder.
- Ancre, réserve, formule de rétention (KL à T=2, précision ÷ précision d'orig, pseudo-étiquettes) : conformes à la consigne [code]. Entraînement en bf16 et
  évaluation en fp16 : sans effet attendu.

## 1. L'inversion des deux scores réels
Le calcul tient : 0,0114 = S_loss × S_pred × S_ret, avec S_loss entre 0,18 et 0,42, donne S_pred × S_ret entre 0,03 et 0,06. Mais la conclusion « le gold est étalé
et le sub trop pointu » n'est pas la seule possible. **Une note de prédiction faible apparaît aussi si le gold est SÛR de lui** [supposé] : quand le gold est
presque aussi pointu qu'orig, KL(gold‖orig) (le dénominateur) est petit, et tout écart du sub, même modeste, donne un rapport ≥ 1, donc une note nulle.
Les deux lectures prédisent l'inverse pour PG_T85 (très étalé, max-prob T=2 0,80). Autres explications à garder :
- S_ret bas si le jeu de contrôle est fait d'images **jamais vues** par orig (lecture de l'orchestrateur, `t3-analyse-orchestrateur.md` §1) et non de membres
  (votre lecture). La consigne ne tranche pas : « held-out control set », « preserve … on the rest of the dataset ».
- Copies modifiées cachées plus fortes que vos « copies légères » : orig y a déjà une loss non nulle, donc d_orig est plus petit et S_loss plus fragile.
- Définition de S_pred : rapport calculé image par image puis moyenné, ou rapport des deux moyennes ? Le PDF est illisible à cet endroit. Dans le second cas,
  quelques bascules mal placées à loss 6 dominent le numérateur. C'est l'argument principal pour adoucir les bascules (voir §4).
- La porte est déjà informée gratuitement : tout score > 0 signifie « porte passée ».
Départager en 3 ou 4 envois : voir les expériences E2 et E3. Le tableau (α 0,6 / 1,0) + « sans bascule » sépare « le gold est étalé » de « le gold est précis » et
« les bascules rapportent » de « les bascules coûtent ».

## 2. La réserve comme substitut du jeu de contrôle
- Bon substitut si le contrôle est fait de **membres restants des 45 mêmes classes** : la réserve vient du même jeu, orig y est saturé comme sur des membres.
  Biais probable [supposé] : plutôt **pessimiste**, parce que les images de la réserve sont de la même « famille » que les images oubliées (si le jeu à oublier
  a été tiré par classe ou par sous-groupe, ses voisines sont plus proches que la moyenne des images à garder).
- Si le contrôle est fait d'images **jamais vues** : la réserve est **optimiste**. Sur une image non saturée, le même déplacement de logits donne une KL plus forte, et
  l'accord à 1,000 ne garantit rien là où orig hésite déjà. Mesurez au moins la KL de rétention sur des vues dégradées de la réserve où orig tombe à ~85 %
  (`pg_logits` les fournit déjà pour orig) : c'est un substitut grossier d'images « non mémorisées ».
- **Les 55 classes absentes ne sont pas couvertes** : l'ancre « mélanges » ne montre que les 45 classes. Risque concret [supposé] : les cibles PG mettent de la masse
  sur des voisines qui peuvent appartenir aux 55 ; la tête apprend à relever ces lignes partout, donc sur le contrôle de ces classes. Seule mesure possible sans
  données : des images synthétisées à partir d'orig (inversion de modèle, voir E5). Surveillez au moins la norme et le biais de chaque ligne de la tête : moyenne
  et maximum du déplacement sur les 55 classes contre les 45.
- La réserve ne mesure qu'un modèle entraîné sur 2 802 images, pas le fichier final (3 500 images, mêmes pas) : petit biais, à signaler.

## 3. Les vues dégradées imitent-elles un non-membre ?
Biais :
- Une erreur due à la dégradation (flou : la texture disparaît, couleur : la teinte change) n'est pas une erreur de non-membre. Le gold se trompe surtout sur
  les images **atypiques** de leur classe, celles qu'orig a dû apprendre par cœur. Rien ne dit que « instable sous flou » = « atypique ». Votre propre mesure :
  l'instabilité sous augmentations est presque indépendante des deux autres mesures de difficulté (kNN, tête validée croisée) [mesuré par le 1er agent].
- Moyenne sur 20 *entrées* différentes, et non sur plusieurs *modèles* : l'étalement par famille de dégradation n'a pas de raison d'avoir la forme de l'étalement
  par graine.
- Classe-puits : vous l'avez contrôlée (aucune famille ne renvoie vers une classe fourre-tout) [mesuré]. Point en règle.
Meilleures estimations sans données externes :
- (a) quelles images le gold rate : **profondeur de prédiction** (*prediction depth*, Baldock et al. 2021) = couche à partir de laquelle le CLS intermédiaire,
  classé par un kNN en laissant l'image dehors, donne la bonne classe. Les exemples tardifs sont les atypiques, ceux qu'un modèle qui ne les a pas vus rate.
  Et **perturbation des poids** : du bruit gaussien sur les 4 derniers blocs, réglé pour qu'orig tombe à ~85 % sur les images exactes. Les images mémorisées
  basculent les premières (« memorization through curvature », Garg et al. 2023). Changer les poids imite mieux « une autre graine » que changer l'image.
- (b) vers quelle classe : la deuxième classe d'orig à T=2 sur l'image exacte (connaissance de second rang) croisée avec le vote des vues ; les deux sont gratuites.
- (c) moyenne et écart-type de la loss du gold : la consigne dit que std(L_gold) est **la moyenne des écarts-types de chaque modèle gold**, alors que p_gold est la
  **moyenne des distributions**. Le score de loss vise donc un modèle individuel (erreurs franches), le score de prédiction un gold moyen (étalé). Un ViT-B
  entraîné de zéro sur un petit jeu se trompe avec assurance : une loss moyenne 0,8 à 1,5 et un écart-type 2 à 2,6 (H3 et H4) sont plausibles [supposé].
  Le gold voit aussi moins d'images des 45 classes (on lui en retire jusqu'à environ un quart) : il est plus faible sur elles que sur la validation.

## 4. Le compromis bascules / prédiction
- Il est structurel [supposé] : S_loss veut environ 15 % d'erreurs franches (niveau d'un modèle individuel), alors que S_pred n'en veut qu'environ la moitié (le gold
  moyen ne change d'argmax que si la majorité de ses modèles se trompe). Avec p1 ∝ p2², une forte loss T=1 exige une vraie bascule.
- Ce qui coûte, ce n'est pas le nombre de bascules, c'est leur loss : une bascule à loss 6 met p2(y) ≈ 0,05. Si le gold garde p2(y) = 0,5 sur cette image,
  elle perd la moitié de sa note (rapport ≈ 1,15 / 2,4, ordre de grandeur [supposé]). À loss 2 à 3, la perte est bien moindre. Quant à l'écart-type,
  il dépend surtout de la *fraction* f de bascules : à moyenne et écart-type visés fixés, on a f ≈ moy² / (moy² + sd²), soit environ 0,11 pour H2.
- Mon choix : garder f entre 10 et 15 %, ramener la loss des bascules à 2,5 à 4, et compenser par une étape de plus pour que le modèle *atteigne* sa cible
  (aujourd'hui il n'en atteint que la moitié). Ne pas trancher à la main : E1 calcule ce compromis en une minute.

## 5. La porte à 70 %
Le flou fort de la réserve (0,817 contre 0,867) est un témoin faible : un seul type de dégradation, des images des 45 classes seulement, et une précision
absolue qui n'a pas de sens sans connaître la vraie précision d'orig sur la validation. Témoins meilleurs et gratuits :
- **Les envois déjà passés** : 1265 (`S1_r2`, 9,3 % de classes changées, KL hors domaine jusqu'à 1,1) a passé la porte [mesuré, score 0,0273]. Si PG_T85 dérive
  moins que lui sur chaque mesure (CIFAR, mélanges, flou fort), le risque de porte est faible. Repassez `S1_r2` dans `dash.py` pour avoir la même règle.
- Le rapport précision sub ÷ précision orig sur les 20 vues dégradées de la réserve (orig ~85 %) et sur CIFAR (accord d'argmax). Seuil prudent : 0,95.
- Repli prêt : l'interpolation (α 0,8 → flou fort 0,834).

## 6. Expériences suivantes (classées par gain ÷ coût)
**E1. Simulateur de score multi-pseudo-golds, hors ligne (CPU, environ 1 min, aucun entraînement).** Prendre les 5 familles de `pg_logits` comme 5 « modèles gold ».
S_loss visé = moyenne des moments par famille ; S_pred visé = moyenne des p2 des familles. Construire la cible avec 4 familles et noter contre la 5e
(validation croisée par famille, pour ne pas tourner en rond). Le faire pour f ∈ {0, 5, 10, 15, 25 %}, une loss de bascule dans {2–3, 2,5–4, 2–6} et les deux
définitions de S_pred (par image / rapport des moyennes). Ajouter les logits *réels* de PG_T85 et de ses interpolations.
→ Décide f et la loss des bascules. Si l'optimum est à f = 0 sous la définition « rapport des moyennes », c'est le signe que les bascules sont dangereuses.

**E2. Courbe α de PG_T85 au serveur (2 à 3 envois, aucun entraînement).** Envoyer α = 1,0, puis 0,6 ou 1,3 selon le premier résultat.
→ score(1,0) > score(0,6) × 1,5 : le gold oublie plus que nous ; on pousse (α 1,3, plus de pas, queue plus lourde). Score qui baisse avec α : le gold est plus
précis que prévu (lecture « gold sûr », §1) ; on passe à PG_T92. Score = 0 à α 1,3 : la porte est franchie, on reste à 1,0.

**E3. PG « sans bascule » (f = 0, même étalement), 1 entraînement + 1 envoi.** Seule S_loss baisse (pas de queue) ; S_pred et S_ret restent environ les mêmes.
→ Comparé à PG_T85 : si le score sans bascule est ≥, les bascules coûtent plus en prédiction qu'elles ne rapportent en loss. On passe alors à des bascules
plus douces, à loss 2 à 3. S'il est nettement plus bas, la queue paie : on la renforce. Avec E2, c'est le test qui sépare S_pred de S_loss.

**E4. Pseudo-gold par perturbation des poids (hors des réponses évidentes), environ 5 min de calcul.** Bruit gaussien sur les 4 derniers blocs, 8 tirages,
réglé pour qu'orig tombe à 85 % sur les images exactes (et non sur des vues). Mesurer : le recouvrement des 15 % d'images ratées avec celles des vues, la
corrélation avec la profondeur de prédiction, et la loss moyenne et l'écart-type obtenus. → Recouvrement fort (> 60 %) : la sélection des bascules est
crédible, on garde. Recouvrement faible : on construit les cibles sur la moyenne « poids bruités + vues », moins arbitraire qu'une seule famille. Les moments
mesurés donnent un 2e estimateur de (c).

**E5. Ancres et témoin pour les 55 classes absentes, par inversion de modèle (environ 10 min de génération + 1 entraînement).** Synthétiser 20 images par classe
en optimisant le bruit d'entrée pour la classe c (avec gigue et flou de régularisation). C'est l'esprit de *Zero-Shot Machine Unlearning*, cité par la consigne,
et ça ne demande que le modèle. Ajouter ces images à l'ancre « mélanges » et mesurer la KL(orig‖sub) sur des images tenues à part.
→ KL du candidat actuel sur ces images > 0,1 : la dérive sur les 55 classes coûte de la rétention ; on garde l'ancre élargie. KL < 0,03 : on abandonne cette piste.
