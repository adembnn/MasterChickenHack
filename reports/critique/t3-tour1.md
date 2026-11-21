# T3 Unlearning : critique, tour 1 (agent critique de Titouan, 21/11, 22 h)

Décision débloquée : quelles expériences lancer après 0,1962 (`PG_T85_f30_l814c`, soumission 1624), et lesquelles ne pas lancer.
Je n'ai écrit aucune des méthodes d'Adem et je n'ai rien envoyé au serveur. Lu : `reports/sources/Unlearning.txt`, gabarit,
`docs/BRIEF-T3.md`, `reports/t3-*.md`, `tracks/t3-unlearning/`, l'issue #5 en entier (jusqu'au point d'Adem de 21 h 35).
Non lus car pas poussés sur GitHub : `reports/t3-tableau.md`, `reports/critique/t3-verification.md`, `t3-poids-tour1.md`, `t3-sondes-tour1.md`.
Marquage : **[MESURÉ]** = sortie de commande (la mienne ou celle citée sur #5) ; **[SOURCE]** = article ou consigne ; **[DÉDUIT]** = calcul sur la formule ; **[SUPPOSÉ]** = non vérifié.

## 1. La formule exacte (consigne et gabarit, [SOURCE])
- **Oubli** = S_loss × S_pred, chacun moyenné d'abord sur deux groupes de même poids : les 3 500 images exactes et des copies retouchées cachées.
  - S_loss = 1 − clip(d_sub / d_orig, 0, 1), avec d = |moyenne(L) − moyenne(L_gold)| + |écart-type(L) − écart-type(L_gold)|.
    L = entropie croisée à **température 1** avec la **vraie** étiquette. moyenne(L_gold) et écart-type(L_gold) sont **calculés modèle par modèle puis moyennés** sur les golds.
  - S_pred = 1 − clip(moyenne KL(p_gold ‖ p_sub) / moyenne KL(p_gold ‖ p_orig), 0, 1), softmax à **température 2**, p_gold = **moyenne des distributions** des golds.
    C'est un rapport de **moyennes** (pas une moyenne de rapports) : le dénominateur est dominé par les images où le gold est sûr d'une autre classe qu'`orig`.
- **Rétention** = clip(acc_sub / acc_orig, 0, 1) × 1 / (1 + moyenne KL(p_orig ‖ p_sub)) sur un jeu de contrôle caché (KL à T=2).
- **Score** = oubli × rétention ; 0 si précision < 70 % sur une validation cachée. Public = 30 % des images (tirage par hachage, une copie suit son image).
- Ce n'est **pas** une attaque d'inférence d'appartenance : seulement deux moments de la loss + une KL par image vers la moyenne des golds.

Trois conséquences que je n'ai pas vues écrites dans les rapports :
1. **[DÉDUIT] La loss à T=1 est une fonction de la distribution à T=2.** Si p2 = softmax(z/2), alors L = −2·log p2(y) + log Σ_k p2(k)².
   On ne peut donc pas régler S_loss et S_pred séparément sur une image : toute loss gagnée se paie par une probabilité p2(y) plus petite.
2. **[DÉDUIT] Ce paiement est nul là où le gold met ~0 sur la vraie classe.** KL(p_gold ‖ p_sub) ne dépend de p_sub(y) qu'à travers p_gold(y)·log(1/p_sub(y)).
   Sur une image que le gold rate avec certitude, on peut monter la loss presque gratuitement ; sur une image que le gold réussit, chaque nat de loss coûte ≈ p_gold(y)/2 nat de KL.
   C'est pourquoi 0,0735 → 0,1487 en montant la loss des bascules : beaucoup d'images basculées sont des erreurs du gold (le vérificateur l'a lu dans le même sens).
3. **[DÉDUIT] Une seule classe fausse est optimale pour une loss donnée.** À loss T=1 fixée, p2(y) est maximal quand toute la masse fausse va sur une seule classe
   (convexité de Σ exp(a_k/2)). Étaler une bascule sur plusieurs classes voisines coûte plus de KL pour la même loss. Garder le choix actuel (une classe voisine).

## 2. Ce qui limite 0,1962 (lecture des points réels de #5)
- [MESURÉ, #5] Famille `PG` : 0,0241 → 0,0323 → 0,0735 → 0,1487 → 0,1962, avec score ≈ 0,022 à 0,027 × (m + s), sans cassure jusqu'à m 2,56 / s 4,29 (exactes).
  Tant que m < m_gold et s < s_gold, S_loss = (m + s)/(m_gold + s_gold) [DÉDUIT de la formule, orig ayant m ≈ s ≈ 0]. La cassure n'est pas encore vue.
- [MESURÉ, #5] Copies légères de `PG_T85_f30_l814c` : 2,11 / 3,69, soit (m + s) = 5,80 contre 6,85 sur les exactes : le groupe « copies » rapporte ≈ 15 % de moins en S_loss.
  Les copies cachées du serveur sont inconnues : si elles sont plus fortes que nos « copies légères », l'écart est plus grand.
- [MESURÉ, #5] Rétention de réserve : 0,85 (20 % de bascules) → 0,795 (30 %) → 0,716 (40 %). À 40 %, le gain de loss (+33 % de bascules) dépasse encore la perte de rétention (−10 %)
  si la loi linéaire tient, mais le vérificateur prédit un effondrement de S_pred au-delà de 40 % avec un choix d'images au hasard.
- **Lecture d'ensemble [DÉDUIT]** : le moteur actuel (plus d'images × loss plus haute) n'est pas épuisé ; le tableau public (0,27 en tête) est à ×1,4 de nous,
  ce qu'une m + s ≈ 9,5 donnerait si la loi tient. Le plus gros levier restant n'est pas la hauteur mais **le choix des images et de la classe** (point 2 ci-dessus) et les copies.

## 3. Ce que fait probablement l'équipe à 0,27 (aucune preuve possible)
- [SOURCE] Aucun article 2023-2026 lu par l'équipe ou par moi ne propose une méthode « sans données de rétention » réglée pour cette métrique (moments de loss + KL par image vers des golds).
  Les gagnants NeurIPS 2023 utilisaient tous un jeu de rétention (`reports/t3-communaute-et-combinaisons.md`, §2.2). MU-Mis (ICLR 2026) est sans rétention mais vise la sensibilité aux entrées, pas cette métrique.
- [SUPPOSÉ] Le chemin le plus court vers 0,27 est le nôtre poussé plus loin (plus d'erreurs confiantes ciblées, ancre sur mélanges). Je ne peux pas l'établir.
  Une piste que les équipes d'un laboratoire de confidentialité (« Incoming PhD ») connaissent : cibles par image venant de modèles « hors pli » (Local Teacher Distillation, LTD),
  c'est exactement `oofA_b8` de t3-poids. Elle n'a pas encore été envoyée.

## 4. Mesures faites par moi (h200c, venv privé `~/critique-t3`, poids et images officiels re-téléchargés de HuggingFace, rien d'autre)
Question : les « sélecteurs » d'images que le gold rate sont-ils d'accord entre eux ? Si oui, le choix actuel (instabilité sous vues dégradées) est probablement bon ;
sinon, le facteur 2 à 3 annoncé par le vérificateur est encore sur la table. Code et sorties : `tracks/t3-unlearning/critique/` (`sel.py`, `noise.py`, `adv.py`, `cons.py`, `sel.csv`, `sel_summary.json`).
Étiquette = classe prédite par `orig` (45 classes), mode évaluation, fp16 comme le serveur.

| Sélecteur (score par image) | Résultat [MESURÉ] |
|---|---|
| Marge d'`orig` (logit 1er − 2e) | moyenne 12,24 |
| Profondeur de prédiction (Baldock et al. 2021) : 1er bloc à partir duquel un kNN (k=10, cosinus, sans l'image elle-même) donne toujours la classe finale | kNN exact 11 % au bloc 0, 38 % au bloc 8, 87 % au bloc 10, 99,6 % au bloc 11 ; 2 268 images (65 %) ne sont lues qu'à partir des blocs 9 à 12 |
| Sonde logistique hors pli, bloc 8 (5 plis) | précision 0,709, loss 0,96 / 1,12 (t3-poids trouvait 0,693 : **reproduit** de façon indépendante) |
| Bruit gaussien dans les poids des blocs 6-11 (σ = 2 % à 50 % de l'écart-type de chaque matrice) | **aucune bascule** jusqu'à 15 % ; 0,02 % à 50 % : inutilisable comme sélecteur |
| Plus petite perturbation adversariale L∞ (PGD 7 pas) | bascule à 0,25/255 : 28 images ; 0,5 : 528 ; 1 : 1 979 ; 2 : 930 ; ≥ 4 : 35 |

Accords entre sélecteurs (recouvrement des 20 % « les plus durs » ; **le hasard donne 0,20**) :
- marge & profondeur 0,31 ; marge & sonde 0,26 ; profondeur & sonde 0,39 ; adversarial & marge 0,44 ; adversarial & profondeur 0,19 ; adversarial & sonde 0,20.
- Corrélations de rang : profondeur ~ sonde 0,48 ; marge ~ sonde −0,15 ; marge ~ profondeur −0,12 ; adversarial ~ marge 0,38.
- Classe fausse : l'attaque adversariale atteint la **2e classe d'`orig` dans 84 %** des cas ; la sonde hors pli, quand elle se trompe, ne tombe sur la 2e classe d'`orig` que dans 28 % ;
  kNN et sonde, quand les deux se trompent, ne choisissent la même classe fausse que dans 30 %.

Ce que j'en déduis :
1. **Les sélecteurs sont presque indépendants** : chacun voit une autre « difficulté ». Le choix des images est donc très ouvert, et un seul envoi par sélecteur tranchera plus que des réglages de hauteur.
2. **La piste « plus petite perturbation adversariale » (point 4a d'Adem) n'apporte presque rien de neuf** : elle redonne la marge et la 2e classe d'`orig`. Ne pas y dépenser de GPU.
3. **Le bruit dans les poids ne révèle rien** : la mémorisation de ces images est robuste, cohérent avec « Fisher non localisé » de t3-poids.
4. Les familles « caractéristiques internes » (profondeur, sonde hors pli) s'accordent entre elles (0,39-0,48) mais pas avec les familles « sortie d'`orig` » (marge, adversarial, et probablement l'instabilité actuelle, non recalculée ici).
   Si le gold rate surtout les images « lues tard » par le réseau (ce que dit Baldock et al. : la profondeur prédit l'erreur d'un modèle qui n'a pas vu l'image [SOURCE]), le consensus des deux premières est le meilleur pari.
5. Fichier livré : `sel.csv` (3 500 lignes, ordre de `forget_images.npz`) avec, par image, marge, profondeur, classe fausse kNN, prédiction hors pli, `consensus`
   (moyenne des rangs profondeur + loss hors pli + marge inverse) et `vote_class` (classe fausse votée entre 2e d'`orig`, kNN, sonde ; vote à 2 voix ou plus pour 42 % du top 30 %).

Bruit du classement public [MESURÉ par simulation, hypothèse : la distribution de loss de `f20_l814c`, 19,9 % d'images à loss ≈ 9,1, les autres ≈ 0] :
sur 4 000 tirages d'un public de 30 %, le rapport (m + s) public / privé a un écart-type de 0,045 (IC 95 % : 0,91 à 1,09).
**Deux fichiers à moins de 10 % l'un de l'autre au public ne sont pas départagés** ; le classement final peut les inverser. Préférer, à score public proche, celui qui retient le mieux.

## 5. Expériences proposées (au plus 5), classées par gain attendu
**E1. Trio de sélecteurs à budget égal, avec le consensus en plus** (étend l'expérience déjà prévue `PS30_rand / PS30_oof / PS30_err`).
- Quoi : même recette que 0,1962 (30 % de bascules, loss 8-14, ancre mélanges), seul le choix des images change : (a) instabilité (actuel, = 0,1962), (b) top 30 % de `consensus` de `sel.csv` avec `vote_class`, (c) au hasard.
- Ce que ça tranche : si (b) > (a) de plus de 10 % (seuil de bruit ci-dessus), les erreurs du gold suivent les caractéristiques internes et pas la sortie d'`orig` → toutes les recettes suivantes passent sur ce sélecteur.
  Si (c) ≈ (a), le choix ne compte pas et seule la hauteur compte.
- Coût : 2 entraînements de la recette existante (≈ 20 min H200 chacun d'après #5) + 2 créneaux. Validation locale sans envoi : impossible sur le gold ; on vérifie seulement porte et rétention de réserve
  (doivent rester ≥ celles de `PG_T85_f30_l814c` : 0,795) et le recouvrement des deux jeux d'images (s'il dépasse 0,7, ne pas envoyer : l'envoi n'apprendrait rien).

**E2. Bascules graduées : hauteur de loss proportionnelle à la certitude** (piste hors des réponses évidentes, tirée de §1 point 2).
- Quoi : même nombre d'images et même m + s que le meilleur, mais la loss visée de chaque bascule suit la certitude que le gold la rate (par exemple 14-18 si les 3 sélecteurs sont d'accord, 6-8 si un seul) ;
  et sur les images douteuses non basculées, garder p2(y) ≈ 0,5-0,7 plutôt que 0,98.
- Ce que ça tranche : à S_loss égal, gagne-t-on en S_pred en déplaçant la loss vers les images les plus sûres ? Prédiction [DÉDUIT] : gain positif si le sélecteur est meilleur que le hasard, nul sinon.
- Coût : 1 cible à fabriquer (`mk_tgt.py`, quelques minutes) + 1 entraînement + 1 créneau. Validation locale : m, s identiques au meilleur à ±5 % sur exactes et copies ; rétention de réserve ≥ 0,795.

**E3. Rétention sans perte d'oubli : moyenne de poids de graines, ou delta limité aux blocs hauts.**
- Quoi : (a) moyenner les poids de 2 à 3 entraînements de la même recette avec graines différentes (les cibles sont identiques, donc l'oubli voulu est commun ; la dérive de chaque graine sur le contrôle est en partie aléatoire et se moyenne) ;
  (b) entraîner seulement les blocs 8-11 au lieu de 4 derniers blocs + tête, ou projeter le delta existant sur les blocs 8-11 (t3-poids a mesuré 6 fois moins de dérive à oubli égal en restreignant la négation aux blocs hauts).
- Ce que ça tranche : la rétention (0,795) peut-elle remonter vers 0,85-0,9 sans toucher m et s ? Gain borné à ≈ +10 % du score.
- Coût : (a) 1 à 2 entraînements de plus ; (b) zéro entraînement pour la projection (minutes). Validation locale complète sans envoi : m / s sur exactes et copies (≥ 95 % du meilleur), KL et précision de la réserve de 698 images.
  N'envoyer que si les deux tiennent.

**E4. Bascules robustes aux copies.**
- Quoi : appliquer pendant l'entraînement des bascules les retouches plausibles du serveur (recadrage, rotation légère, JPEG, flou léger, couleur douce) ; t3-methodes prépare déjà les rotations (`PGr`).
- Ce que ça tranche : le groupe « copies » (50 % du score) passe-t-il de m + s = 5,80 au niveau des exactes (6,85) ? Gain borné à ≈ +8 % de S_loss moyen [DÉDUIT : (6,85 − 5,80)/(2 × 6,85)].
- Coût : 1 entraînement. Validation locale : m + s sur un jeu de copies **tenu à l'écart des retouches d'entraînement** (sinon la mesure est circulaire), rétention de réserve ≥ 0,795.

**E5. Distillation vers la cible hors pli `oofA_b8` (LTD), en mélange avec la recette actuelle.**
- Quoi : sur les images où la sonde hors pli se trompe (29 %, recoupé : 0,709 chez moi), prendre sa classe fausse (et non la 2e d'`orig`) et la loss 8-14 de la recette ; le reste inchangé.
- Ce que ça tranche : la classe fausse du gold ressemble-t-elle plus à celle d'une sonde qui n'a pas vu l'image qu'à la 2e classe d'`orig` ? (Elles ne coïncident que dans 28 % des cas, donc l'envoi est informatif.)
- Coût : 1 cible + 1 entraînement + 1 créneau. Validation locale : comme E1. Recoupe en partie E1 (b) : à fusionner avec lui si les créneaux manquent.

## 5 bis. Mise à jour 22 h 15 : si le gold du vérificateur est juste (m 2,1-2,7 / s 3,5-4,2, précision 0,57-0,62)
Nous sommes à m 2,56 / s 4,29 sur les exactes : **m est dans la fourchette, s est déjà au-dessus** [DÉDUIT de ces deux chiffres].
- Le score de loss est alors proche de son plafond sur les exactes (≥ 0,8 avec D = m_gold + s_gold ≈ 6). Monter la hauteur ne rapporte plus : chaque point de s en trop retire autant qu'un point de m rapporte.
  **`PG_T85_f20_l1218c` et le croisement 30 % × 12-18 devraient baisser ou stagner** (ils montent s de ≈ 1,5 à 2 points). À n'envoyer que comme sondes du plafond, pas comme candidats.
- Rapport s/m du gold ≈ 1,6. Pour une loss en deux paliers (fraction f d'images à loss h, le reste ≈ 0), s/m = √((1 − f)/f), donc f ≈ 1/(1 + 1,6²) ≈ 0,28 et h ≈ m_gold/f ≈ 7,5 à 9,6 [DÉDUIT].
  **La recette actuelle (30 %, loss 8,5) est déjà calée sur ce gold.** Il ne faut plus bouger ces deux réglages, sauf sur les copies, où m + s reste à 5,80.
- D'où vient alors l'écart 0,20 → 0,27 ? Avec S_loss ≈ 0,8-0,9 et rétention ≈ 0,8 (réserve), 0,1962 implique **S_pred ≈ 0,27 à 0,31** [DÉDUIT : 0,1962 / (0,85 × 0,8)].
  C'est désormais le facteur limitant. Il dépend uniquement de **quelles** images basculent et **vers quelle** classe, et de la douceur du reste.
- Nouvel ordre conseillé : **E1** (choix des images, à 30 %/loss 8-14 fixes), **E5** (classe fausse hors pli), **E2** (bascules graduées : garder m et s, déplacer la loss vers les images sûres),
  puis **E4** (copies : seul endroit où la loss manque encore), **E3** (rétention, validable sans envoi). Gain attendu d'E1 + E5 si le sélecteur gagne : de ×1,2 à ×1,5 sur S_pred,
  soit 0,24 à 0,29 [DÉDUIT, borne haute = le facteur 2 à 3 du vérificateur ramené à la part déjà acquise].
- Rétention : la porte n'est pas en jeu à 30 % (marge ≥ 10 points mesurée par `G10_PG_T70`). Ne pas monter à 40 % (rétention de réserve 0,716).

## 6. À ne pas faire (mesuré)
- Attaque adversariale comme sélecteur ou pour choisir la classe fausse : redonne la 2e classe d'`orig` (84 %) [MESURÉ ici].
- Bruit dans les poids comme sélecteur : aucun signal [MESURÉ ici].
- Étaler une bascule sur plusieurs classes voisines : plus de KL pour la même loss [DÉDUIT §1].
- Départager deux fichiers à moins de 10 % au public : bruit [MESURÉ par simulation].
- Déjà réfutés par des envois (#5) : adoucir tout le modèle (0,0057), « 9 classes oubliées » (0,0), ancre sur vues très dégradées telle quelle (0,0916).

## Sources
- Consigne `reports/sources/Unlearning.txt`, gabarit `reports/sources/t3/task_template.py`.
- Baldock, Maennel, Neyshabur, « Deep Learning Through the Lens of Example Difficulty », NeurIPS 2021, https://arxiv.org/abs/2106.09647
- Maini et al., « Can Neural Network Memorization Be Localized? », ICML 2023, https://arxiv.org/abs/2307.09542 (mémorisation portée par quelques neurones ; non testé ici)
- MU-Mis (sans rétention, sensibilité aux entrées), cité dans `reports/t3-unlearning-etat-de-lart.md`.
- Chiffres de l'équipe : issue #5 (commentaires jusqu'à 21 h 35).
