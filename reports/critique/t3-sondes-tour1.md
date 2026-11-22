# Critique T3 · session `t3-sondes`, tour 1

Décision débloquée : faut-il encore viser « 9 classes oubliées en entier », et quel envoi trancher ensuite (porte, écart-type du gold).
Critique indépendant, 21/11 vers 20 h 15. Lu : `reports/t3-t3-sondes.md`, `sim.py`, `fit.py`, `explore.py`, `explore9.py`, `mk_bank.py`, `mk_zero.py`, `observations.json`, `journal.md`, issue #5. Calculs numpy en local sur `E:\EAC-lab\t3\t3-sondes\logits\*.npz` et `bank.npz`. Aucun envoi, aucun GPU.

Fait nouveau, absent du rapport de la session : **`Z9_b0` a été noté 0,0 (soumission 1397)**. Le simulateur prévoyait au moins 0,16 sous l'hypothèse (voir §3).

## 1. La formule dans `sim.py` : conforme, avec 4 écarts
Conforme : `group()` (l. 84) prend la moyenne sur les golds de la moyenne et de l'écart-type par gold de la loss, comme le texte officiel. Le score de prédiction (l. 87) est un rapport de **moyennes** de KL(p_gold‖p), avec p_gold = moyenne des softmax à T=2 (`Bank.gold`, l. 71). L'oubli (l. 105) fait la moyenne des deux groupes avant le produit loss × prédiction. La rétention (l. 96) vaut ratio de précision tronqué à 1, divisé par 1 + KL(p_orig‖p_sub) à T=2. Les copies suivent leur image source dans le tirage public (l. 142).
Écarts :
- **E1, le substitut du jeu de contrôle (l. 93-96), le plus lourd.** Le jeu de contrôle officiel est fait d'images hors forget set. Le simulateur le remplace par des copies des images à oublier (`held_*`). Il mesure donc la dérive *sur les images qu'on a fait oublier*, pas ailleurs. Résultat : rétention simulée de 0,44 à 0,87, contre 0,90 mesuré par `t3-methodes` sur sa réserve pour `PG_T85` (le simulateur lui donne 0,69). Sur `held_*`, `a_o` est en plus l'accord d'orig avec sa propre étiquette (≈ 1), pas une précision.
- **E2, l. 126 :** avec `q9` et `ctrl='mixes'`, `self.o.y[i]` indexe les étiquettes du forget set avec des indices de `mixes`. Le résultat est faux, mais l'erreur n'est pas déclenchée aujourd'hui (explore9 n'utilise que `held_*`).
- **E3, l. 99-108 :** `score()` ignore `q9` alors que `Fast.ret()` l'applique. Les deux chemins ne calculent pas la même chose.
- **E4, `fit.gold_stats` :** ce qui est appelé « précision » est la part d'images où la loss est sous log 2 (p_vraie > 0,5), pas un argmax. L'écart est faible, mais le nom induit en erreur dans le rapport.
Détail sans effet : `np.std` sans correction (ddof 0) contre `torch.std` (ddof 1) sur environ 1 000 images, soit 0,05 %.

## 2. Fragilités du raisonnement
**Lectures concurrentes de « 8 × 130 + 1 × 129 » (sur les étiquettes d'orig, sûres ici : probabilité max ≥ 0,925 sur les 3 500 images).**
- (a) Thèse de la session : 130 = taille de chaque classe à l'entraînement (jeu de 13 000 images), donc 9 classes vidées.
- (b) Plafond de tirage : le générateur répartit 3 500 images sur 45 classes avec un maximum de 130 par classe et reverse l'excédent sur les autres. Les classes plafonnées s'empilent à 130 et l'arrondi donne un 129. Ce schéma explique aussi le trou entre 104 et 129 : les poids des classes au-dessus du plafond sont simplement tronqués.
- (c) 130 = 10 % d'une classe de 1 300 images, la taille usuelle d'une classe ImageNet, donc un quota relatif.
- (d) Jeu déséquilibré : ces 9 classes font 130 images, les autres plus.
Le 129 de la classe 6 affaiblit (a) : aucune image n'a 6 comme second choix avec une marge serrée (marge minimale 4,2 logits). Il ne manque donc pas une image « mal lue » : sous (a), la classe 6 garde une image d'entraînement. Les effectifs seuls ne départagent pas (a) de (b) à (d). Seuls les scores réels le peuvent.
**Ce qui départage déjà : `Z9_b0` = 0.** Sous (a), le gold a perdu ces classes en validation comme nous, et la porte des 70 % (précision minimale exigée sur la validation) a été réglée pour qu'il passe, donc `Z9` passe aussi. De plus m_gold ≈ 0,33 × (loss des 9 classes ≥ 10) ≥ 3,3, donc notre score de loss est positif. Un 0 exact n'est compatible avec (a) que si la porte a tranché. Or c'est possible : `t3-poids` mesure que le contenu ne prédit la classe qu'à 66,5 % (régression sur les caractéristiques du bloc 8), donc la précision d'orig en validation pourrait être entre 0,72 et 0,80. Avec −9 % elle tomberait sous 0,70. **La thèse est très affaiblie, pas réfutée, tant que la porte n'est pas mesurée.**
**Le gabarit de gold tiré d'orig biaise le calage, dans trois sens mesurables.**
- Moyenne et écart-type sont liés dans la banque. Mesuré : niveau 3 → loss 1,60 / écart-type 3,34 ; niveau 4 → 2,39 / 3,97 ; niveau 5 → 3,68 / 4,80. La conclusion « écart-type du gold ≥ 3,4 » découle de cette famille et n'a pas été mesurée. Un gold à loss élevée mais étalée (écart-type ≈ 1) n'existe pas dans la grille, sauf faiblement via γ < 1.
- Les erreurs du gabarit viennent d'orig : 21 % d'entre elles tombent sur le second choix d'orig. Le score de prédiction des fichiers dérivés d'orig est donc surestimé, ce qui oblige le calage à abaisser le score de loss pour retrouver les petits scores réels. Cela pousse vers un gold à loss très haute, donc vers la thèse.
- E1 sous-estime la rétention, ce qui pousse dans l'autre sens. Le bilan net n'est pas connu.
- `bg = 5` donne au gold 2,3 % de masse sur les 9 classes pour les images des autres classes : il « prédit » des classes qu'il n'aurait jamais vues. `bg ≥ 0` n'est pas cohérent avec l'hypothèse qu'il code.
**Le calage ne prouvait pas la thèse avant l'envoi.** Sur la grille de `fit.py`, 213 hypothèses sans `bg` collaient à 1,5 près. Le « aucune sans bg » vient de la grille réduite d'explore9.

## 3. Contrôles indépendants (numpy, données locales)
- **3a. Structure d'orig, 9 classes contre 36 [MESURÉ].** Marge médiane top1 − top2 : 12,06 contre 12,20 logits. Logit max : 23,3 contre 23,1. Corrélation effectif / marge par classe : 0,04. Précision sous flou fort : 0,867 contre 0,839 ; sous couleur forte : 0,511 contre 0,543. Aucune signature : orig traite les 9 classes comme les autres. C'est attendu sous toutes les lectures (orig a tout vu), donc ce contrôle est neutre.
- **3b. Le simulateur, revu avec les 6 scores réels [MESURÉ, `explore9.py 1.5`].** 5 hypothèses sur 5 400 passent à 1,5 près, toutes avec `bg = −5` ou `−10`, et elles prédisent pour `Z9_b0` **0,16 à 0,32** (médiane 0,20). Observé : 0. Aucune hypothèse de la grille ne reproduit donc les 7 scores. Soit la porte a tranché, soit la famille de gabarits est mal posée. Il ne reste que 5 hypothèses sur 5 400, systématiquement fausses sur `NEGce` (0,015 prédit, 0,022 réel) : c'est le signe d'un modèle mal posé, pas d'une identification.
- **3c. Une loi simple sur les 6 scores [MESURÉ, à vérifier].** On suppose prédiction × rétention constante et on prend le score de loss des seules images exactes, avec orig à 0,000 / 0,002. Toutes les lois de ce type gardent une erreur d'au moins un facteur 1,35. Les meilleures fixent **l'écart-type du gold s_g ≈ 1,1 à 1,4** et laissent la moyenne m_g libre (de 1,3 à 6,5, car tous nos m sont en dessous). Avec s_g ≥ 1,5, `NEGce` (s = 1,49) devrait rendre 1,5 fois plus par unité de loss qu'observé, et `PG_T70` 1,4 fois plus. Il faut donc soit s_g ≈ 1,2, soit un facteur prédiction × rétention propre à `NEGce` deux fois plus bas : sa dérive sur `mixes` est la plus forte, KL 0,205 et accord 0,72. Les deux lectures sont ouvertes et un envoi les sépare (§4, expérience 1). Corrélations : score–m 0,92 ; score–(m+s) 0,78 ; score–m/(1+KL_mixes) 0,96.
- **3d. Le groupe des copies modifiées pèse lourd.** Orig sur `held_color_strong` : m 2,98, s 4,27. Si les copies cachées sont fortes, d_orig y est petit et le score de loss de ce groupe devient très sensible. La loi de 3c ne change pas en y ajoutant `held_blur_strong`.

## 4. Expériences suivantes, classées par information attendue
1. **Envoyer `NEGce_all_hi_a4` (prêt ; m 0,37, s 2,13).** Il départage s_g ≈ 1,2 de s_g ≥ 3 : c'est ce qui décide entre une loss étalée et une queue d'erreurs.
   - Sous s_g ≈ 1,2 : score ≈ 0,5 × celui de `PG_T70`, soit 0,015 à 0,02, et toutes les méthodes visent une loss étalée sur toutes les images avec un écart-type d'environ 1,2.
   - Sous s_g ≥ 3 : 0,05 ou plus, la queue d'erreurs confiantes paie, et on pousse les fichiers à forte loss comme `NEGce_all_top_a8`.
2. **Mesurer la porte avec `PG_T70` + 9 classes absentes du forget set mises à zéro (tête).** Prendre les 9 de plus faible masse dans `classes_absentes.json`. L'effet sur l'oubli est quasi nul, la perte en validation est la même que pour `Z9` si la validation est équilibrée.
   - Environ 0,027 à 0,032 : la porte tolère −9 %, donc `Z9` = 0 vient de l'oubli et la thèse est réfutée pour de bon.
   - 0 : la porte est serrée (précision d'orig en validation probablement < 0,78). `Z9` ne disait rien, et toutes les méthodes ont une marge de précision très étroite (limite à fixer par dichotomie sur k).
3. **Lire `TS_tau4` (déjà envoyé) avec `y1180` et `temp_2.0`.** Ces fichiers sont adoucis sans aucun changement de classe : ils isolent prédiction × rétention, ce qui calibre le c de la loi 3c.
   - Score ≈ 0,075 × m (≈ 0,03) : ni le ciblage ni les changements de classe ne comptent, et la famille τ devient la voie la plus courte.
   - Score nettement plus bas : la rétention mange l'adoucissement global, et il faut le garder sélectif comme `PG`.
4. **Local, sans envoi : recaler le simulateur.**
   - Remplacer E1 par la réserve de `t3-methodes` comme jeu de contrôle.
   - Ajouter un gabarit où moyenne et écart-type sont découplés : par exemple p_gold = mélange de la sortie d'orig adoucie par une température propre à chaque image, et d'une erreur vers la classe voisine.
   - Imposer `Z9` = 0 et `S0_tau3` = 0,0010 comme contraintes.
   - Critère : une famille reproduit les 8 points à 1,3 près ; sinon le simulateur ne sert plus à choisir les envois.
5. **Seulement si l'expérience 2 donne 0 : `PG_T70` + `Z2`** (classes 0 et 7 à zéro, 260 images, 7,4 % du forget set, environ 2 % de la validation).
   - Score en hausse : la thèse revit, sous une porte serrée.
   - Score en baisse : la thèse est réfutée sans ambiguïté.

## Résumé : ce qui change une décision
1. `Z9_b0` = 0 alors que les 5 seules hypothèses compatibles du simulateur prédisaient 0,16 à 0,32 : la thèse « 9 classes vidées » ne tient plus que si la porte des 70 % a tranché.
2. Ne pas envoyer `Z9_bm5` ni `Z9_b0_tau1.5` : même risque de porte, et pas d'information de plus.
3. Le simulateur est mal posé (aucune hypothèse ne reproduit les 7 scores) : ne plus l'utiliser pour prédire un envoi avant de l'avoir recalé (expérience 4).
4. Son jeu de contrôle (copies des images à oublier) sous-estime la rétention : 0,69 simulé contre 0,90 mesuré pour `PG_T85`.
5. « Écart-type du gold ≥ 3,4 » est un produit du gabarit, pas une mesure. Une loi simple sur les 6 scores donne plutôt 1,1 à 1,4.
6. Envoi le plus informatif maintenant : `NEGce_all_hi_a4` (s = 2,13). Il tranche entre une loss étalée et une queue d'erreurs confiantes.
7. Puis le test de porte (`PG_T70` + 9 classes absentes à zéro). Il borne la précision qu'on peut sacrifier, pour toutes les méthodes.
8. Les lectures « plafond de tirage » et « 10 % d'une classe de 1 300 » expliquent les effectifs aussi bien que la thèse, et le 129 l'affaiblit.
9. Bug l. 126 (`q9` avec `mixes`) et incohérence `score()` / `Fast` à corriger avant toute réutilisation.
10. Le score suit m (corrélation 0,92) mieux que m + s (0,78) : continuer à monter m tant que s reste près de 1,2, en attendant l'expérience 1.
