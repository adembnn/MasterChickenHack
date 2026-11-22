# EAC T3 — nouveau levier : moyenne des postérieurs à T=2 avec CE exactement conservée

**22/11/2026, suite de la recherche TOP1. Statut : CODE COMMITÉ, preuve algébrique et smoke CPU synthétique OK ; aucun nouveau poids ViT généré, aucune mesure H200 ou score EAC dans cette session.**

## Baseline et erreur à éviter
- Dernier meilleur public vérifié sur #5 : **0,3942**, candidat **SOUPs1**, soumission #2531 (02h39), pas 0,777 : le 0,777 du dépôt `eac-handoff` est un proxy m/s local d'un fichier qui a fait **0,327 public**.
- Source actuelle de redistill : moyenne de trois partitions différentes (Spearman pair à pair ≈0,89 ; top-70 concorde 91–92 %) ; fausse classe stable seulement à 75,5–77,3 % d'une partition à l'autre. Huit partitions sont **DÉJÀ** en cours d'entraînement chez Adem, avec moyenne de classe et de rang prévue. Ne rien dupliquer.
- La loss hors pli des professeurs de redistill est ≈1,0–1,7, très inférieure à celle du gold inféré (5,0–5,2) : ne pas recopier leur CE. Le meilleur PG dispose déjà d'une cible de CE par image calibrée à ~4,85/4,57 en sortie.

## Une innovation orthogonale, au niveau EXACT de la métrique
Le serveur évalue `KL(p_gold(T=2) || p_sub(T=2))`, où `p_gold` est la **moyenne ARITHMÉTIQUE** des sorties T=2 de plusieurs golds. La moyenne des logits, ou la moyenne des p(T=1) suivie d'un unique changement de température, **n'est pas** cette moyenne.

Pour chaque image i, K professeurs hors pli déjà présents :
1. `p2_k = softmax(logits_oof_k / 2)`, puis `g2 = mean_k(p2_k)` (pas mean(logits), pas mean(p1) avant T2).
2. Exclure la classe `yhat_i` d'orig, et normaliser les 99 autres classes : `r_gold = normalize(g2[non-yhat])`.
3. Lire la cible PG existante `q1_i` et calculer sa queue conditionnelle T2 `r_base = normalize(sqrt(q1_i)[non-yhat])`.
4. Un unique changement : `r_beta = (1−beta)*r_base + beta*r_gold`, avec beta=.2/.4/.6.
5. Inversion **exacte** vers les cibles T1 compatibles avec la T2 : `q1_new[yhat] = q1_old[yhat]`; `q1_new[wrong] = (1−q1_old[yhat]) * normalize(r_beta**2)`; puis `q2_new = normalize(sqrt(q1_new))`.

Ainsi la **CE cible individuelle** `−log q1[yhat]` ne change pas, donc ses moyenne/écart-type et les indices sélectionnés restent identiques. L'effet recherché est uniquement l'adoucissement/calibration de la distribution des classes fausses à **T=2**. Attention : les CE mesurées du ViT après entraînement peuvent changer, donc mesurer `dash.py` sur le nouveau poids.

## Validation SANS envoyer au serveur
Nouveau script : [`t3_t2_tail_calibrator.py`](../tracks/t3-unlearning/experiments/t3_t2_tail_calibrator.py), branche `research/top1-t1-t3-nov21` (PR #34).

1. **Pas de GPU pour le calibrateur** : exporter les distributions OOF [3500,100] de >=3 partitions indépendantes, en respectant EXACTEMENT l'ordre `forget_images.npz`. La source actuelle est `oof_rd.pt` (`probs`, `logits`, `yhat`) ; les sorties des 8 partitions doivent être exportées depuis `t3-redistill`. Le fichier réel `tgt_RDt24r_70_l1016...` étant sur la VM et non dans Git, **inspecter ses clés** `torch.load(...,weights_only=True).keys()` et ajuster `--base-key` au besoin.
2. `python tracks/t3-unlearning/experiments/t3_t2_tail_calibrator.py --base .../tgt_RDt24r_70_l1016.pt --yhat .../s1_prep.pt --teachers .../oof_seed0.pt .../oof_seed1.pt .../oof_seed2.pt --out ~/runs/t3_t2_tail --betas 0,.2,.4,.6`.
3. Sur chaque beta, **crossvalidation laisser-une-partition-dehors** : évaluer la KL entre la distribution T2 fausse d'un professeur OOF et la moyenne des autres, sur les images basculées (importance pondérée par la masse hors yhat du professeur). Exiger une réduction solide de KL conditionnelle T2 versus beta=0 **sur plusieurs partitions** ; sinon stop. Cette validation ne prouve pas un gain EAC (gold inconnu).
4. Si beta positif, brancher `q1` et `q2` dans le générateur/entraîneur PG existant **sans changer** fraction=70%, seed, `--lam 3`, `--steps 1500`, `--mstar`, ancre, vues, sélection ni recette de CE-hauteur. Un seul facteur par candidat. Faire `dash.py`, exactes+copies + vues fortes, rétention/proxy+porte >70%, et vérification indépendante ; accepter UNIQUEMENT si signe clair de gain.
5. Partager chemin, sha256, comparaison au **meilleur courant** sur #5. **Adem seul soumet**, selon CLAUDE.md décision 01:00. Zéro requête T2.

**Smoke test CPU synthétique effectué** : beta=0 reconstruit la cible à 2,94e−8 près ; la CE individuelle/moyenne/std est conservée numériquement (800 exemples synthétiques, 4 professeurs) ; trois options produisent des tenseurs normalisés et finis. **Les KL et beta préférés obtenus sur ces données artificielles ne sont PAS des résultats EAC.**

## Revue arXiv/GitHub : les alternatives moins prioritaires
- [MU-Mis (arXiv:2402.15109 v4, ICLR 2026)](https://arxiv.org/abs/2402.15109) : supprime la sensibilité aux images à oublier sans retain ; méthode et [code](https://github.com/poppopbean0903/MU-Mis) crédibles, mais peut **abaisser la loss sur les forget**, ce qui serait destructeur ici où le gold a des erreurs très confiantes. À tenter seulement en régularisation légère et à CE/gate constants.
- [DAMP (CVPRW 2026)](https://arxiv.org/abs/2604.15166) : projection par couche de directions classe-specific ; son [dépôt](https://github.com/armanhtm/DAMP) ne contient actuellement qu'un README (« Will be published soon! »). **Ce n'est pas une recette prête à l'emploi**, et son contexte class removal ≠ instance-level 3500.
- [Unmerge (arXiv:2609.38895)](https://arxiv.org/abs/2609.38895) : avertit explicitement que le sous-espace séparant des images aléatoires du retain peut être inexistant. Le prototype précédent `t3_retain_nullspace.py` basé sur les mixes de forget mérite donc une **priorité inférieure** à l'expérience T2-tail, et n'apporte pas une justification de gain.
- [Interference-Aware Multi-Task Unlearning](https://arxiv.org/abs/2605.19042) : gradient orthogonalization requiert un bon signal retain, indisponible chez nous ; éviter la chirurgie de gradients agressive sans garde.

**Conclusion** : si l'équipe a déjà moyenné les p2 EXACTS des huit professeurs, ne pas refaire cette expérience. Sinon, c'est un test orthogonal et direct sur le facteur S_pred du barème, plus défendable qu'une projection arbitraire des poids.
