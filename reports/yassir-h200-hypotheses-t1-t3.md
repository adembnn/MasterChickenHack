# Une H200 pour T1 et T3 · expériences prioritaires

21 novembre 2026. Au relevé de 22 h 04 à Paris, il manque à MasterChickenHack 0,01265 en T1 (0,73843 contre 0,75108) et 0,09806 en T3 (0,21793 contre 0,31599). Le point T3 de référence est depuis confirmé par le reçu 1716. Le public porte sur une partie des données. [Classement](http://34.147.229.55/leaderboard_page) · [Reçu T3](https://github.com/adembnn/EAC/issues/5#issuecomment-7069226675)

## T1 · choisir entre affiner un contour et détecter une zone entièrement manquée

**Pilote exécuté.** Sur les cartes PixelSeal partagées (`pred_train_pixelseal.npz`, SHA-256 commençant par `12ad1413c5af`), j’ai sélectionné pour chaque image les 2 048 cellules où le logit est le plus proche de zéro. J’ai entraîné sur 140 images et évalué sur les 30 de la réserve commune. Métrique officielle par image, seuil zéro, aucun réglage sur la réserve durant ce pilote. Cette carte a déjà servi aux diagnostics d’équipe ; une réserve réutilisée donne une mesure de développement. [Comparaison précédente](yassir-revue-trois-tracks.md)

| Sur les mêmes 30 images | mIoU PixelSeal | Écart à la base |
|---|---:|---:|
| Carte partagée | 0,63625 | — |
| Régression logistique sur signaux de luminance, texture et logits voisins, CPU | 0,63362 | −0,00263 |
| Petit réseau résiduel sur voisinage de logits 5×5 et luminance native 4×4, six époques, H200-6 | 0,63375 | −0,00249 |
| Correction parfaite de 2 048 cellules au hasard, diagnostic avec vérité | 0,66419 | +0,02794 |
| Correction parfaite des 2 048 cellules les plus incertaines, diagnostic avec vérité | 0,74862 | +0,11237 |

IC95 apparié du pilote H200 [−0,00628 ; +0,00173]. Ce résultat écarte le petit correcteur testé. L’oracle mesure la marge accessible *si toutes les étiquettes sélectionnées étaient connues*. Il ne constitue pas une prédiction de gain. PointRend explique comment échantillonner des caractéristiques visuelles fines aux positions ambiguës ; notre pilote utilise des pixels et des logits, sans encodeur fin appris. [PointRend, arXiv](https://arxiv.org/abs/1912.08193) · [Échantillonnage officiel](https://github.com/facebookresearch/detectron2/blob/main/projects/PointRend/point_rend/point_features.py)

**Premier entraînement T1 proposé.** Partir du réseau pleine résolution d’Adem. Sur les images dites pleines, le modèle manque entièrement 151 trous propres sur 272 pour PixelSeal et 170 sur 279 pour TrustMark. Produire des paires à contenu, message et filigrane identiques : une version sans trou, une version avec une petite région propre recopiée depuis la même photo avant marquage. Tirer tailles et formes sur les plis d’apprentissage. Comparer deux entraînements à nombre d’images, de pas et de graines identiques : recette actuelle, puis même recette avec ces paires. Vérifier sur les vraies images le rappel des trous par taille et le nombre de faux trous par image, puis la mIoU sur les trois méthodes. WAM fournit un précédent pour les masques localisés ; les trois encodeurs du kit produisent les exemples du concours. [Diagnostic d’Adem](https://github.com/adembnn/EAC/issues/3#issuecomment-7069208868) · [WAM, arXiv](https://arxiv.org/abs/2411.07231) · [Code des auteurs](https://github.com/facebookresearch/watermark-anything)

**Deuxième essai T1 seulement si les trous sont déjà touchés.** StegaStamp présente ce profil selon le diagnostic d’Adem. Geler la carte de base et entraîner une tête de type PointRend avec les premières activations spatiales du réseau pleine résolution. Comparer à la même tête avec points uniformes, à budget et entrée identiques. Vérifier séparément l’IoU des trous touchés et les effets sur les images partielles. Pour envisager un envoi, l’équipe exige une hausse de 0,02 sur la moyenne de réserve des trois filigranes, un IC95 apparié excluant zéro et une vérification indépendante. [Décision équipe](https://github.com/adembnn/EAC/blob/a5692ea/CLAUDE.md)

## T3 · améliorer la distribution de sortie à sélection d’images identique

**Premier essai T3.** Reproduire `PG_T85_f35_l814c` et conserver les cibles, les 35 % sélectionnés, les augmentations, l’ancre et les lots. Modifier uniquement la loss à T=2. Pour chaque image, ajouter à la KL complète un terme qui compare les distributions des 99 autres classes, après exclusion de la pseudo-classe d’origine et renormalisation. Deux bras fixés avant lecture des résultats : coefficient additionnel 0 et 0,25. Dans les deux cas, conserver intégralement la perte à T=1. C’est un test de l’apprentissage d’une même cible. Adem teste déjà la construction de la cible, et Titouan le choix des images. [Simulation de l’équipe](https://github.com/adembnn/EAC/issues/5#issuecomment-7069180604) · [DKD, arXiv](https://arxiv.org/abs/2203.08679) · [Implémentation des auteurs](https://github.com/megvii-research/mdistiller/blob/master/mdistiller/distillers/DKD.py)

Le module `t3_conditional_kl.py` est prêt à brancher sur la recette exacte. Il attend les logits cibles à T=1. Si la recette stocke des probabilités, transmettre leur logarithme après stabilisation. Sur deux lots synthétiques de contrôle à 100 classes (mêmes données, T=1 et T=2), coefficient nul : la valeur reproduit la KL classique à la précision affichée et l’écart maximal des gradients est inférieur à 2×10⁻⁹. Ce contrôle porte sur le calcul de la loss uniquement. [Module](../tracks/t3-unlearning/experiments/t3_conditional_kl.py)

**Décision avant soumission.** Avec des golds issus du simulateur d’équipe, comparer séparément la KL de prédiction à T=2, les moments de loss et la rétention. Garder un monde simulé hors choix des paramètres ; rejouer avec une deuxième graine. Sur le vrai ViT, comparer l’oubli sur exactes et copies, puis l’accord et la KL de rétention sur des images indépendantes de l’ancre. Une hausse sur simulateur avec recul de la rétention réelle estimée demande un nouvel examen. Les golds officiels et la porte d’exactitude de 70 % sont cachés. Le créneau #5 revient au meilleur candidat validé conjointement.

**Si la rétention baisse.** Mesurer après échauffement le cosinus des gradients d’oubli et d’ancre sur 64 lots. Si leurs directions s’opposent régulièrement, comparer la mise à jour courante à une projection unilatérale du gradient d’oubli sur le plan orthogonal au gradient d’ancre, seulement lorsqu’ils s’opposent. Garder la même cible et la même pondération. PCGrad fournit la méthode de projection ; son application à l’oubli d’images est ici une hypothèse. Chronométrer le surcoût d’un second calcul de gradient. [PCGrad, arXiv](https://arxiv.org/abs/2001.06782) · [Code des auteurs](https://github.com/tianheyu927/PCGrad)

## Reproduire le pilote T1

Dans un environnement avec le kit officiel `wmloc`, PyTorch, Pillow, NumPy, SciPy et scikit-learn. Le dernier appel utilise CUDA. Les entrées dérivées restent hors Git.

```bash
cd tracks/t1-watermark/experiments
python pilot_uncertain_cells.py /data/train.zip /data/pred_train_pixelseal.npz
python pilot_point_head.py prepare /data/train.zip /data/pred_train_pixelseal.npz /tmp/point-inputs.npz
python pilot_point_head.py fit /tmp/point-inputs.npz
```

Le pilote sur `sapphire-perch` a utilisé 146,2 MiB alloués par PyTorch au pic. Cette exécution vérifie l’accès CUDA au nœud 6. Son affectation exclusive et la disponibilité de la recette et du checkpoint 1716 sont des questions distinctes.
