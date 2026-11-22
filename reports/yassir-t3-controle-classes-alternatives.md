# T3 · contrôle de l’allocation des classes alternatives

La file d’Adem compare déjà l’effet de la masse sur la classe d’origine et plusieurs mélanges de pseudo-modèles. Nous avons isolé l’allocation des 99 autres classes dans la cible de SOUPs1, puis contrôlé la distribution complète. Calculs CPU sur H200-2 avec CUDA désactivé. [Pistes de l’équipe](https://github.com/adembnn/EAC/issues/5#issuecomment-7072865457) · [Diagnostic du vérificateur](https://github.com/adembnn/EAC/issues/5#issuecomment-7072873552)

## Allocation conditionnelle

Les deux références sont les cibles de redistillation `RD6m2t24` et `RDm3t24`. Les variantes sont choisies sur 1 452 images basculées par la base, puis mesurées sur 998 autres. Les voisins DINO sont calculés hors pli. Le diagnostic inclut DINO comme comparaison externe. Toute option entraînable retenue devrait reposer sur le modèle et les données autorisés.

| Variante | Δ KL sur référence 6 blocs | Δ KL sur référence multi-partitions |
| --- | ---: | ---: |
| 10 % de distribution S1 | −0,00552 | −0,00498 |
| 10 % de distribution d’orig sur quatre variantes d’image | −0,00698 | −0,00629 |
| 10 % de distribution globale du modèle DINO | +0,01398 | +0,01837 |
| 10 % uniforme sur les 99 classes | **−0,01311** | **−0,00931** |
| 10 % de distribution multivue, second moment conservé | −0,00262 | −0,00236 |

Aucun mélange avec un voisinage DINO n’a gagné sur les 1 452 images de calibration. Une part importante du gain conditionnel vient du lissage de la distribution, puisque le mélange uniforme bat les distributions proposées.

## Distribution complète à température 2

Le test reconstruit les logits sur les seules images basculées et préserve exactement la CE de la classe d’origine à température 1. Sur les 1 400 images réservées, uniformiser à 10 % **augmente** le KL complet de **0,01847** et **0,02119**. Intervalles bootstrap appariés à 95 % respectifs **[0,01705, 0,01993]** et **[0,01996, 0,02247]**. Sur les 2 100 images de calibration, la grille 0 à 25 % sélectionne **0 %**.

Les références restent deux approximations liées, et la mesure ne donne pas le score officiel. Ce contrôle invalide le gain apparent du mélange uniforme quand la classe d’origine compte dans le KL. Aucun entraînement GPU ni cible exportée.

Journaux sur H200-2 dans `~/runs/t3_semantic_cpu/`.
