# T3 · test d’ajustement de la classe d’origine sur H200-2

SOUPs1, reçu #2531 et meilleur T3 d’équipe à 0,39422, sert de témoin apparié. Cette expérience isole la probabilité de la classe prédite par le modèle d’origine sur chaque image à oublier. La cible ne modifie que le logit de cette classe pour les images qui avaient déjà changé de classe sous SOUPs1. Les 99 autres logits gardent exactement leurs valeurs dans la cible. Le modèle apprend l’ajustement par la tête et les projections de sortie de l’attention et du MLP du dernier bloc. Pénalité KL sur les mélanges vers le modèle d’origine.

| Réglage | CE exacte | CE JPEG | KL cible exacte | KL cible JPEG | Rétention proxy mélanges | Débit d’entraînement | VRAM réservée |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SOUPs1 | 4,7855 | 4,7915 | 0,53127 | 0,56665 | 0,60435 | | |
| CE cible ×1,04 exacte et ×1,05 JPEG, étape 160 | 4,8308 | 4,8376 | 0,54265 | 0,57866 | 0,60025 | 4 245 images/s | 2 054 MiB |
| Marginale cible à température 2, étape 160 | 4,8801 | 4,8873 | 0,54711 | 0,58311 | 0,59655 | 2 528 images/s | 2 054 MiB |

Évaluation appariée sur 384 images et leurs copies JPEG. Cible KL extraite du cache `RDt24r_70_l1016`. Les deux variantes ratent le filtre combiné. Aucun checkpoint exporté. Mild et CIFAR étaient réservés à la validation d’un candidat ayant passé le filtre. Ils n’ont donc pas servi à sélectionner un checkpoint.

Un calcul analytique, réservé au diagnostic, remplace directement et séparément le logit de la classe d’origine pour chaque image pour rapprocher sa probabilité à température 2 de la cible, avec un déplacement limité à 0,6. Ce calcul donne un KL cible de **0,52443** sur exactes, **0,55906** sur JPEG et **0,60395** sur mild (témoin mild **0,61425**). Les moyennes CE résultantes sont 5,152, 5,143 et 4,429. Les deux essais apprennent une correction moyenne qui ne reproduit pas le gain individuel de ce diagnostic.

Le [script de recalibrage](../tracks/t3-unlearning/experiments/t3_margin_recalibrate.py) réutilise les métriques du [comparatif ridge](../tracks/t3-unlearning/experiments/t3_ridge_repair.py). Métriques sur H200-2 dans `~/runs/t3_margin_swarm/result_v1/metrics.json` et `result_v2/metrics.json`. Temps total 9,94 s et 14,82 s. [Classement officiel](http://34.147.229.55/leaderboard_page) · [File et score d’équipe](https://github.com/adembnn/EAC/issues/5#issuecomment-7072852391)
