# Contrôle T3 du 22 novembre 2026

Au relevé de 03 h 04 à Paris, MasterChickenHack est premier en T1 avec **0,80327** (candidat 35, reçu **2550**) et troisième en T3 avec **0,39422** (SOUPs1, reçu **2531**). Le premier en T3 est à **0,43606**. [Classement officiel](http://34.147.229.55/leaderboard_page) · [Reçu T1 confirmé par l’équipe](https://github.com/adembnn/EAC/issues/3#issuecomment-7072363911) · [Référence du reçu T3](https://github.com/adembnn/EAC/issues/5#issuecomment-7072505163)

## Projection des mises à jour de SOUPs1 sur H200-6

Le premier lancement cherchait SOUPs1 dans `candidates/`. Le fichier de référence se trouve dans `~/runs/pg/inbox/SOUPs1.pt`. La relance a comparé huit couples rang et coefficient, puis exporté les trois candidats ci-dessous. Les ratios sont calculés relativement à SOUPs1 dans le test de présélection. Un ratio d’oubli proche de 1 conserve le signal de référence. Le ratio de dérive se mesure sur des mélanges synthétiques et baisse quand le candidat se rapproche du modèle d’origine.

| Rang | Coefficient | Ratio d’oubli | Ratio de dérive | SHA-256 du fichier sur H200-6 |
| ---: | ---: | ---: | ---: | --- |
| 8 | 0,50 | 0,9610 | 0,9018 | `cca14293dad168d992f201df35c1b0efa866d469f5a529225137ac6f553967ad` |
| 32 | 0,25 | 0,9603 | 0,9141 | `2d7451bfcf44f8c628223ab76ace094a4365ea04c4c6d6cf30e4a52f74384dde` |
| 16 | 0,25 | 0,9708 | 0,9318 | `4d63e734c1857b34d63ce2dbc32c4481d5e4a4b979dc3434c3421c45a835e6f6` |

Résultats et checkpoints dans `~/runs/t3_nullspace`. Les empreintes distinguent cette relance des fichiers générés sur H200-4. [Essai H200-4](https://github.com/adembnn/EAC/issues/5#issuecomment-7072587547)

L’équipe a déjà entraîné huit partitions OOF et déposé quatre candidats. Les évaluations locales des trois variantes principales ressemblent à celles de la base. La variante qui modifie seulement le rang déplace 2,8 % des images du groupe des 70 %. La variante qui modifie la classe cible change 14 % des bascules. [Résultats et chemins des fichiers OOF](https://github.com/adembnn/EAC/issues/5#issuecomment-7072643584)

## Comparaison des modèles complets

| Modèle | CE oubli exacte moyenne / écart-type | CE oubli augmentée moyenne / écart-type | Accord / KL sur mélanges | Accord / KL sur CIFAR |
| --- | --- | --- | --- | --- |
| SOUPs1 | 4,869 / 4,553 | 4,238 / 4,173 | 0,703 / 0,295 | 0,636 / 0,504 |
| Rang 8, coefficient 0,50 | 4,852 / 4,544 | 4,220 / 4,164 | 0,707 / 0,294 | 0,636 / 0,504 |
| Rang 8, normes des lignes restaurées | 4,852 / 4,543 | 4,222 / 4,164 | 0,706 / 0,294 | 0,637 / 0,504 |

Le gain de 9,8 % au test de présélection correspond à la norme d’une perturbation des logits de la tête, calculée sur les features du modèle d’origine. Le passage sur le ViT complet améliore l’accord sur mélanges de 0,4 point et laisse CIFAR au même niveau à trois décimales. Ce résultat reste sous le seuil d’envoi. Journal `~/runs/t3_nullspace/dash_n6_measured.log`. Durée du comparatif avec référence d’origine incluse de 82,9 s sur le GPU partagé. Pic PyTorch du processus de 1,17 Gio alloués et 1,26 Gio réservés.

Le troisième modèle restaure la norme L2 de chaque ligne de la tête à sa valeur dans SOUPs1. Contrôle exécuté en 57,3 s, référence d’origine incluse. Gain CIFAR de 0,1 point d’accord, KL inchangée à trois décimales. La restauration de norme reste sous le seuil d’envoi. [Retour d’expérience sur Reddit](https://www.reddit.com/r/LocalLLaMA/comments/1oypwa7/a_more_surgical_approach_to_abliteration/) · [Implémentation de la préservation des normes](https://github.com/p-e-w/heretic/blob/master/src/heretic/modifiers/abliteration.py)

Le handoff décrit huit projections sur les blocs tardifs, avec un gain CIFAR d’au plus 0,004. Les directions d’oubli et de rétention y avaient un cosinus de 0,85 à 0,94. Les sweeps `gold_blend` et la manipulation de marge figurent déjà dans le chat Cursor « Euro AI Championship 2026 ». La file d’Adem couvre aussi la fraction d’oubli 68–72 %, le réentraînement de six blocs et les cibles à huit partitions. [Essais précédents](https://github.com/yassirboudda/eac-handoff/blob/main/T3-ABLATE.md) · [File T3](https://github.com/adembnn/EAC/issues/5#issuecomment-7072706411)
