# T3 · Fusion des sorties de checkpoints et distillation

SOUPs1 (reçu 2531, 0,39422 officiel) sert de témoin. Les autres membres sont `PG_RDt24r_70_l1016_fix`, `PG_RD6m2t24_70_l1117` et `YSOUP4_RDt24r_70`. La fusion porte sur les **prédictions des fichiers entraînés**. Elle moyenne leur queue conditionnelle à température 2, puis reconstruit les logits pour garder la CE de SOUPs1 image par image. Ce calcul ne préserve pas automatiquement la probabilité de la classe d’origine à température 2. [Référence SOUPs1](https://github.com/adembnn/EAC/issues/5#issuecomment-7072505163)

## Tri CPU

192 images et quatre vues, moitié calibration et moitié test. Choix du poids de fusion sur exactes et JPEG. `mild` et `rot15` restent réservées. La fusion uniforme complète est sélectionnée. Sur les 96 images de test, la diminution de KL selon trois cibles de substitution va de **0,117 à 0,206** sur `mild` et de **0,125 à 0,181** sur `rot15`. Ces trois cibles sont liées à la redistillation et le diagnostic mesure une combinaison de sorties, avant transfert dans un ViT unique.

CPU H200-6, quatre threads, 129,37 s pour 3 072 prédictions de vues, soit 23,75 prédictions/s. Journal `~/runs/pg/fusion/cpu_pilot/screen.json`. [Script](../tracks/t3-unlearning/experiments/t3_output_fusion.py)

## Distillation dans SOUPs1

Ajuster la tête et les projections d’attention et MLP des deux derniers blocs. Une réplique utilise le KL complet à température 2. Une seconde normalise sur les seules classes alternatives dans le terme de distillation et renforce la conservation de la CE par image. Les deux gardent une ancre sur les mélanges et utilisent les mêmes images exactes et JPEG. Les images `mild`, `rot15` et CIFAR servent au contrôle du finaliste. Les deux variantes furent évaluées par `dash.py` sur le cache intégral.

| Fichier | CE exacte, moyenne / écart-type | CE copies, moyenne / écart-type | KL vers cible exacte | Accord / KL mélanges | Accord / KL CIFAR |
| --- | --- | --- | ---: | --- | --- |
| SOUPs1 | 4,869 / 4,553 | 4,238 / 4,173 | 0,536 | 0,703 / 0,295 | 0,636 / 0,504 |
| Fusion, KL complet | 4,804 / 4,504 | 4,178 / 4,129 | 0,526 | 0,718 / 0,276 | 0,639 / 0,504 |
| Fusion, KL conditionnel | 4,843 / 4,529 | 4,220 / 4,150 | **0,518** | 0,718 / 0,281 | 0,636 / 0,515 |

Sur le lot réservé, la version conditionnelle réduit le KL vers la cible de **0,59585 à 0,58243** sur `mild` et de **0,63148 à 0,62030** sur `rot15`. Le compromis est la légère hausse du KL CIFAR du comparatif intégral. Le pic de mémoire PyTorch est de 9 164 Mio réservés pour les deux versions. La version conditionnelle prend 17,01 s avec une capture à 3 067 vues/s. La version complète prend 28,84 s, capture à 2 097 vues/s sur la carte partagée. [Script](../tracks/t3-unlearning/experiments/t3_fusion_distill.py)

Checkpoints et journaux dans `~/runs/pg/fusion` sur H200-6. SHA-256 version complète `bc17fe75ab975a43b9ef3dc2c055eb8976d30a0475cc3941a121665258fc7f32`. Version conditionnelle `7a09b47c534d1e92c53cbb8fb823f86b265f6e27f04e3c05e85f361f2fddb7d1`. Ces mesures locales restent à comparer au protocole indépendant de l’équipe.

Un troisième comparatif ouvre quatre blocs, utilise un taux d’apprentissage de 4 × 10⁻⁵ et augmente le poids de la fusion à 0,95. Il réduit davantage le KL sur le lot réservé après plusieurs époques, jusqu’à **0,4804** sur les exactes contre **0,5452**, mais la CE tombe de 4,910 à 4,715 et le proxy mélanges régresse de 0,5527 à 0,5339. Le seul finaliste compatible avec les contrôles, à une époque, réduit le KL de **0,0059** sur les exactes et de **0,0040** sur mild. La version à deux blocs reste la meilleure des variantes acceptées. Comparatif quatre blocs en 52,58 s, 2 026 vues/s et 12 612 Mio réservés.

Un [contrôle séparé](yassir-t3-controle-classes-alternatives.md) montre pourquoi une réduction du KL sur les seules classes alternatives ne suffit pas. À CE fixe, le lissage uniforme simple augmente la KL de la distribution complète.
