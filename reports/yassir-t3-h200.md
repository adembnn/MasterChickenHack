# Candidat T3 sur H200

MasterChickenHack est premier en T1 à 0,634988 mIoU public au relevé de
17 h 44, le 21 novembre 2026. La soumission 1103 d'Adem reste la référence.
Le même relevé montre un meilleur score public T3 de 0,126194 et aucune
soumission T3 classée pour notre équipe.

## Candidat T3 disponible

L'expérience `t3_head.py` fige le ViT-B/16 original. Elle ajuste uniquement
sa tête linéaire pour rapprocher ses prédictions de la moyenne du professeur
sur trois vues recadrées. Elle utilise uniquement les 3 500 images fournies.
La sélection se fait sur 700 images réservées. Le candidat est ensuite
réajusté sur les 3 500 avec le coefficient sélectionné de 0,85.

| Mesure sur la réserve | Modèle original | Tête ajustée avant réajustement |
|---|---:|---:|
| KL vers le proxy d'oubli, image exacte | 0,05840 | 0,04690 |
| KL vers le proxy d'oubli, recadrage léger | 0,04965 | 0,03768 |
| Accord top-1 avec l'original, mélanges d'images | 100 % | 97,43 % |

Ces mesures utilisent des cibles locales. Seul le serveur connaît le modèle
réentraîné et la précision sur le contrôle caché qui déclenche le seuil de 70 %.
Le fichier respecte le format `state_dict` strict, float32, valeurs finies,
343 937 306 octets.

- Hermes peut lire `/home/hermes/eac/t3-codex/candidate_head_t3.pt` et
  `metrics_t3.json`.
- Depuis le compte H200 `adem` ou `titouan`, les mêmes fichiers sont dans
  `$HOME/eac-h200/t3-codex/`. SHA-256 du checkpoint :
  `530dee873e94c3fea10f5f1abde470bca25810185203ee399e9e06a4b281dc81`.
- Pour reproduire, lancer `tracks/t3-unlearning/experiments/t3_head.py` avec
  le Python du kit GPU, depuis le dossier contenant `model.pt` et
  `forget_images.npz`. Ajouter `reports/sources/t3` du dépôt à `PYTHONPATH`
  pour réutiliser le chargeur officiel.

La voie de soumission est le client officiel T3 avec la configuration
d'équipe déjà sur JURECA. Le candidat est prêt à transmettre à un membre
ayant cet accès. Garder la clé API dans la configuration d'équipe.
