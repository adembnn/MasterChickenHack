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

## Deux variantes avec les derniers blocs entraînables

`t3_suffix.py` garde les dix premiers blocs figés, met leurs activations
en cache sur H200 et entraîne les deux derniers blocs, la normalisation
finale et la tête. Les 700 images réservées suivent le même split que
l'essai de la tête. Les mélanges d'images ancrent le comportement au
professeur, dont les poids restent figés.

| Candidat sur la réserve | KL exact | KL recadrage léger | KL rétention | Accord top-1 rétention |
|---|---:|---:|---:|---:|
| Original | 0,05839 | 0,04964 | 0 | 100 % |
| Tête seule | 0,04690 | 0,03768 | 0,02498 | 97,43 % |
| Deux blocs, ancrage 2,0 | 0,05411 | 0,04524 | 0,00101 | 99,14 % |
| **Deux blocs, ancrage 0,45 (v2)** | **0,04738** | **0,03822** | **0,00908** | **98,14 %** |

La v2 réduit le KL du proxy d'oubli de 18,9 % sur les images exactes et de
23,0 % sur les recadrages légers. Elle est le premier candidat de ce lot
pour une soumission officielle. Le checkpoint ne modifie que 32 tenseurs
du suffixe, se recharge en mode strict et produit des logits finis.

Depuis le dossier des données, avec le même `PYTHONPATH` que pour la tête :

```bash
python "$REPO/tracks/t3-unlearning/experiments/t3_suffix.py" \
  --lr 4e-5 --epochs 8 --anchor-weight 0.45 --out candidate_suffix_v2_t3.pt
```

## Dépôt officiel

- H200, comptes `adem` et `titouan` :
  `$HOME/eac-h200/t3-codex/candidate_suffix_v2_t3.pt`.
- H200, compte `hermes` :
  `/home/hermes/eac/t3-codex/candidate_suffix_v2_t3.pt`.
- Taille : 343 631 070 octets. SHA-256 :
  `05cd0dde2c6bc1e3f1dc41c5098780b853ebe935420003f900488d25afb5f082`.

Déposer le fichier via le client officiel
`reports/sources/t3/submission.py` et la configuration d'équipe existante.
Le seul endpoint concerné est `/submit/31-unlearning`. Le fichier a été
copié et vérifié sur les trois comptes H200. Les agents peuvent l'envoyer
depuis une session qui a déjà accès à la clé du concours. Elle reste dans
la configuration privée. Reporter le score exact retourné par le serveur
dans l'[issue T3](https://github.com/adembnn/EAC/issues/5) avant de choisir
la prochaine expérience.
