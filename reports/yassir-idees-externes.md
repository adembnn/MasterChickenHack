# Ce que les challenges précédents font mieux que nous (agent de Yassir)

Veille faite le 21 novembre à 17h20 sur les dépôts publics des challenges qui ressemblent au nôtre.
Objectif : voler les idées, pas les scores. Ce qui suit est vérifié en lisant les dépôts, et je
distingue clairement ce qui est mesuré de ce qui est déclaré par les auteurs.

## T3, désapprentissage : le NeurIPS 2023 a résolu le même problème

Le challenge NeurIPS 2023 Machine Unlearning notait exactement comme nous : comparer un modèle
désappris à des modèles réentraînés sans les données à oublier, sur la distribution des pertes et sur
les prédictions, plus un terme de rétention. Notre T3 est la version 2026 de la même tâche.

Trois dépôts utiles :

- `unlearning-challenge/starting-kit` : le kit officiel, avec le carnet CIFAR-10 et la définition de
  la métrique. À lire pour la structure, pas pour le code.
- `AlbinSou/unlearning-challenge-metric` : reproduction de la métrique. L'auteur dit lui-même que
  certains détails restent cachés côté organisateurs, donc les scores ne sont pas comparables au
  classement. Mais la structure est la bonne : distribution des pertes du modèle désappris contre
  distribution des pertes de modèles réentraînés, plus rétention. **C'est reproductible chez nous, et
  c'est ce qui nous manque le plus : un juge local.**
- `spicy-shawarma/Machine-Unlearning` : soumission au challenge. La méthode est une régression douce,
  pas une destruction : on inverse l'influence des exemples à oublier en réentraînant sur le jeu de
  rétention, avec un fin réglage qui conserve les performances. Chiffres déclarés par l'auteur, non
  vérifiés : accuracy rétention 99,5 % vers 99,0 %, test 88,2 % vers 83,6 %, et attaque d'appartenance
  à 0,50 sur le modèle réglé, soit le hasard.

**La leçon.** Le gagnant type ne détruit pas la confiance, il rapproche le modèle de ce qu'il serait
sans les données, en gardant de bonnes performances sur le reste. C'est exactement l'inverse de mon
premier essai, qui a fait 0,00 sur le classement : j'ai bougé le modèle de 95 % sur les images à
oublier, et la porte d'utilité a dû tomber.

**Ce qu'on peut faire tout de suite avec ça.** Si le classifieur est un CIFAR-100 agrandi, alors on a
50 000 images de rétention gratuites, et la méthode devient faisable en une heure sur la H200 :
distillation sur les images à oublier vers une cible « jamais vue », et entropie croisée normale sur
les images de rétention. C'est le schéma des gagnants, appliqué à nos données. Je vérifie
l'hypothèse CIFAR-100 en ce moment en mesurant l'accuracy du modèle fourni sur le test CIFAR-100.

## T1, localisation : l'architecture de Watermark Anything

`facebookresearch/watermark-anything`, le dépôt de référence du filigrane localisé (1 133 étoiles).
Leur décodeur ne fait pas un score global puis un seuil. Il sort deux têtes en même temps :

- une carte de détection par pixel, 1 canal en 256 × 256, passée au sigmoïde, c'est le masque ;
- les bits du message par pixel, agrégés en pondérant chaque pixel par sa probabilité de masque
  (`msg_predict_inference(bit_preds, mask_preds)`).

Autrement dit, la détection et le décodage sont appris ensemble, et le message se lit en pondérant par
la carte. Chez nous, l'extracteur officiel est figé et ne donne qu'un accord global par image, donc
on reconstitue la carte par fenêtres. L'idée à voler : entraîner une petite tête de segmentation sur
les caractéristiques de l'extracteur, avec une perte de segmentation, au lieu d'apprendre à imiter une
carte bruitée. C'est ce que mon U-Net a raté faute de signal d'entrée, et c'est corrigeable en
remplaçant l'entrée carte par les cartes par position de l'extracteur, qui sont calculées à la bonne
échelle.

## T2, égalité de modèles : deux dépôts pour Titouan

Je ne touche pas à T2, mais la veille est utile à celui qui la tient.

- `irenasaracay/model-equality-testing` (24 étoiles) : tests statistiques entre une API de type
  boîte noire et une distribution de référence.
- `sunblaze-ucb/LLM-api-audit` : audit d'API de modèles avec des tests de distance sur les sorties.

Les deux confirment la structure de notre approche : beaucoup de réponses courtes ne suffisent pas,
il faut des réponses longues et une statistique de distance stable sous quantification et sous LoRA.

## Ce que je n'ai pas pu vérifier

- Les writeups Kaggle des trois premiers du NeurIPS 2023 renvoient une page vide au téléchargement
  (contenu chargé en JavaScript). Je cite donc le dépôt de soumission, pas les writeups.
- Les chiffres de `spicy-shawarma` sont déclarés par l'auteur, sans reproduction de ma part.
- Rien ici n'a été exécuté sur nos données, sauf mention contraire.
