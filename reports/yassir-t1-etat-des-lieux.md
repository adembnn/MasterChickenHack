# T1 · où on en est vraiment (agent de Yassir)

21 novembre, 21h UTC. Trois GPU utilisés en parallèle, un quatrième nœud monté, le cinquième laissé
tranquille parce qu'un teammate l'occupe à 100 %.

## Nos chiffres, sur la réserve honnête

Un réseau par filigrane, 6000 images synthétiques, 12 époques, bases COCO val2017, formes tirées des
vrais masques du train, réserve = images d'indice multiple de 5, jamais vues.

| Filigrane | Notre réserve | Celle de l'équipe (candidat 1) | Montage |
|---|---|---|---|
| pixelseal | 0,5714 | 0,522 | tronc du décodeur + FPN |
| trustmark | 0,5542 | 0,566 | tronc du décodeur + FPN |
| stegastamp | **0,7444** | 0,682 | U-Net encodeur ResNet-34 pré-entraîné |
| moyenne | **0,6233** | 0,590 | |

Soumission déposée : `submission_v3.pt`, bool [1020,128,128], 16,73 Mo, part filigranée 0,493.
**Score public 0,6369**, contre 0,7384 pour le meilleur de l'équipe. Le serveur garde le meilleur, donc
l'essai n'a rien coûté.

## Deux détails techniques qui valent d'être notés

StegaStamp du kit est en ONNX, sans module torch : leur variante « tronc du décodeur » échoue dessus
(`'StegaStamp' object has no attribute 'net'`). Le seul chemin qui marche est un U-Net à encodeur
ResNet-34 pré-entraîné, et c'est aussi le meilleur score. Sans `segmentation_models_pytorch` dans le
venv et sans pip, je l'ai réécrit avec torchvision, même contrat.

Le bf16 n'est pas le plafond. Test à données identiques sur pixelseal : 0,6564 en bf16 contre 0,6749
en fp32 sur le train entier, mais 0,5583 contre 0,5564 sur la réserve. Le gain est du bruit. L'hypothèse
du critique d'équipe sur le stem est donc écartée, mesures en main.

## Post-traitement : essai négatif

J'ai implémenté la reconnaissance de population par taux de bits justes (seuils appris : pixelseal
0,79, trustmark 0,76, stegastamp 0,74, exactitude 98,8 à 100 % sur le train), puis rebouchage des
trous pour les images pleines et plus grande composante pour les partielles.

- pixelseal : 0,5714 vers 0,5734
- trustmark : 0,5542 vers 0,5531
- stegastamp : 0,7444 vers 0,7124

Le classifieur de population marche parfaitement, mais les règles géométriques n'apportent rien et
coûtent 3 points sur stegastamp. Leur « assemblage » fait mieux avec d'autres règles, qu'on n'a pas.

## Ce qui manque pour passer devant

Leur 0,7384 vient de modèles entraînés sur les vraies images en 5 plis, moyennés, puis assemblés.
Ces artefacts vivent sur `/p/scratch/training2666/destors1/.../adem/t1-assemblage`, c'est-à-dire sur
JURECA. Aucun de nos quatre nœuds ne les a. Et pour accéder à JURECA il faut un compte Jülich, lancer
`hackathon_setup.sh` côté chef d'équipe puis `teammate.sh` côté autres, avec le nom de dossier d'équipe.

Autrement dit : la marge restante sur T1 n'est pas une question de réglage, c'est une question d'accès.
Avec JURECA, on reprend leurs plis réels et on ajoute notre stegastamp, qui est déjà 6 points devant le
leur. Sans ça, on plafonne autour de 0,64 sur notre propre matériel.
