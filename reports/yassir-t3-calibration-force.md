# T3 · calibration de la force d'oubli (agent de Yassir)

21 novembre, 20h30 UTC, sur le troisième nœud H200 (66.201.6.94, IP privée 10.48.0.159), qui est
monté, synchronisé (9,1 Go de kit, modèles et release présents) et qui tourne.

Le vrai problème de T3 n'est pas la méthode, c'est qu'on ne peut pas mesurer. Chaque soumission coûte
30 minutes et on ne voit que la moitié publique du score. J'ai donc construit un jeu d'observation :
500 des 3500 images à oublier sont mises de côté et ne servent **jamais** au désapprentissage. Si le
modèle les traite encore comme avant, c'est que les poids n'ont pas dérivé globalement, donc que
l'accuracy sur un jeu caché tient encore.

## Le barème, à réglage identique (tête plus deux blocs, lr 1e-4, lot 32, température 2)

| Pas | KL sur les 500 en observation | Accord top-1 observation | KL sur les 3000 cibles | Confiance cibles |
|---|---|---|---|---|
| 1 | 0,0000 | 100 % | 0,0000 | 0,9996 |
| 2 | 0,0001 | 100 % | 0,0002 | 0,9993 |
| 4 | 0,0006 | 100 % | 0,0018 | 0,9979 |
| 5 | 0,0056 | 99,8 % | 0,0045 | 0,9960 |
| 6 | 0,0170 | 99,4 % | 0,0150 | 0,9913 |
| 7 | 0,1076 | 96,4 % | 0,1130 | 0,9625 |
| 8 | 0,7437 | 80,0 % | 0,8583 | 0,8812 |

## Ce que ça dit

La bascule n'est pas progressive, elle est brutale. Jusqu'à 6 pas, on ne bouge presque rien : KL de
0,015 sur les cibles, autant dire aucun oubli. À 8 pas, tout casse, y compris sur les images qu'on
n'a jamais touchées : 80 % d'accord seulement, donc le modèle a perdu du savoir général. Le seul
point qui achète de l'oubli sans casser le reste est **7 pas**, avec 96,4 % d'accord sur les images
en observation.

C'est exactement la zone où se joue le 0,0273 de l'équipe : un tout petit changement. Le score de
prédiction veut un modèle qui a oublié, le score d'utilité veut un modèle intact. Le curseur est
étroit, et il est mesurable localement grâce à ce découpage.

## Candidat poussé

`t3_cand_steps7.pt`, 344 Mo, KL 0,113 sur les cibles, accord 96,4 % sur les images en observation.
Soumission en boucle automatique sur la fenêtre de 30 minutes, qui se ferme dès qu'un autre agent
soumet. Le serveur garde le meilleur des deux, donc l'essai ne peut pas faire perdre le 0,0273.

## Suite logique

Balayer le pas 7 en finesse (7,2 puis 7,4 puis 7,6, par pas de 0,2 sur le nombre d'échantillons vus)
pour trouver le point exact où l'accord en observation commence à casser. C'est là que le score de
prédiction est maximal sous contrainte d'utilité.
