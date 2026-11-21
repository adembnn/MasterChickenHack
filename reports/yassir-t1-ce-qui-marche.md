# T1, ce qui marche et ce qui ne marche pas (agent de Yassir)

Relevé du 21 novembre, 18h30 UTC, sur les deux H200. Tous les chiffres sont mesurés sur les 510
images de train avec le score officiel du kit, sauf quand je précise « public », qui vient du serveur.

## Ce qui marche

- **Décodage par canevas**, fenêtre 256, pas 48, seuils par méthode : **0,4644** sur le train, et
  **0,4575** en public, ce qui veut dire que le train et le test se ressemblent à un demi-point près.
- **La carte par position de l'extracteur** vaut 4,8 points à elle seule. Sans elle, PixelSeal tombe
  de 0,4719 à 0,3355.
- **Le canevas plutôt que le recadrage** vaut 6,8 points, et 19 points sur StegaStamp seul.

Notre meilleur score d'équipe en public est de **0,697**, obtenu par un autre agent, avec une méthode
qui n'est pas la mienne. Ma contribution sur T1 est la mesure, pas ce score.

## Ce qui ne marche pas, et c'est la partie utile

| Essai | Résultat | Verdict |
|---|---|---|
| Recadrage de la fenêtre au lieu du canevas | 0,3943 contre 0,4627 | échelle du motif cassée, abandonné |
| Moyenne de deux échelles de fenêtre (256/64 + 320/80) | 0,4622 | aucun gain, le bruit ne se compense pas |
| Seuil par quantile de la carte (0,4 à 0,7) | 0,2993 à 0,3089 | catastrophique, abandonné |
| Seuil moyenne plus k écart-type | 0,2828 à 0,3095 | idem |
| Seuil milieu des deux modes 20/80 | 0,3131 | idem |
| Nettoyage morphologique du masque | 0,4627 | strictement aucun effet |
| Rectangle englobant le masque seuillé | 0,4563 | pire, les retouches ne sont pas des rectangles |
| U-Net image plus carte de décodage, 5 plis | 0,4519 | sous le décodage seul |
| U-Net image seule, un pli, 30 époques | 0,2231 | l'hypothèse « les retouches laissent des traces visibles » est fausse |

Deux conclusions nettes.

**Le seuil global calibré par méthode est la bonne règle de décision.** Toutes les variantes
adaptatives perdent 15 points ou plus. La carte d'accord a une échelle comparable d'une image à
l'autre, et vouloir l'adapter par image détruit l'information.

**Ajouter un réseau par-dessus le décodage ne sert à rien aujourd'hui.** Ni avec la carte en entrée,
ni avec l'image seule. Le signal utile est déjà dans la carte d'accord, et les 510 masques sont trop
peu nombreux pour qu'un réseau apprenne autre chose que du bruit.

## Ce que ces échecs déplacent

Le plafond du décodage seul est autour de 0,46. Le 0,697 de l'équipe est donc obtenu par autre chose
que du réglage. Pour progresser sans refaire ce travail, la seule voie raisonnable est de demander à
l'agent qui tient ce pipeline quels signaux il utilise, et de les compléter, au lieu de repartir de
zéro chacun de son côté.

## Détail technique réutilisable

Le décodeur StegaStamp en ONNX réclame jusqu'à 1,43 Go par tampon quand le lot de fenêtres grossit :
il faut découper par 8, et même ainsi une passe à 121 fenêtres par image finit par épuiser l'arène
mémoire de l'ONNX Runtime. En pratique, rester à 36 fenêtres maximum par image (fenêtre 256, pas 48)
et oublier les fenêtres 192 avec pas 48.
