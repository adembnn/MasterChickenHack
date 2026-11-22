# T1 · Comparaison PixelSeal hors ligne

Candidat 35 tient la première place avec 0,80327 (reçu 2550). Sa validation croisée sur 510 images est à 0,8138. Le seuil d’envoi suivant demande +0,02 sur les mêmes plis et des gains concordants entre types, plis et réserve. [Vérification du candidat 35](https://github.com/adembnn/EAC/issues/3#issuecomment-7072283437) · [Reçu 2550](https://github.com/adembnn/EAC/issues/3#issuecomment-7072363911)

Sur deux anciennes cartes PixelSeal, j’ai choisi parmi cinq coefficients de mélange en ajustant séparément sur les filigranes pleins et partiels. Chaque pli index modulo 5 est évalué avec les coefficients des quatre autres. Les cinq ajustements choisissent 50 % de carte v3SAM pour les pleins et 75 % pour les partiels.

| Ensemble | Images | Base | Mélange | Écart | IC95 apparié, stratifié plein/partiel |
| --- | ---: | ---: | ---: | ---: | ---: |
| Toutes | 170 | 0,65805 | 0,67321 | +0,01516 | [+0,00877 ; +0,02252] |
| Réserve, pli 0 | 30 | 0,63625 | 0,64130 | +0,00505 | [−0,00224 ; +0,01452] |

Sur la réserve, le gain maximal parmi les cinq coefficients choisis rétrospectivement par image est de +0,00655 sur PixelSeal, soit au plus +0,00218 toutes méthodes si le reste est identique. La réserve seule renseigne sur le transfert, car ces cartes datent d’avant la génération hors-pli complète de l’équipe. Je n’ai soumis aucun fichier. Les cartes du candidat 35 et les nouvelles cartes hors-pli sont demandées pour refaire la comparaison sur la base actuelle. [Demande de transfert](https://github.com/adembnn/EAC/issues/3#issuecomment-7072713175)
