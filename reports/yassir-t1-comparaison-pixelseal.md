# T1 · Comparaison PixelSeal hors ligne

Candidat 35 tient la première place avec 0,80327 (reçu 2550). Sa validation croisée sur 510 images est à 0,8138. Le seuil d’envoi suivant demande +0,02 sur les mêmes plis et des gains concordants entre types, plis et réserve. [Vérification du candidat 35](https://github.com/adembnn/EAC/issues/3#issuecomment-7072283437) · [Reçu 2550](https://github.com/adembnn/EAC/issues/3#issuecomment-7072363911)

Sur deux anciennes cartes PixelSeal, j’ai choisi parmi cinq coefficients de mélange en ajustant séparément sur les filigranes pleins et partiels. Chaque pli index modulo 5 est évalué avec les coefficients des quatre autres. Les cinq ajustements choisissent 50 % de carte v3SAM pour les pleins et 75 % pour les partiels.

| Ensemble | Images | Base | Mélange | Écart | IC95 apparié, stratifié plein/partiel |
| --- | ---: | ---: | ---: | ---: | ---: |
| Toutes | 170 | 0,65805 | 0,67321 | +0,01516 | [+0,00877 ; +0,02252] |
| Réserve, pli 0 | 30 | 0,63625 | 0,64130 | +0,00505 | [−0,00224 ; +0,01452] |

Sur la réserve, le gain maximal parmi les cinq coefficients choisis rétrospectivement par image est de +0,00655 sur PixelSeal, soit au plus +0,00218 toutes méthodes si le reste est identique. La réserve seule renseigne sur le transfert, car ces cartes datent d’avant la génération hors-pli complète de l’équipe. Je n’ai soumis aucun fichier. Les cartes du candidat 35 et les nouvelles cartes hors-pli sont demandées pour refaire la comparaison sur la base actuelle. [Demande de transfert](https://github.com/adembnn/EAC/issues/3#issuecomment-7072713175)

### Correcteur dense et sonde du décodeur

Le champ `full_watermark` est réservé au train. Les coefficients par groupe du premier essai servent au diagnostic. Le correcteur dense s'appuie uniquement sur les deux cartes de logits et sur les dérivées des canaux natifs de luminance et de chrominance, agrégées par cellule de 4 × 4. Une petite convolution à deux échelles apprend un résidu borné, avec loss mIoU et BCE. Sur les cinq plis, le mélange fixe passe de 0,67251 à 0,67353, soit +0,00103, IC95 [−0,00071 ; +0,00273]. Réserve 0,64153 → 0,64364. Recul de −0,00278 sur le pli 2. Entraînement CPU n2, 15,7 s.

Sur H200 n2, le décodeur PixelSeal a produit 7 cartes 128 × 128 sur les 170 images en 2,0 s, soit 83,2 images/s pour le lot d'inférence et de lecture. Pic PyTorch 529,8 MiB. Les logits signés normalisés ont obtenu 0,47477 seuls, les signes binaires 0,47151, et le mauvais message normalisé 0,28292. Une correction du mélange fixe apprise sur quatre plis avec chaque carte n'a donné aucun gain sur la réserve. Le canal de confiance du décodeur donne +0,00008, IC95 [−0,00413 ; +0,00404]. GPU libéré après la sonde.

Ces mesures utilisent les anciennes cartes accessibles. Le candidat 35 était la référence à cette étape. Aucun nouveau candidat à soumettre. Adem teste déjà les paires contrefactuelles à λ = 0,1, les petits trous et les entrées passe-haut sur les modèles pleine résolution. [Suivi d’Adem](https://github.com/adembnn/EAC/issues/3#issuecomment-7072799664)

### Contrôle des remplacements SAM sur petits masques

Candidat 36 est à 0,8262 hors pli contre 0,8138 pour 35, soit +0,0124 [0,0067 ; 0,0188]. Le seuil d’envoi reste +0,02. PixelSeal gagne +0,0268 en moyenne, avec deux plis négatifs. Sur la réserve, un petit rectangle perd 0,314 mIoU après remplacement par un masque SAM. [Comparaison 36–35](https://github.com/adembnn/EAC/issues/3#issuecomment-7072858473) · [Diagnostic PixelSeal](https://github.com/adembnn/EAC/issues/3#issuecomment-7072889358)

Sur les cartes partagées antérieures, j’ai essayé deux contrôles réglés sur quatre plis puis appliqués au cinquième. Le premier revient à la carte d’origine si les deux petits masques divergent fortement. Le second cherche parmi les propositions SAM celle qui augmente la mIoU attendue sous trois températures des logits. Les deux contrôles conservent PixelSeal v4 dans chacun des cinq plis, à 0,67571 global et 0,63988 sur la réserve. Les archives hors pli de 35 et 36 restent nécessaires pour confirmer ce résultat sur le remplacement précis du candidat 36.

### Nouveau meilleur officiel

Le relais d’Adem a soumis le candidat 36 après contrôle indépendant et dérogation au seuil. Le reçu 2754 confirme **0,8130917**, soit **+0,00982** contre le candidat 35. La base de validation croisée devient **0,8262**. Les deux contrôles ci-dessus portent sur les cartes antérieures. [Reçu 2754](https://github.com/adembnn/EAC/issues/3#issuecomment-7073085003)
