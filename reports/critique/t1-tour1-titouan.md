# T1 · critique et recherche, tour 1 (agent critique de Titouan)

Sam. 21/11, ~23 h 45. Le fichier `t1-tour1.md` existe déjà (critique de l'agent d'Adem, 17 h) : ce rapport porte donc un nom distinct.
Lu : issue #3 en entier (1 140 lignes), les critiques `t1-*.md`, `t1-recherche-trous.md`, le kit. Rien soumis.
Décision débloquée : ce qu'on arrête sur PixelSeal et les pleines TrustMark, et les 5 expériences suivantes, classées.
Scripts : `tracks/t1-watermark/experiences/critique_titouan/`. Sorties : `h200b:~titouan/t1crit/log*.txt`.

## 1. Ce qui limite chaque filigrane (10 lignes)

1. **Le score se joue sur la petite classe.** Aucune image n'est à une seule classe. Le score d'une image vaut à peu près 0,5 + 0,5 × l'IoU de la petite zone. Rater un trou de 3 % coûte autant que rater la moitié de l'image.
2. **PixelSeal (0,648).** Pleines : IoU du trou 0,37, et 0,26 pour la famille « taches » (53 % des pleines, 62 % de la perte). Le réseau classe bien les régions (AUC 0,965), mais son tracé est flou et il rate les trous de moins de 3 %.
3. **PixelSeal, rectangles.** IoU filigrané 0,24. C'est une limite physique pour la moitié : 21 rectangles sur 43 ont un petit côté de moins de 32 px, et le décodeur ne les relit pas même avec la boîte exacte (0,53 de bits justes).
4. **PixelSeal, ce qui ne marche pas.** Le décodeur d'origine décide sur toute l'image : ses votes position par position sont justes à 0,79 dans le trou comme dehors. Le filtre adapté, SAM et la carte position par position (AUC 0,66) n'aident pas. Le passage du premier étage en pleine résolution, qui a payé pour TrustMark, n'a pas encore été poussé à fond pour PixelSeal.
5. **TrustMark pleines (0,659).** IoU du trou 0,40 en réserve (0,49 hors-pli avec `f_s1`). Trous de moins de 2 % : 0,24. Le seul levier qui a payé est la résolution du premier étage (pas 1 au lieu de 2 : +0,065). Le motif TrustMark est calculé sur une grille de 256 points étirée sur l'image, d'où le gain.
6. **TrustMark partielles (0,843).** Réglé par les cadres calés au pixel (re-tamponnage) et le vote SAM. Il reste peu à gagner.
7. **StegaStamp (0,842).** Les trous sont trouvés mais mal dessinés. Pas prioritaire.
8. **Le post-traitement est épuisé.** Même un seuil parfait image par image ne rend que +0,02 à +0,03. Les formes paramétriques expliquent au plus 15 % du gain, SAM rien sur les trous. Mes quatre mesures ci-dessous le confirment de quatre autres façons.
9. **Ce qui limite, c'est l'information dans la carte, pas la décision.** Aux petits trous, la carte n'a pas de creux caché sous le seuil (E3). Connaître la taille du trou n'aide pas non plus (E4).
10. **Conséquence.** Le prochain +0,02 doit venir de nouvelles cartes : plus de résolution, plus de vraies images (même sans étiquette), d'autres signaux. Pas de la fusion.

## 2. Mesures faites (4 expériences de moins de 30 min, CPU et un passage décodeur sur la H200)

**Protocole.** La réserve est celle du vérificateur : indice global % 5 == 0 dans `release/train/metadata.json` (30 PS / 39 TM). Confirmation demandée sur #3. Les réglages sont faits hors réserve. On compte deux chiffres : (a) validation croisée en 4 plis sur le hors réserve (le pli `idx % 5`), (b) réserve lue une seule fois. Les écarts sont appariés, avec un IC95 par bootstrap. La population pleine/partielle vient de la vérité (`full_watermark`). Au test, le routeur par les bits a une exactitude de 98 %.
**Cartes.** J'ai utilisé les seules lisibles pour moi sur h200b : `/tmp/eac-t1-partage/pred_train_{pixelseal,trustmark}.npz` (PixelSeal fusion v5, TrustMark `p_mixr`). Les cartes du candidat 20 (`f_s1a`) sont dans `/home/adem`, qui m'est fermé.
**Base.** Un seuil par population, réglé hors réserve. Il reproduit les chiffres du vérificateur : PS 0,6362, TM 0,5985.

| Expérience (ce qu'elle tranche) | PS réserve | PS CV hors réserve | TM réserve | TM CV hors réserve | Verdict |
|---|---|---|---|---|---|
| **E1 Règle à IoU espérée maximale** (Nowozin 2014) : probas calibrées par régression logistique hors réserve, puis pour chaque image le nombre k de cases filigranées qui maximise 0,5·(E-IoU₁ + E-IoU₀). Proposée par deux critiques, jamais mesurée. | −0,013 [−0,019 ; −0,008] | −0,001 [−0,005 ; +0,002] | +0,014 [+0,003 ; +0,026] | +0,004 [−0,001 ; +0,010] | **Nul.** Le gain TM en réserve (pleines +0,021) ne se retrouve pas en CV (+0,000 sur les pleines) : c'est du bruit. Fermé. |
| **E2 A priori de position** : carte moyenne de la vérité hors réserve (par filigrane et population, lissée σ = 3), ajoutée aux logits avec un poids w ∈ {0 ; 0,25 ; 0,5 ; 1 ; 2}. Les trous longent le bord 79 % du temps. | −0,015 [−0,025 ; −0,005] | −0,004 [−0,008 ; −0,000] | +0,003 [−0,005 ; +0,012] | −0,003 [−0,008 ; +0,002] | **Négatif.** La position n'apporte rien que la carte n'ait déjà. Fermé. |
| E1 + E2 | −0,017 | −0,006 | +0,008 | +0,003 | Fermé. |
| Lissage gaussien des logits (σ = 1 ou 2) avant le seuil | +0,001 / −0,001 | +0,002 / +0,003 | −0,001 / −0,003 | −0,000 / −0,002 | Effet ≤ 0,003. Inutile. |
| **E3 Chapeau haut-de-forme noir** sur les pleines : L′ = L − a·(fermeture(L, s) − L), qui fait ressortir les petits creux locaux. a ∈ {0,5 ; 1 ; 2}, s ∈ {5 ; 9 ; 17 ; 33}, seuil re-réglé. | 0,000 | 0,000 | 0,000 | 0,000 (pleines) | **a = 0 choisi dans 100 % des plis sur les pleines.** Aux trous ratés, la carte n'a pas de creux caché sous le seuil : l'information n'y est pas. |
| **E4 Aire du trou estimée par le décodeur officiel** : la marge du décodeur sur l'image entière sert à prédire la part propre, puis le seuil de chaque image est choisi pour rendre cette aire (mélange λ avec le seuil global, réglé hors réserve). | 0,000 (λ = 0 choisi) | 0,000 | +0,002 [−0,003 ; +0,006] | +0,000 [−0,003 ; +0,003] | **Nul**, mais voir le fait nouveau ci-dessous. |

**Fait nouveau mesuré (E4).** La marge du décodeur officiel sur l'image entière prédit la taille du trou des pleines.
- Corrélation de Spearman entre la marge moyenne et la part propre : **TrustMark −0,76**, PixelSeal −0,49 (−0,59 avec la moyenne des tanh). Sur les partielles : −0,54 et −0,36.
- Le décodeur TrustMark « sent » donc quelques pour cent de surface propre. Il sait combien, pas où.
- Imposer cette aire au seuil ne sert à rien : la carte ne classe pas les cases du trou au-dessus des autres.
- En revanche, une recherche de région guidée par le décodeur a de quoi mesurer son effet. C'est l'expérience 5.

Fichiers : `margins_train.npz` (marge, exactitude, médiane et tanh des 510 images, 3 filigranes) dans `h200b:~titouan/t1crit/`. Je le copie dans `/tmp/eac-t1-partage/` si besoin.

## 3. Expériences suivantes (5), classées par gain attendu

Rappel : le prochain feu vert demande +0,02 sur la moyenne des trois, soit environ +0,06 en somme sur les filigranes. Aucune piste de post-traitement n'y arrive seule.

| # | Expérience | Ce qu'elle tranche | Gain attendu | Coût |
|---|---|---|---|---|
| 1 | **PixelSeal U-Net en pleine résolution, recette `f_s1` complète.** Pré-entraînement de 8 000 pas sur exemples fabriqués (toile réelle carrée 512 + s), puis affinage en 5 plis sur le réel. Ensuite, moyenne des logits avec G et `f_deca`, seuil par population hors réserve. Aujourd'hui : 0,613 brut à 6 000 pas en fabriqué seul, et déjà mieux que le tronc décodeur sur les partielles (0,59). | Le levier « résolution du premier étage » (+0,065 sur TrustMark) vaut-il aussi pour PixelSeal ? | PS +0,02 à +0,04 | 2 à 3 h H200, 10 min d'assemblage |
| 2 | **Auto-apprentissage sur les 1 020 images de test** (hors évident). Pseudo-étiquettes du candidat 20 aux seules cases sûres (\|logit\| > τ, le reste ignoré dans la perte), puis affinage de 1 000 pas des U-Net TM et PS, comme l'affinage sur le réel. Ce sont 340 images par filigrane contre 136 vraies : les vraies images ont été le meilleur levier (StegaStamp +0,07 à +0,12). Aucune étiquette de test utilisée. | Plus de vraies images, même sans étiquette, améliorent-elles le tracé des trous ? Juger sur la réserve et l'IoU du trou des pleines. | TM et PS +0,01 à +0,03 chacun | 1 à 2 h H200 |
| 3 | **Ensemble de logits PixelSeal sur les pleines : G (candidat 20) + `f_deca`** (réserve pleines 0,670 contre 0,647), plus le U-Net du n° 1 quand il existera. Précédent : moyenne A + D, +0,013 [+0,002 ; +0,026]. Pour TrustMark, `f_s1` + `f_s1a` est déjà nul : ne pas refaire. | Les deux troncs se trompent-ils sur des trous différents ? | PS +0,01 | 10 min CPU, une fois `f_deca/pred_*` partagé |
| 4 | **Fusion « réseau + re-tamponnage + effaceur » (PFUSE3, t1-residus) refaite sur la base `f_s1` / `f_s1a`**, au lieu de `rangP-reel-mixr`. Sur l'ancienne base : pleines TM +0,025 ± 0,006 hors réserve, et l'effaceur complète le re-tamponnage. | Le gain reste-t-il sur une carte déjà meilleure ? | TM +0,005 à +0,01 | 30 min (cartes du test déjà calculées par t1-residus) |
| 5 | **Vérifier les régions candidates par le décodeur, en ajoutant du filigrane** (hors évident). Pour une pleine TM, on prend les K régions candidates (composantes des cases de logit le plus bas, ou cases de 32 px). Pour chacune, on y colle `embed(x, message)` à la place de x et on lit la hausse de marge du décodeur. Une région propre fait monter la marge, une région déjà filigranée ne la change pas. E4 montre que le décodeur TM est sensible à quelques pour cent de surface (ρ = −0,76). C'est le sens inverse de l'essai « effacer et lire la chute » de t1-pixelseal (AUC 0,80) : coller du filigrane ne détruit pas l'image et ne dépend pas d'un remplissage. | Le décodeur peut-il départager les trous candidats que la carte classe mal ? AUC par région, puis fusion. | Inconnu, plafond TM pleines +0,05 si AUC ≥ 0,9 | 30 min de code, ~20 appels décodeur par image |

**À arrêter**, mesuré ici ou sur #3 :
- règles de décision et de seuil (E1, E4, seuil par image) ;
- a priori de position et de forme (E2, formes paramétriques) ;
- lissage et morphologie (E3) ;
- SAM sur les trous des pleines ;
- carte position par position du décodeur PixelSeal.

## 4. Critique du protocole (court)

- **La réserve de 102 images reste le goulot.** IC95 de ±0,02 sur la moyenne, et l'écart pleines réserve / hors réserve va jusqu'à ×2 pour TrustMark (+0,043 contre +0,085). E1 en donne un exemple : +0,021 sur les pleines TM en réserve, +0,000 en CV.
- **Mon conseil.** Garder la CV en 4 plis hors réserve comme second juge obligatoire (déjà proposé par le vérificateur), et refuser un gain qui n'apparaît que sur la réserve.
- **Bonne pratique à garder.** Le vérificateur reproduit les chiffres à la 4ᵉ décimale. Ma base les reproduit aussi (0,6362 et 0,5985), ce qui valide le découpage utilisé ici.
- **Le score des pleines dépend de cases rares.** Un plancher de 0,5 % de cases propres est neutre (mesuré par t1-assemblage). Ne pas confondre « plus de sorties tout-1 » avec un gain.

[agent critique de Titouan · T1]
