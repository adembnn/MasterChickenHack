# T3 Unlearning : étape zéro, mesures réelles (21 nov. 2026)

Décision débloquée : quel est le jeu de 100 classes, à quel point orig est sûr de lui sur l'oubli, existe-t-il un proxy public. Tout vient de commandes lancées sur JURECA (RTX 8000, torch 2.15, timm 1.0.31), fp16 autocast comme le serveur. Chiffres bruts : `E:\EAC-lab\t3\step0.json`. Scripts : `E:\EAC-lab\t3\scripts\`. Planches : `contact.png`, `classes_0.png`, `classes_1.png` (même dossier).

## En images simples
- Orig connaît ces 3500 fiches par cœur : sûr à 99,97 % en moyenne, sur toutes les images.
- Ce ne sont pas des photos ImageNet ou CIFAR : ce sont des images générées par IA (lanternes, figurines de renard, phares, pains, ruches). Jeu maison.
- Les 3500 images ne couvrent que 45 classes sur 100.
- Aucun proxy public identifié. Repère hors-domaine mesuré : sur des photos ImageNet, orig tombe à 0,74 de confiance moyenne.

## 1. Orig sur les 3500 images (mesuré, `a_base.py`, chargement strict, `predict()` du template, batch 64)
```
maxprob_mean 0.99968   median 0.99999   share p>0.9 = 1.0   p>0.99 = 0.9937   p>0.999 = 0.9554
hist max-prob (10 cases de 0 à 1) : [0,0,0,0,0,0,0,0,0,3500]
entropie moyenne 0.00214 (médiane 0.000115)   max-prob à T=2 : 0.9855
loss moyenne vs pseudo-label 0.000323 (écart-type 0.00244)
marge logit top1 - top2 : moyenne 12.24, médiane 12.14
classes prédites présentes 45, vides 55 ; effectifs min 46, max 130, écart-type 43.7
```
- Les pseudo-labels (prédiction d'orig) sont fiables : 100 % des images au-dessus de 0,9 (le seuil du plan était 95 %). Ce sont les étiquettes d'orig, pas les vraies, qu'on n'a pas.
- Les effectifs ne sont pas « ≈35 par classe » : 45 classes seulement ; huit classes à exactement 130 (0, 7, 8, 29, 30, 31, 33, 38).
- Classes présentes : 0-8, 12, 15, 19, 21, 23, 25, 28-33, 35-45, 51-53, 57, 69, 72, 81, 83, 85, 86, 88, 92, 93.
- Seules 34,6 % des images ont pour 2e classe la 2e classe la plus fréquente de leur classe prédite : la « mauvaise classe » varie d'une image à l'autre (mesuré).

## 2. Ce que montrent les planches (vues à l'œil)
- Images 224x224 nettes, style rendu de génération d'images : bokeh fort, lumière chaude, cadrage « film ». Texte déformé sur certains objets, signe typique de génération (impression visuelle, non mesurée).
- Classes nommées à l'œil : 0 figurine de renard ; 2 figurine de lièvre ; 5, 6, 7 trois sortes de lanternes ; 15 champignon ; 21 aloe vera ; 28 phare ; 29 à 33 bateaux ; 36 chaise pliante ; 38 chaise design ; 39 éolienne ; 41 moulin à eau ; 43 plaid rouge ; 44 panier en osier ; 51 bottes ; 52 baskets ; 57 pain ; 72 cerf-volant (avec des oiseaux) ; 85 ruche ; 86 nichoir ; 88 origami ; 92 ancre ; 93 cadran solaire.
- Les classes sont fines et voisines (3 lanternes, 5 bateaux, 3 chaises). Le gold, qui n'a pas vu ces images, se trompera surtout vers la classe sœur. C'est le terme « quelle mauvaise classe » du score (déduction, non mesurée).
- Images ambiguës par construction (cerf-volant/oiseau, barque/bateau) : probables « difficiles » du gold.

## 3. Identification du jeu
- Mesuré : gradient horizontal moyen sur 1000 images : oubli 5,59 ; ImageNet-100 val redimensionné à 224 : 8,41. Les photos IN100 font ~160 px de côté court (tailles vues : 160x213, 194x160). Les images oubliées sont plus lisses, mais cela peut venir du bokeh du style IA, pas d'un agrandissement.
- Contenu : trois lanternes, cadran solaire, ruche, origami en 100 classes équilibrées ne correspondent ni à ImageNet-100, ni à CIFAR-100, ni à Food-101.
- Verdict : jeu synthétique maison, confiance estimée 90 % (jugement, pas mesure) que ce n'est ni IN100 ni CIFAR-100. Non identifié plus précisément. Pas de recherche de doublon exact : aucun candidat public plausible.

## 4. Robustesse d'orig sur copies modifiées (mesuré, 3500 images, `c_robust.py`)
Colonnes : accord avec le pseudo-label, max-prob moyenne, loss moyenne vs pseudo-label.
```
propre                      1.0000   0.9997   0.0003
recadrage 0.5-1             0.9929   0.9926   0.0268
flip horizontal             0.9997   0.9996   0.0006
recadrage centre 0.7        0.9997   0.9989   0.0013
JPEG q=50                   1.0000   0.9990   0.0010
bruit s=0.05                0.9997   0.9990   0.0012
bruit s=0.10                0.9754   0.9801   0.0828
flou s=1.5-2.5              0.8817   0.9350   0.4545
couleur (0.4,0.4,0.4,0.1)   0.5183   0.8386   3.0373
```
- Orig est très stable aux coupes, flip, JPEG, bruit léger. Seuls couleur forte (51,8 % d'accord) et flou cassent sa confiance.
- La confiance ne tient donc pas à un pixel exact. On ne peut pas séparer « mémorisé » de « classe facile » sans images non vues du même jeu : non mesuré.
- Si les copies cachées du serveur sont de type couleur ou flou, c'est là que le gold et orig diffèrent le plus.

## 5. Proxy et hors-domaine (mesuré, 5000 images chacun, `d_ood.py`)
Pas de jeu public du même domaine : pas d'estimation directe de la loss d'un non-membre. Caractérisation hors-domaine seulement (étiquettes non comparables, donc pas de loss) :
```
ImageNet-100 val (5000)     max-prob moy. 0.737, médiane 0.778, p>0.9 : 35.0 %, entropie 0.80, 100 classes prédites (classe 89 : 300 fois)
CIFAR-100 test agrandi      max-prob moy. 0.752, médiane 0.808, p>0.9 : 38.8 %, entropie 0.78, 90 classes prédites (classe 83 : 1425 fois)
Oubli (référence)           max-prob moy. 0.9997, p>0.9 : 100 %, entropie 0.0021
```
- Hors-domaine, orig n'est pas saturé : il hésite quand il n'a pas vu l'image. Cela appuie l'idée que le gold hésite sur l'oubli, mais ces images sans rapport ne sont pas un substitut du gold.
- IN100 et CIFAR ne servent que d'ancre « ne pas déformer ailleurs » (ancre OOD de P1), pas de proxy.
- Estimation (non mesurée) : accuracy non-membre du gold de 0,8 à 0,9 sur classes fines ; le plan prenait 0,85. Rien ici ne le confirme.

## 6. Baseline du template (mesuré)
```
Saved baseline submission to adem/submission_baseline.pt.
strict load: <All keys matched successfully>
dtypes {'torch.float32'}  nan/inf False  size MB 343.63  n_params 85893988
```
Conforme : float32, strict, 344 Mo (limite 450). Rien envoyé au serveur.

## Ce qui change pour le plan
1. À T=2 orig reste à 0,985 et la marge de logit est de 12 : orig ne fournit aucune hésitation à reprendre. La cible de loss doit être fabriquée (N1, N2), un simple adoucissement de température (P3) ne suffit pas.
2. L'effort porte sur 45 classes aux voisines proches : choisir la mauvaise classe par similarité des vecteurs de classe de la tête (lanternes 5-7, bateaux 29-33, chaises 35-38).
3. Pas de proxy in-domain : P2 (effacer puis réparer) manque de matière. Reste les 3500 images augmentées ou des images générées par nous. Les 55 classes non oubliées n'ont aucune image réelle pour surveiller le retain.
4. Couleur et flou sont les copies modifiées les plus risquées : à garder en contrôle hors entraînement.

## Mesuré / vu / deviné
- Mesuré : tous les blocs de code ci-dessus.
- Vu à l'œil : noms de classes, style généré.
- Deviné : jeu synthétique à 90 %, accuracy non-membre 0,8-0,9, nature des copies modifiées.

## Questions (regroupées, avec défaut)
1. Aux superviseurs : (a) images générées du même style autorisées comme proxy ? (défaut : non, on s'en passe) ; (b) quota de soumissions ? ; (c) liste des 100 classes ?
2. Générer un proxy avec un modèle d'images gratuit local ? Recommandation : seulement si 1(a) est oui. Défaut : non.

Note : `pyarrow` installé dans `adem/pylib` du cluster seulement (venv partagé intact). Le fichier .env de l'équipe n'a pas été lu.
