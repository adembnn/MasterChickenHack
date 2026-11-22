# T3 : le jeu d'images et le comportement du gold (recherche)

Décision débloquée : régler la cible « gold » (loss moyenne, écart-type, classes choisies) de notre méthode d'unlearning.
Termes : **gold** = modèle réentraîné sans les 3 500 images, moyenné sur plusieurs entraînements ; **loss** = entropie croisée avec la vraie étiquette (que seul le serveur connaît) ; **forget set** = les 3 500 images à oublier.

## Conclusion chiffrée (5 lignes)
1. Aucune source publique ne donne le jeu, ses étiquettes, ses effectifs ou le gold : la fiche Hugging Face n'a pas de README. Tout chiffre sur le gold est une estimation (confiance faible à moyenne).
2. Estimation du gold sur le forget set : précision 0,5 à 0,7, loss moyenne 2 à 4 (centre 3), écart-type 3 à 5 ; la loss est bimodale : environ 33 % des images à loss 6 à 11, le reste à loss 0,3 à 1.
3. Cette estimation repose sur une déduction de l'équipe (9 classes à 130 images = classes entières retirées), que j'ai relue et qui tient : un plafond identique de 130 sur 8 classes (+ 129 pour une, erreur d'`orig`) n'a pas d'autre explication simple. Confiance moyenne, non confirmée.
4. Le jeu n'est pas un jeu connu : ce sont des images de synthèse « photo cinématographique » de 100 objets ou scènes fins (figurines de renard, lanternes, sabliers, barques, chaises, mer vue du ciel…). Pas ImageNet-100, pas CIFAR-100 agrandi. L'effectif d'entraînement probable est 13 000 (130 × 100).
5. Littérature : un ViT-B/16 entraîné de zéro sur ≈ 50 000 images de 100 classes atteint 60 % (CIFAR-100, 100 époques) ; sur des classes visuellement distinctes de synthèse, 80 à 90 % est plausible mais non sourcé. Aucune publication ne donne la distribution de la loss hors entraînement : ce chiffre n'est pas sourcé.

## 1. Fiche Hugging Face et dépôts de l'organisation
- [vérifié] https://huggingface.co/datasets/SprintML/unlearning : « No dataset card yet » (pas de README). Créé le 2026-11-17, dernière modif. 2026-11-19, 870 466 309 octets au total. Auteur de tous les commits : `maitri01` (Maitri Shah, nommée dans l'énoncé). Le visualiseur Hugging Face affiche `requirements.txt` comme un jeu de 5 lignes de texte (`torch>=2.3`, `timm>=0.9.12`, `numpy>=1.24`, `requests>=2.28`) : c'est un artefact, pas des données.
- [vérifié] Fichiers (API `https://huggingface.co/api/datasets/SprintML/unlearning/tree/main`) : `forget_images.npz` 526 848 266 o ; `model.pt` 343 617 075 o ; `task_template.py` 7 215 o ; `submission.py` 2 742 o ; `requirements.txt` 78 o. Aucun fichier de jeu complet, aucun jeu de validation, aucun évaluateur.
- [vérifié] Historique : 15 commits, du 17 au 19/11. Seuls signaux : « Participant files (ViT-B) » ; « Task template: runnable baseline, uint8 NHWC loader » ; « Size limit 450 MB » ; « task_template: forget score is product of the two scores » ; « task_template: describe stronger hidden copies » ; « task_template: generic wording for hidden copies » (18/11). Les copies cachées ont donc été durcies puis leur description rendue générique la veille de l'événement : prudence sur les transformations légères (consigne à la lettre : « stronger »). Source : `https://huggingface.co/api/datasets/SprintML/unlearning/commits/main`.
- [vérifié] `https://huggingface.co/datasets/SprintML/hackathon` : README de 8 232 o, deux scripts (`hackathon_setup.sh`, `teammate.sh`). Le script télécharge `SprintML/Hackathon_Watermark_Localisation_Task`, `SprintML/model_equality` et `SprintML/unlearning` ; rien sur le jeu d'images, ni sur l'évaluation.
- [vérifié] Liste des 23 jeux de l'organisation (`https://huggingface.co/api/datasets?author=SprintML`) : aucun autre jeu d'unlearning. Les `tml26_task1/3/4`, `P4Ms-hackathon-vision-task`, `iar`, `model_tracer` sont d'autres tâches (attaque d'appartenance sur ResNet-18 32×32 à 9 classes, tâche de ResNet à 9 classes, LMM, filigrane). Aucun n'aide pour ce jeu. Modèles : `SprintML/tfms`, `SprintML/tml26_task2` (modèles suspects, attribution), hors sujet.
- [vérifié] GitHub https://github.com/sprintml (52 dépôts, 15 visibles sur la page) : aucun ne parle d'unlearning, de hackathon ou de championnat (copyrighted_data_identification, BitMark, agent_prov, finding_dori, Radium…). `agent_prov` correspond à T2 (déjà dans `reports/t2-recherche-agentprov.md`).
- Pages CISPA des éditions précédentes (Varsovie, Vienne, Stockholm, Barcelone, Munich, Paris, finale) : descriptions de tâches en prose (attaque adverse, attribution de modèle, filigrane, données privées), jamais de code ni d'évaluateur. Aucune ne cite l'unlearning. https://cispa.de/en/cispa-hackathon-championship-paris
- [vérifié] Template local `E:\EAC\reports\sources\t3\task_template.py` : « le gold est le même modèle, même recette, mêmes données SANS les images à oublier » ; entrées normalisées à [-1, 1] ; score de loss et de prédiction comme dans l'énoncé ; « pour chaque image (et ses variations) votre modèle doit prédire ce qu'un modèle qui ne l'a jamais vue prédirait, y compris QUELLE mauvaise classe il choisirait ».

## 2. Le jeu ressemble-t-il à un jeu connu ?
Fichier présent en local : `E:\EAC-lab\t3\forget_images.npz` (3 500 × 224 × 224 × 3). J'en ai extrait 40 images au hasard et 2 classes (par prédiction d'`orig`, 24 images chacune) ; planches dans le dossier temporaire de la session, non versionnées.

[vérifié, à l'œil] Contenu et style :
- Un seul style : photographie « cinématographique » de lumière dorée ou de contre-jour, faible profondeur de champ, teintes chaudes ou vert-bleu, grain léger. Aucune image web hétérogène (pas de fonds blancs de catalogue, pas de sujets humains).
- Les classes sont des sujets fins et voisins, avec un sujet central presque toujours net : figurine de renard (blanche, dorée, bois, bleue, sur table, au jardin, mur), chaise (design jaune à pieds fuselés, fauteuil de bureau, chaise longue avec plaid), lanterne (au moins 3 variantes), sablier (au moins 2), barque / voilier / bateau de pêche, mer vue du ciel avec un petit bateau, origami (grue), héron, éolienne et moulin, nichoir, ancre, hibou en figurine, plaid tricoté, pain sur étagère.
- Variations d'une même classe : très fortes en décor, éclairage, matière et angle (blanc / doré / brun), mais le sujet reste identifiable. Dans la planche « renard », les 24 images sont toutes des figurines ou sculptures de renard ; dans la planche « chaise », toutes montrent une chaise ou un fauteuil. Deux images d'une même classe se ressemblent donc par le sujet, pas par le fond.
- Artefacts de génération : formes un peu lissées, objets parfois déformés (pieds de chaise, mains de la figurine), reflets incohérents ; l'ensemble est cohérent avec un générateur de diffusion récent (à prompts du type « sujet + décor + éclairage »). Je n'ai pas pu identifier le modèle.
- Résolution d'origine : le spectre de puissance décroît régulièrement jusqu'aux hautes fréquences, sans motif en blocs de 8 pixels (écart de gradient colonne par colonne constant ≈ 5,65 sur les 8 positions) : pas de compression JPEG visible. Pas de signe d'agrandissement fort depuis 64 ou 128 px ; compatible avec des images natives plus grandes (≥ 512) réduites à 224.
- [vérifié] Pas ImageNet-100 : les classes (figurines, scènes aériennes de mer, lanternes) ne sont pas des classes ImageNet isolées et le style est unique. Pas CIFAR-100 agrandi : le détail est net à 224. Pas un jeu publié de type « ImageNet-SD » / « SynthCLIP » / « GenImage » (ils utilisent des noms de classes ImageNet / de légendes web, des styles variés). Recherche web de deux requêtes : aucune correspondance. Le jeu est « custom », comme l'énoncé le dit.
- [supposé] 13 000 images d'entraînement : 130 par classe × 100 classes. Raison : 8 classes à exactement 130 dans le forget set, une à 129. À défaut, 130 est le nombre d'images d'une classe entière ; je n'ai aucune preuve du nombre total.

## 3. Précision et loss de test d'un ViT-B/16 de zéro sur petit jeu
Chiffres sourcés :
- [vérifié, lu dans le PDF] Tab. 9 de « Efficient Training of Visual Transformers with Small Datasets » (https://arxiv.org/abs/2106.03746) : ViT-B/16 (86,4 M de paramètres) entraîné de zéro 100 époques, résolution 224 : CIFAR-10 71,70 %, **CIFAR-100 59,67 %**, Infograph 11,79 %. Avec leur perte auxiliaire : 73,91 / 61,42 / 12,22.
- [vérifié] Même article, Tab. 3, ImageNet-100 (≈ 126 000 images, 100 classes), 100 époques de zéro : CvT-13 85,62 %, Swin-T 82,66 %, T2T-ViT-14 82,67 % ; à 300 époques : 90,16 / 89,68 / 87,76. Ce sont des ViT hybrides, plus efficaces que ViT-B/16 sur petit jeu. L'écart entre architectures de ViT sur petit jeu atteint 20 à 45 points sur certains jeux.
- [vérifié sur résumé] « Understanding Why ViT Trains Badly on Small Datasets » (https://arxiv.org/abs/2302.03751) : un ViT a une précision d'évaluation « nettement plus basse » qu'un ResNet-18 de taille voisine. Chiffre cité par la recherche web pour ce papier (non relu dans le texte) : CIFAR-100 ViT 54,31 % contre ResNet-18 70,7 %.
- Non retrouvé : le tableau précis de DeiT-B de zéro sur ImageNet-100, et la table de « Vision Transformer for Small-Size Datasets » (https://arxiv.org/abs/2112.13492, PDF trop gros pour mon outil).

Ce qui manque : **aucun papier retrouvé ne donne la distribution de la loss sur des images de test (moyenne, écart-type, part d'erreurs confiantes) pour un ViT de zéro sur petit jeu.** Les chiffres de loss du gold ci-dessous sont donc des déductions, pas des sources.

Déduction pour le gold (supposé) :
- Précision hors entraînement sur les classes qu'il connaît : 0,75 à 0,90. Raison : classes visuellement distinctes mais voisines (3 lanternes, 5 bateaux), 100 images par classe et un ViT-B de zéro ; la borne basse (0,6) est la valeur CIFAR-100 ci-dessus, la haute vient d'ImageNet-100 avec des modèles à biais convolutif. L'énoncé exige ≥ 70 % sur la validation pour la porte, ce qui est cohérent avec un modèle à 85 à 95 % (l'organisateur laisse une marge) ; pas une preuve.
- Un ViT de zéro mémorise son entraînement (loss ≈ 0 sur ses données, comme `orig` : probabilité maximale moyenne 0,9997 sur le forget set, loss pseudo-étiquette moyenne 0,0003, mesures de l'équipe `E:\EAC-lab\t3\step0.json`). Sur des images non vues il est donc très confiant : les erreurs ont une loss de 4 à 12, pas de 1 à 2.
- Pour les 9 classes à 130 (1 169 images, 33,4 %), si elles sont absentes de l'entraînement du gold, la bonne classe n'existe pas : p(vraie classe) ≈ 1e-3 à 1e-5 donc loss 7 à 11 (supposé : 5 à 12). Pour les 36 classes partielles : loss moyenne 0,4 à 1,2 (précision 0,75 à 0,90, erreurs à 4 à 8).
- Mélange : moyenne ≈ 0,334 × 8 + 0,666 × 0,8 ≈ 3,2 ; écart-type ≈ 3,5 à 5 ; précision forget ≈ 0,666 × 0,85 ≈ 0,57. Recoupe l'estimation de l'équipe (`reports/t3-t3-sondes.md` : familles de golds compatibles aux scores observés : précision 0,49 à 0,74, loss moyenne 2,0 à 5,0, écart-type 3,4 à 6,7).
- Moyenner plusieurs golds (énoncé) : la moyenne des moyennes ne change pas, l'écart-type moyen par modèle reste celui d'un seul gold ; mais la distribution de prédiction moyenne `p_gold` est plus douce (température 2, moyenne de plusieurs distributions) : les KL ne seront pas minimisés par une prédiction « dure » sur la classe sœur.

## 4. Code d'une tâche semblable
- Cherché sur Hugging Face (23 jeux, 2 modèles) et GitHub (page de l'organisation) : aucun dépôt d'unlearning, aucun évaluateur, aucun script de création de jeu. Les pages CISPA décrivent des tâches différentes à chaque édition (« les tâches sont complètement différentes à chaque fois », Boenisch, rapport de Vienne) et ne publient pas de code.
- La métrique ressemble à celle du défi NeurIPS 2023 d'unlearning (comparaison à des modèles réentraînés) et à la « Towards Unbounded Machine Unlearning » citée dans l'énoncé (https://arxiv.org/abs/2302.09880) ; aucun code de l'organisateur n'a été retrouvé.

## Cherché sans trouver
- README, nom, générateur, effectifs ou script du jeu : introuvables (pas de fiche).
- Liste des 100 classes ; étiquettes vraies du forget set : introuvables (le serveur les garde).
- Gold, ou toute distribution de loss de test d'un ViT-B de zéro sur 13 000 images : aucun papier retrouvé.
- Code d'une édition précédente ou d'un évaluateur : rien de public.
- Le contenu des PDF de https://arxiv.org/abs/2112.13492 (trop lourd) et le tableau DeiT-B sur ImageNet-100 : non lus.
- Je n'ai contacté ni 34.147.229.55 ni aucun service payant.

## Ce que ça change pour la méthode
- Tenir la cible « gold à loss moyenne 3 environ, écart-type 4 environ, très bimodale » comme l'hypothèse de travail, et la confirmer par les sondes planifiées dans `reports/t3-t3-sondes.md` (la sonde `Z9_b0` tranche).
- Les classes sont fines et voisines : la « mauvaise classe » du gold est presque toujours la classe sœur visible (renard : autre figurine ; lanterne ou sablier : variante voisine). Utiliser la ressemblance visuelle (représentations d'`orig`) pour choisir la classe sœur des 9 classes oubliées.
- Les copies cachées ont été « durcies » le 18/11 : ne pas viser un réglage qui ne vaut que sur les images exactes.
