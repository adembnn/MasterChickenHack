# T3 Unlearning : recherche de la méthode manquante (origine de la métrique, méthodes « prédire le réentraîné »)

Date : 21 nov. 2026. Décision débloquée : quelle méthode implémenter en 12 h pour passer de 0,03 à plus de 0,2. Rien lancé, aucun service payant.
Étiquettes : **vérifié** = page arXiv, GitHub ou CISPA ouverte et lue pendant cette recherche (l'outil de lecture rend un résumé de la page, pas le PDF complet : les chiffres de figures ne sont pas lus) ; **supposé** = déduction ou souvenir non revu.
Vocabulaire (une fois) : **gold** = moyenne de modèles réentraînés sans les images à oublier ; **KLoM** = « KL des marges » : pour chaque image, on compare l'histogramme de la marge (logit de la bonne classe moins log-somme-exp des autres) sur 100 modèles réentraînés et sur 100 modèles « oubliés » ; **proxy de mémorisation** = mesure bon marché qui classe les images selon à quel point le modèle les a apprises par cœur ; **oracle matching** = fine-tuner le modèle pour qu'il produise les sorties qu'un modèle réentraîné aurait (prédites par un estimateur).

## (a) Décision recommandée (10 lignes)
1. Aucun papier ne définit cette métrique exacte (loss moyenne + écart-type, puis KL par image à T=2 vers la MOYENNE du gold). La plus proche est KLoM (Georgiev et al., « Attribute-to-Delete », ICLR 2025) et la métrique du concours NeurIPS 2023 : c'est une métrique maison des concepteurs (vérifié par absence : plusieurs recherches, rien).
2. La seule méthode publiée qui vise explicitement « la sortie qu'aurait le réentraîné » est le Datamodel Matching (arXiv 2410.23232) : estimer cette sortie par attribution de données, puis fine-tuner pour l'imiter, avec 5 à 400 fois plus d'images « retain » que d'images à oublier. Elle exige au moins 1 000 modèles réentraînés : impossible ici. On en garde le **cadre** (cible par image + ancre massive), pas l'estimateur.
3. **Méthode 1 à implémenter : oracle matching avec pseudo-gold fabriqué.** Cible par image = sortie d'un modèle qui n'a jamais vu l'image : (i) un « élève » fine-tuné depuis orig sur des images externes pseudo-étiquetées (le « holdout retraining » de Zhao & Triantafillou, proxy de mémorisation ρ = 0,62 à 0,67) ; (ii) option : sondes linéaires hors-pli sur des features d'un réseau préentraîné externe (si autorisé). Puis distiller orig vers cette cible sur les 3 500 images et leurs vues, avec 5 à 20 fois plus d'images externes comme ancre. Coûte 20 à 40 GPU-min par essai.
4. **Méthode 2 à implémenter en parallèle : classe voisine par exemple adversarial** (AMUN 2503.00917 et Cha et al. 2301.11578, retain-free pour Cha). La « mauvaise classe » de chaque image = classe vers laquelle orig bascule sous la plus petite perturbation ; sert de cible aux images instables et d'ancre de frontière. 5 à 10 GPU-min par essai. Sert à tester si la bonne mauvaise classe fait monter S_pred.
5. Pourquoi elles collent à CETTE métrique : S_pred est une KL par image vers une moyenne de modèles qui n'ont pas vu l'image ; il faut donc la distribution complète (bonne classe peu sûre + bonnes voisines), pas une loss haute. Les méthodes « monter la loss » (NPO, SalUn, SCRUB, gradient ascent) ne la fournissent pas : leur seul levier est S_loss.
6. Aucun papier ne valide qu'un proxy de mémorisation prédit la CLASSE erronée du réentraîné : ils prédisent combien l'image est mémorisée (l'écart de loss), pas où va la probabilité. Cette partie reste à mesurer sur le simulateur CIFAR-100 (N3).
7. Aucune édition précédente du championnat (Paris 2025, Vienne, Stockholm, Munich, Barcelone, Varsovie, finale) n'a, sur les pages publiques, de tâche d'unlearning ; aucun dépôt de solution publique trouvé. T3 Paris 2026 semble la première : pas de code à reprendre.
8. Le goulot reste le jeu d'images externes dans le domaine (questions à poser aux superviseurs : images externes autorisées, poids préentraînés autorisés ?). Sans elles, les méthodes 1 et 2 perdent leur ancre.
9. Coût total : 2 à 4 h de GPU pour deux balayages, 8 à 10 h-personne. Porte à respecter : accord argmax avec orig sur l'ancre > 95 %.
10. Chemin : `E:\EAC\reports\t3-recherche-methode-manquante.md`.

## (b) Tableau des méthodes
Légende « retain » : données retain nécessaires ? (non = seulement les images à oublier ; externe = images non étiquetées d'ailleurs). « Cible réentraîné » : la méthode cherche-t-elle à imiter la sortie du réentraîné ?
| Méthode | Principe en 2 lignes | Retain ? | Cible réentraîné ? | Code | Statut |
|---|---|---|---|---|---|
| Datamodel Matching / Oracle Matching (Georgiev et al., ICLR 2025) https://arxiv.org/abs/2410.23232 | Estime par régression de datamodels la sortie du réentraîné, puis fine-tune pour l'imiter. Adam, batch 512, lr {1e-3..1e-6}, 1 à 8 époques, ancre retain = 5 à 400 x taille du forget set | oui (échantillon retain) | **oui, c'est le but** | pas de dépôt trouvé (le dépôt MadryLab/unlearning répond 404) | vérifié (résumé + annexe C) ; impossible tel quel : 1 000 à 20 000 modèles réentraînés (TRAK : 300 checkpoints) |
| KLoM + Easy Data Unlearning Bench https://arxiv.org/abs/2602.16400 ; https://github.com/easydub/EasyDUB-code | Métrique « KL des histogrammes de marge » sur 100 réentraînés, CIFAR-10, 2 000 oracles fournis. Exemple fourni : P95 KLoM de pretrain = 9,26 sur forget, 0,32 sur retain | non (c'est un banc) | oui (c'est la référence) | oui (MIT) | vérifié ; auteurs Rinberg, Puigdemont, Pawelczyk, Cevher. Seul banc public où le gold est une moyenne de centaines de modèles |
| Holdout retraining (proxy de mémorisation) Zhao & Triantafillou https://arxiv.org/abs/2410.16516 | Fine-tune le modèle sur des données non vues ; la baisse de confiance des images d'entraînement mesure leur mémorisation. Spearman 0,67 (CIFAR-10) et 0,62 (CIFAR-100) avec Feldman-Zhang ; coût 3 à 16 % d'un réentraînement | données non vues (labellisées dans l'article) | partiel (rang de mémorisation, pas la classe) | non trouvé | vérifié (tableau 1) |
| Confiance moyenne par époque ; accuracy binaire (même article) | Proxies lus pendant l'entraînement : Spearman -0,80 / -0,91 (confiance) | non | partiel | non | vérifié ; inutilisables ici (il faut les journaux d'entraînement d'orig, qu'on n'a pas) |
| Courbure de la loss (Garg et al.) https://arxiv.org/abs/2307.05831 | Courbure de la loss autour de chaque image, moyennée sur l'entraînement ; corrèle avec Feldman-Zhang (cosinus élevé) ; un seul réseau | non | non (classe la mémorisation) | non trouvé | vérifié (résumé) ; coût 2 à 2,4 fois un réentraînement dans Zhao & Triantafillou (corrélation 0,69 à 0,70) ; version « modèle final seul » non vérifiée |
| AMUN (Ebrahimpour-Boroojeny) https://arxiv.org/abs/2503.00917 ; thèse 2512.06993 | Fine-tune sur les exemples adversariaux les plus proches de chaque image à oublier (diminue la confiance localement). Attaque d'inférence d'appartenance au niveau du hasard sur 10 % aléatoires de CIFAR-10 | à vérifier | oui (« similarité des prédictions au réentraîné » est le moteur annoncé) | pas de lien trouvé | vérifié (résumé) ; supposé : étiquette adversariale = classe voisine |
| TRW / TREW (même thèse ; ICML 2026 2506.20893) https://github.com/CrowdDynamicsLab/ICML_TREW | Cible = sortie d'orig sans la classe oubliée, inclinée vers les classes similaires | oui | oui, mais suppression de CLASSE | oui | vérifié (résumé) ; déjà lu par l'état de l'art du 21/11 |
| Cha et al. « Learning to Unlearn » (AAAI 2024) https://arxiv.org/abs/2301.11578 | Exemples adversariaux pour garder la frontière + importance des poids ; ne nécessite que le modèle et les instances à oublier | **non** | non | pas de lien trouvé | vérifié (résumé) |
| Bad Teacher (Chundawat et al., AAAI 2023) https://arxiv.org/abs/2205.08096 ; https://github.com/vikram2000b/bad-teaching-unlearning | KL vers le professeur compétent (orig) sur le retain et vers un professeur incompétent (réseau aléatoire) sur l'oubli | retain (non étiqueté suffit : supposé) | non (cible = bruit) | oui (ViT, ResNet18, CIFAR) | vérifié (résumé) ; mauvaise adéquation : pousse vers l'uniforme, ni S_loss ni S_pred |
| ViT-MUL (baselines sur ViT-B/16, 2403.09681) https://arxiv.org/html/2403.09681 | Compare fine-tune, CF-k, AdvNegGrad, UNSIR, SCRUB, ARU ; AdvNegGrad et SCRUB en tête, instables ; ARU : réinitialiser 10 % des poids pour ViT-B/16 | oui | non | non | vérifié ; bancs MUFAC/MUCAC, pas de l'oubli aléatoire |
| SCRUB, SalUn, NPO (références du sujet) | Monter la loss sur l'oubli (max-min, labels aléatoires, préférence négative) | oui | non | oui | non relu ici ; l'énoncé les cite comme références |
| ReGUn 2603.11210 ; LTD 2605.31317 ; SCAR 2404.12922 ; MU-Mis 2402.15109 ; gagnants Kaggle 2023 | voir `E:\EAC\reports\t3-unlearning-etat-de-lart.md` | | | | non relu ici |
| UnAct 2610.04426 https://github.com/abdulmuizz0903/UnAct | Sans gradient : atténue les unités de fin de réseau les plus actives sur l'oubli, jusqu'à 20 tours | **non** | non | oui | vérifié (résumé) ; suppression de CLASSES, ViT-B/16 « préliminaire » ; écarté |
| MUNKEY (2603.15033) | Efface une clé d'identification d'instance dans un transformeur à mémoire | non | non | non trouvé | vérifié (résumé) ; exige une architecture spéciale : écarté |
| Découverte « distillation mémorise moins » (Meta, 2601.15394) | Les élèves distillés mémorisent plus de 50 % de moins que le fine-tuning ; la distillation douce hérite 2,7 fois moins que la dure | non | indirect | non | vérifié (résumé) ; **LLM seulement** ; soutient seulement par analogie l'idée « élève distillé = pseudo-gold » (supposé) |

## 1. Origine exacte de la métrique (priorité 1)
- **Pas d'article source trouvé.** Requêtes (anglais et chinois) sur « loss distribution score », « prediction score », moyenne de modèles réentraînés, température 2, « matching retrained model predictions » : aucun résultat ne contient la combinaison. Les concepteurs (Maitri Shah, Jack Le, Vincent Hanke) n'ont pas de papier d'unlearning de ViT trouvé ; le seul travail de Hanke est sur les LLM (vu dans `labos-partenaires-attendus.md`, non revérifié).
- **Ascendants vérifiés de la métrique** :
  - KLoM (Georgiev et al. 2410.23232 ; banc 2602.16400) : compare, image par image, la distribution de la marge sur des centaines de réentraînés. Ici : le même esprit, mais en moyenne de distributions (T=2) pour la prédiction et en moments (moyenne, écart-type) pour la loss.
  - Concours NeurIPS 2023 : métrique « ressembler au réentraîné ».
  - Piste supposée : Martin Pawelczyk signe à la fois le banc KLoM (2602.16400, vérifié) et un papier avec Hanke (d'après le rapport labos, non revérifié). Les concepteurs ont donc probablement lu la lignée KLoM ; c'est un indice, pas une preuve.
- **Meilleurs scores et réglages dans la littérature de cette métrique** : sans objet (pas d'article). Chiffre le plus proche : KLoM P95 forget de pretrain = 9,26 contre 0,32 en retain sur CIFAR-10 (EasyDUB, vérifié) : l'écart au gold se concentre sur les images à oublier.
- Le jeu `SprintML/unlearning` sur Hugging Face contient `forget_images.npz`, `model.pt`, `requirements.txt`, `submission.py`, `task_template.py`, sans fiche (vérifié via l'API ; pas de classes ni de source indiquées). Il ne contient aucune image retain.

## 2. Méthodes qui prédisent ce que ferait un réentraîné (priorité 2)
Ce que dit la littérature lue :
- **Estimer la sortie du réentraîné** : (1) par régression de datamodels (coûteux : 1 000 à 20 000 modèles) ; (2) par TRAK (300 checkpoints) ; (3) par réentraînement partiel (coût de 60 % d'un réentraînement pour égaler l'oracle matching à moins de 5 %, vérifié). Toutes demandent des réentraînements. Aucune méthode publiée ne le fait avec un seul modèle.
- **Idées de contournement (supposées, à tester)** :
  1. **Élève sans contact** : fine-tuner (ou réinitialiser partiellement) un modèle à partir d'orig sur des images externes seulement, par distillation douce ; ses sorties sur les images à oublier sont celles d'un modèle qui n'a jamais vu ces images. Soutien indirect : la distillation douce hérite 2,7 fois moins de mémorisation que la dure (LLM, vérifié). Risque : le domaine externe.
  2. **Fine-tune sur données externes jusqu'à ce que la confiance de non-membre soit atteinte** = holdout retraining : c'est à la fois un proxy de mémorisation (ρ 0,62 à 0,67) et un candidat de pseudo-gold.
  3. **Sondes hors-pli sur features externes** : un réseau préentraîné externe + régression logistique à 5 plis sur les pseudo-étiquettes d'orig ; la prédiction hors-pli est un modèle indépendant. Condition : poids préentraînés autorisés (non précisé par l'énoncé, à demander).
  4. **Classe voisine** : classe de l'exemple adversarial le plus proche (AMUN, Cha, Boundary Unlearning) ; similarité de classe (TREW : β=10).

Règle pour les méthodes « monter la loss » : S_loss pénalise le dépassement (|écart|), donc NPO, SCRUB, SalUn et le gradient ascent ne gagnent qu'à la bonne hauteur, et ne produisent pas la distribution par image : classées en dessous.

## 3. Éditions précédentes (priorité 3)
Pages CISPA ouvertes (toutes vérifiées) ; aucune ne mentionne l'unlearning :
| Édition | Tâches nommées sur la page |
|---|---|
| Paris 2025 (22-23 nov. 2025) | perturbation d'image (exemples adversariaux), attribution d'images générées ; la 3e non décrite |
| Vienne (déc. 2025) | inférence de jeu de données, classification de texte, retrait de filigrane |
| Stockholm (févr. 2026) | inversion de modèle, mauvaise classification, attribution de modèle |
| Munich (mars 2026) | non nommées ; inférence de jeu de données (conférence de Papernot) |
| Barcelone (avr. 2026) | inférence de propriétés avec un modèle fantôme ; les 2 autres non décrites |
| Varsovie (mai 2026) | influence des données d'entraînement, inversion de modèle, interprétabilité |
| Finale (2-3 juil. 2026) | filigrane de texte, membre/généré, reconstruction en apprentissage collaboratif |
- Dépôts publics : un seul trouvé, `https://github.com/floriandgg/cispa` (équipe Loki, Paris 2025, « cascade forensique hybride »), non ouvert, hors unlearning. Les dépôts de la finale cités dans le rapport labos portent sur le filigrane de texte. **Aucune solution publique d'unlearning de ce hackathon.** Les tâches changent à chaque fois (Boenisch, cité par un communiqué : « les tâches sont complètement différentes à chaque fois »).
- La tâche de Varsovie (« influence des données d'entraînement ») est la plus proche en thème de l'estimation de sortie par attribution ; son contenu n'est pas public.

## 4. Estimer la sortie d'un modèle qui n'a pas vu l'image (priorité 4)
| Mesure | Ce qu'elle prédit | Entrée nécessaire | Chiffre | Prédit la classe erronée ? |
|---|---|---|---|---|
| Feldman & Zhang (référence) | mémorisation = écart de précision en laissant l'exemple de côté | milliers de modèles réentraînés | référence | non (scalaire) |
| Holdout retraining | rang de mémorisation | modèle final + données non vues | ρ 0,67 / 0,62 | non (scalaire) |
| Courbure de la loss | rang de mémorisation (images longue traîne, mal étiquetées) | un modèle ; moyenne sur l'entraînement | ρ 0,69 / 0,70 (Zhao & Triantafillou) | non |
| Confiance, entropie, accuracy par époque | rang de mémorisation | journaux d'entraînement | ρ jusqu'à -0,91 | non ; inutilisable (pas de journaux) |
| Sondes hors-pli (idée) | distribution complète hors échantillon | features externes + pseudo-étiquettes | non mesuré | oui, en principe (supposé) |
| Classe adversariale / similarité de classes (AMUN, TREW) | classe voisine | modèle seul / vecteurs de classe | TREW : CMIA 82,1 (par classe) contre 65,6 (par image) d'après l'état de l'art du 21/11 | oui pour la classe, pas pour la hauteur |
Constat : trois familles de mesures donnent « à quel point », une seule donne « laquelle ». Aucune publication lue ne teste laquelle prédit la mauvaise classe choisie par un réentraîné pour une image isolée : c'est l'expérience à faire sur CIFAR-100 (N3).

## 5. Plan 12 h pour les deux méthodes
- H0-1 : test du domaine (classes = ImageNet ? CIFAR ?) et questions aux superviseurs (voir (c)).
- H1-4 : méthode 2 (adversarial, minimale : cible = orig tempéré + classe voisine, ancre = vues) pour mesurer S_pred seul. 5 à 10 GPU-min par essai.
- H1-6 en parallèle : méthode 1 (élève sans contact ou holdout retraining ; distillation vers la cible ; ancre externe 5 à 20 fois plus grande que 3 500). Départ : AdamW lr 2e-5 (grille 1e-5, 5e-5), batch 128 + 128, 6 à 10 époques, blocs 8 à 11, `ls1/ls2.gamma` gelés. Les valeurs sont celles du Datamodel Matching (lr 1e-6 à 1e-3, 1 à 8 époques, multiplicateur 5 à 400) restreintes aux ViT ; supposé.
- H6-12 : fusion des cibles (moyenne de pseudo-gold et de la classe voisine), balayage du taux de mélange, 3 sondes publiques au plus (quota d'un envoi par 10 minutes, décision d'équipe).
- Vérification locale obligatoire avant tout envoi : strict=True, float32, taille < 450 Mo, accord argmax avec orig sur l'ancre > 95 % (porte 70 %).

## (c) Ce que j'ai cherché sans trouver
- Un article ou un benchmark qui contient la métrique exacte (loss moyenne + écart-type, KL T=2 vers la moyenne du gold) : non trouvé.
- Un papier de Maitri Shah, Jack Le ou Vincent Hanke sur l'unlearning de ViT : non trouvé (recherche par noms sans résultat ; aucun auteur correspondant).
- Une édition précédente avec tâche d'unlearning, ou un compte rendu de solution publique : non trouvé. Les pages CISPA des 7 éditions ne mentionnent pas l'unlearning ; les pages de Munich et de Barcelone ne décrivent que 1 tâche sur 3.
- Le code de Datamodel Matching : non trouvé (`github.com/MadryLab/unlearning` renvoie 404 ; il peut exister sous un autre nom).
- Le code de AMUN, de Cha et al., de Zhao & Triantafillou : non trouvé dans les pages ouvertes.
- Une méthode publiée qui estime la sortie du réentraîné avec un seul modèle et sans réentraînement : non trouvée.
- Une étude qui montre quel proxy prédit la classe erronée du réentraîné : non trouvée.
- Les solutions Kaggle 2023 : écrits des 1er, 5e, 9e, 10e déjà lus par le rapport état de l'art ; non relus ici. Un 8e rang (https://junhyunb.github.io/en/projects/neurips2023-unlearning/ : cycles courts de fine-tune, cosinus, bruit gaussien, logits de l'oubli plus proches du réentraîné) a été vu dans la recherche et n'a pas été ouvert.
- Non ouvert : Semantic Scholar / OpenAlex (limites de requêtes déjà signalées par le rapport état de l'art) ; recherches en chinois : un seul passage, sans résultat nouveau.

## Questions (défaut si pas de réponse)
1. Images externes (ImageNet, COCO, autres) et poids préentraînés (DINOv2 ou ImageNet) autorisés ? a. demander aux superviseurs maintenant (défaut) / b. s'en passer : seulement des images mélangées à partir des 3 500.
2. Qui lance quoi ? a. un agent par méthode (défaut) / b. une seule méthode, l'autre en réserve.
