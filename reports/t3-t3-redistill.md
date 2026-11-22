# T3 · t3-redistill : réinitialiser puis redistiller sur des images jamais vues

Session t3-redistill d'Adem, 21/11/2026, 21 h 10 à 23 h 45. Machines `h200e` (injoignable depuis 22 h 28) puis `h200c`, dossier `~/eac/t3/t3-redistill/`. Code : `tracks/t3-unlearning/t3-redistill/`.

**Décision que ce rapport débloque :** faut-il soumettre un modèle « réinitialisé puis redistillé » (non), et faut-il brancher sa sortie hors pli comme sélecteur d'images et de classe fausse dans la recette `PG` de t3-methodes (à trancher par un envoi, voir la dernière section).

## Idée testée
Effacer la tête (couche de classification) et les k derniers blocs du ViT, puis les réapprendre en imitant les sorties d'`orig` (divergence KL à température 2 et 1) sur des images que le modèle n'a jamais vues telles quelles. Attendu : ce qui est général revient, ce qui est mémorisé ne revient pas, et les erreurs sur les 3 500 images apparaissent d'elles-mêmes avec leur classe et leur confiance.

Mesures : conditions du serveur (eval, fp16), étiquette = classe prédite par `orig`. « Images écartées » = images à oublier dont ni l'image ni aucun mélange n'a servi à la redistillation (698 images, la réserve `c` de t3-methodes, ou un pli sur cinq).

## Étape 1 : réservoirs d'images jamais vues
| réservoir | images | classes couvertes (≥ 1 image / ≥ 10) | confiance d'`orig` (max-prob T=1) | part à max-prob > 0,9 |
|---|---|---|---|---|
| (c) mélanges où aucune image ne pèse plus de la moitié (mixup 2-3 images, coupe moitié-moitié, mosaïque 2×2) | 6 000 | 100 / 94 | 0,748 | 0,37 |
| (b) COCO val2017 | 5 000 | 98 / 86 | 0,734 | 0,34 |
| (a) SDXL image-vers-image force 0,70 | 900 | 74 / 45 | 0,919 | 0,77 |
| (a) SDXL force 0,85 | 900 | 89 / 44 | 0,872 | 0,64 |
| (a) IP-Adapter | 900 | 85 / 43 | 0,877 | 0,65 |
| (a) texte-vers-image SDXL ; FLUX | 900 | 91 / 40 ; 96 / 37 | 0,817 | 0,51 |

## Étapes 2 et 3 : grille, usage (i) « le modèle redistillé est le candidat »
Critère : loss moyenne ≥ 1,5 et écart-type ≥ 3 sur les images à oublier, accord ≥ 0,9 avec `orig` sur des images non vues.

| variante | pas | exactes écartées : loss moy / écart-type / classe changée | images non vues : accord avec `orig` |
|---|---|---|---|
| tête seule, mélanges | 1000 | 0,00 / 0,00 / 0,0 % | mélanges 0,96 |
| tête + 2 blocs, mélanges | 1000 | 0,14 / 0,44 / 4,6 % | mélanges 0,65 |
| tête + 2 blocs, images générées (3 600) | 1000 | 0,13 / 0,54 / 3,9 % | générées 0,70 : 0,87 |
| tête + 2 blocs, COCO | 1000 | 0,55 / 1,16 / 15,8 % | générées 0,70 : 0,65 |
| tête + 4 blocs, mélanges | 1000 / 2000 / 3000 | 1,23 / 1,47 / 40,8 % ; 0,84 / 1,24 / 29,5 % ; 0,73 / 1,16 / 24,8 % | mélanges 0,50 ; 0,57 ; 0,59 |
| bruit gaussien dans les 12 blocs, σ = 0,6 × écart-type du tenseur, sans redistillation | 0 | 0,06 / 0,33 / 2,0 % | générées 0,70 : 0,80 |
| même bruit puis redistillation de tous les blocs, mélanges | 500 | 0,00 / 0,02 / 0,0 % | générées 0,70 : 0,89 |

Tête + 6 blocs : arrêté à 800 pas pour tenir la limite de 40 Go (accord du lot 0,45, encore plus loin de la fidélité que 4 blocs).

**Verdict : aucune variante n'approche le critère.** Partout, le modèle se trompe moins sur les images à oublier que sur les images non vues, et ses erreurs sont hésitantes (écart-type ≤ 1,5, max-prob T2 ≈ 0,45), alors que le gold estimé fait des erreurs confiantes.

**Pourquoi (mesuré) :**
1. Une tête neuve, entraînée sans aucune image exacte, classe juste 100 % des 3 500 images : les caractéristiques des blocs gardés contiennent déjà la réponse mémorisée.
2. Un bruit sur tous les poids abîme d'abord les images non vues (20 % de désaccord) et presque pas les images apprises (2 %) ; 500 pas de redistillation ramènent les images apprises à 0 % d'erreur avant les images non vues (11 %). L'entraînement d'`orig` a donné aux images apprises une marge (écart entre le logit de la classe prédite et les autres) plus grande qu'aux images non vues. Une perturbation qui ne vise pas ces images, suivie d'un réapprentissage sur d'autres images, ne peut donc pas les faire tomber en premier.
3. La fuite par les mélanges est réelle à 4 blocs : 12,7 % d'erreurs sur les images sources des mélanges contre 24,8 % sur les écartées.

Aucun fichier n'est écrit dans `candidates/` pour cet usage.

## Étape 4 : usage (ii), cible par image hors pli
Cinq plis stratifiés par classe ; pour chaque pli, tête + 4 blocs réinitialisés et redistillés 1 500 pas sur des mélanges des quatre autres plis ; chaque image est prédite par le modèle qui ne l'a vue sous aucune forme.

`h200:~/eac/t3/t3-redistill/targets/redistill.npy` : float32 [3500, 100], probabilités à T=1, ordre de `forget_images.npz`, sha256 `bc7ddfb0bc02195d`. À côté : `redistill_logits.npy`, les mêmes sur quatre copies légères, et `redistill_x_err.npy` (sélecteur combiné, voir plus bas).

| mesure | valeur |
|---|---|
| précision contre la classe d'`orig` | 0,633 (36,7 % d'images ratées) |
| loss moyenne / écart-type | 1,05 / 1,32 ; loss des images ratées 2,38 |
| max-prob T=1 / T=2 | 0,728 / 0,457 |
| classe fausse = 2e choix d'`orig` | 51 % |
| erreurs vers une des 55 classes absentes du jeu à oublier | 17,5 % (une sonde entraînée sur les 45 classes ne peut pas en produire) |
| même verdict juste/faux sur les copies légères | JPEG 0,92 ; rotation 0,76 ; recadrage 0,78 ; légère 0,80 |
| taux d'erreur par classe | 0,13 à 0,71, médiane 0,33 ; corrélation avec l'effectif de la classe 0,27 |

**Comparaison avec les sélecteurs existants.** Corrélation de rang avec le score d'instabilité de t3-methodes : 0,26 ; dans les 30 % du haut, 432 images communes sur 1 050 (315 attendues au hasard). Sur les images communes, même classe fausse dans 55 % des cas.

**Contrôle croisé sans gold** (`code/assemble.py`, `code/combo.py`) : pour 900 images, t3-nonmembres a une régénération (image non vue issue de la même image). Question : le score du sélecteur sur l'image exacte prédit-il que la régénération trompe `orig` ? AUC = probabilité qu'une image dont la régénération trompe `orig` reçoive un score plus haut qu'une image dont la régénération ne le trompe pas (0,5 = hasard).

| sélecteur | SDXL force 0,70 : AUC [IC 95 %] | SDXL force 0,85 | IP-Adapter | même classe fausse que la régénération (0,70 / 0,85 / IP) |
|---|---|---|---|---|
| instabilité sous vues dégradées (t3-methodes, cible du 0,1962) | 0,639 [0,595 ; 0,686] | 0,606 | 0,550 | 0,39 / 0,36 / 0,30 |
| redistill hors pli | 0,648 [0,607 ; 0,687] | 0,580 | 0,553 | 0,44 / 0,39 / 0,39 |
| moyenne des deux rangs | **0,673** [0,629 ; 0,720] | **0,611** | **0,561** | (classe = redistill) |
| `oofA_b8` (sonde linéaire hors pli, t3-poids) | 0,531 | 0,507 | 0,493 | 0,19 / 0,20 / 0,21 |
| `S30_oof` | 0,591 | 0,545 | 0,516 | 0,14 / 0,20 / 0,24 |
| bruit sur les poids σ 0,9 (sans entraînement) | 0,596 | 0,547 | 0,538 | 0,40 / 0,35 / 0,43 (n ≈ 20) |
| marge d'`orig` sur l'image exacte (témoin) | 0,584 | 0,579 | 0,561 | – |

Lecture : redistill et le sélecteur de t3-methodes sont au même niveau et se recoupent peu, leur moyenne est la meilleure sur les trois jeux ; les intervalles se chevauchent, donc ce contrôle classe la sonde linéaire derrière mais ne départage pas les trois premiers. Limite : la régénération n'est pas le gold, et le sélecteur de t3-methodes est construit sur des vues dégradées de l'image, parent proche de ce contrôle.

## Candidats de l'usage (ii)
Recette exacte du 0,1962 (`train.py --steps 1500 --wce 1 --lam 3 --anchor mix`, 4 blocs, 30 % d'images, loss 8-14), seule la cible change. Cibles faites par `mk_tgt.py` de t3-methodes avec `--rank ext:` et `--wsrc ext:`. Fichiers : `h200:~/eac/t3/t3-redistill/candidates/` (copie `h200e:~/eac/t3/t3-redistill/tm/candidates/`), tous valides (`validate_t3.py`).

| fichier (sha256) | ce qui change | exactes : loss moy / écart-type / classe changée | copies légères | bascules atteintes (loss) | mélanges : accord / KL | réserve `c` (entraîné sur a+b) : rétention exactes / copies légères ; flou fort précision / KL |
|---|---|---|---|---|---|---|
| `PG_T85_f30_l814c` (référence, 0,1962 réel) | – | 2,564 / 4,293 / 29,6 % | 2,105 / 3,691 | 84,2 % (8,54) | 0,802 / 0,152 | 0,795 / 0,752 ; 0,695 / 0,435 |
| `PG_RDc30_l814` (08ebdfd443da7ab6) | classe fausse = moitié redistill, moitié vues dégradées ; mêmes images | 2,689 / 4,457 / 29,7 % | 2,181 / 3,790 | 89,9 % (8,96) | 0,801 / 0,147 | 0,823 / 0,786 ; 0,731 / 0,324 |
| `PG_RDx30_l814` (373e87d557be1c09) | images = moyenne des deux rangs (738 communes sur 1 050) + classe redistill | 2,800 / 4,562 / 29,7 % | 2,282 / 3,897 | 91,0 % (9,33) | 0,802 / 0,148 | 0,815 / 0,787 ; 0,741 / 0,310 |
| `PG_RD30_l814` (6369ebcbf383a522) | images et classe redistill (432 communes) | 2,737 / 4,485 / 29,8 % | 2,244 / 3,849 | 93,3 % (9,11) | 0,803 / 0,153 | non mesurée |

Lecture : à réglages identiques, les classes fausses de redistill sont mieux adoptées (90 à 93 % des bascules atteintes contre 84 %), la loss monte (moyenne + écart-type sur les exactes 7,15 à 7,36 contre 6,86) et les images non entraînées sont moins dérangées (rétention de réserve 0,815 à 0,823 contre 0,795). L'effet sur le score de prédiction ne se mesure pas sans gold : c'est un envoi qui tranche. Ordre conseillé : `PG_RDx30_l814`, puis `PG_RDc30_l814` (lit l'effet « classe fausse » seul). Prédiction demandée au vérificateur T3 sur l'issue #5 ; aucun envoi fait par cette session.

## Candidats à 50 % et 60 % (demande de l'orchestrateur, 22 h 28 et 22 h 50)
La part d'images basculées paie jusqu'à 60 % (réel : 30 % 0,1962 ; 50 % 0,2779 ; 60 % 0,3009). Mêmes recette et code que ces envois, entraînés sur `h200c` (`h200e` injoignable depuis 22 h 28). Classe fausse = 100 % redistill hors pli (`mk_tgt.py --wsrc ext:redistill.npy --wext 1.0`). Fichiers : `h200:~/eac/t3/t3-methodes/candidates/`, lignes dans `diag/row_PG_RD*0_l814c.md`, tous valides.

Accord de classe fausse entre redistill et la cible de t3-methodes sur les images basculées : 47,1 % à 50 % (1 750 images), 47,3 % à 60 % (2 100 images).

| fichier (sha256) | ce qui change | exactes : loss moy / écart-type / classe changée | copies légères | bascules atteintes (loss) | max-prob T2 | mélanges : accord / KL |
|---|---|---|---|---|---|---|
| `PG_T85_f60_l814c` (référence, 0,3009 réel) | – | 4,221 / 4,431 / 57,2 % | 3,649 / 4,021 | 75,6 % (7,03) | 0,674 | 0,736 / 0,263 |
| `PG_RDc60_l814c` (ab79c8f7d538b305) | mêmes images, classe redistill | 4,998 / 4,891 / 58,4 % | 4,111 / 4,364 | 89,0 % (8,33) | 0,841 | 0,721 / 0,265 |
| `PG_RDx60_l814c` (554449dd0f5d066e) | images combinées (1 743 communes) + classe redistill | 5,204 / 4,981 / 58,7 % | 4,275 / 4,448 | 91,1 % (8,67) | 0,854 | 0,725 / 0,264 |
| `PG_T85_f50_l814c` (référence, 0,2779 réel) | – | 3,700 / 4,466 / 48,1 % | 3,124 / 3,970 | 78,6 % (7,39) | 0,732 | 0,752 / 0,230 |
| `PG_RDc50_l814c` (7289781f93130956) | mêmes images, classe redistill | 4,334 / 4,951 / 48,8 % | 3,534 / 4,328 | 89,7 % (8,66) | 0,863 | 0,741 / 0,227 |
| `PG_RDh50_l814c` (74a846153b2cd7f0) | mêmes images, classe moitié redistill moitié vues dégradées | 3,957 / 4,673 / 48,3 % | 3,265 / 4,095 | 87,0 % (7,91) | 0,807 | 0,746 / 0,225 |
| `PG_RDx50_l814c` (526dd210a84676ab) | images combinées (1 367 communes) + classe redistill | 4,554 / 5,072 / 49,1 % | 3,719 / 4,449 | 91,8 % (9,11) | 0,874 | 0,756 / 0,228 |

Rétention sur la réserve `c` à 60 % (trois versions entraînées sur a+b, même code, même graine) :

| version | oubli a+b : loss moy / écart-type | bascules atteintes | réserve exactes : accord / KL / rétention | copies légères : rétention | flou fort : précision (orig 0,867) / KL |
|---|---|---|---|---|---|
| `PG_T85_f60_l814cab` (référence) | 5,067 / 4,856 | 82,1 % | 0,848 / 0,607 / 0,528 | 0,490 | 0,450 / 1,143 |
| `PG_RDc60_l814cab` | 5,685 / 5,211 | 92,6 % | 0,897 / 0,358 / 0,660 | 0,620 | 0,636 / 0,516 |
| `PG_RDx60_l814cab` | 5,829 / 5,262 | 93,5 % | 0,895 / 0,350 / 0,663 | 0,626 | 0,638 / 0,495 |

Lecture corrigée après la relecture du vérificateur : ces fichiers changent plusieurs facteurs à la fois. La cible `RDc` est deux fois plus pointue que celle de la référence sur les bascules (max-prob à T=2 de la cible : 0,672 contre 0,320), car sa répartition hors classe vient de la sortie hors pli à T=1. Le gain de loss et de rétention vient surtout de cette forme, pas de la classe.

### Test à un seul facteur (classe redistill, même forme de cible que le 0,3009)
Logits hors pli divisés par 2,4 avant de former la cible (`redistill_tau2.4.npy`) : max-prob des bascules à T=2 de la cible 0,315 (référence 0,320). Profil des modèles (`profil.py`) :

| fichier (sha256) | réglage | exactes : loss moy / écart-type | bascules : max-prob T2 / atteintes | non basculées : proba T2 | rétention de réserve (version a+b) |
|---|---|---|---|---|---|
| `PG_T85_f60_l814c` (0,3009) | – | 4,221 / 4,431 | 0,531 / 75,6 % | 0,889 | 0,528 |
| `PG_RDt24_60_l915c` (47e050dfd9e71cf2) | ÷ 2,4, loss visée 9-15,5 | 4,054 / 4,453 | 0,531 / 73,7 % | 0,883 | 0,533 |
| `PG_RDt21_60_l814c` (277ced8db1f09091) | ÷ 2,1, loss 8-14 | 4,029 / 4,284 | 0,573 / 76,8 % | 0,893 | 0,566 |
| `PG_RDt24_60_l814c` (c1ba0a3f8a0f1a82) | ÷ 2,4, loss 8-14 | 3,794 / 4,111 | 0,516 / 73,7 % | 0,878 | 0,541 |
| `PG_RDc60_l814c` (cible pointue) | T=1 | 4,998 / 4,891 | 0,778 / 89,0 % | 0,936 | 0,660 |

- `PG_RDt24_60_l915c` est le test propre : tout est égal à la référence sauf la classe fausse (différente pour 53 % des bascules) et une loss moyenne 4 % plus basse. C'est son score réel qui dira ce que vaut la classe redistill.
- À forme égale, la classe redistill ne donne ni plus de loss ni plus de rétention (0,533 contre 0,528) ; seul le flou fort s'améliore (précision 0,53 contre 0,45).
- Constat à part : une cible pointue abîme beaucoup moins la réserve qu'une cible étalée (0,66 contre 0,53) et est mieux atteinte ; le prix est le risque sur le score de prédiction (`PGho_f60` = 0,2701 avec ce profil, selon le vérificateur).

### Suite : prédiction du vérificateur et palier 70 %
- `PG_RDt24_60_l915c` : conforme, prédiction 0,27 à 0,325 (central 0,295) contre 0,3009. Lecture fixée avant envoi : ≥ 0,307 la classe redistill est mieux alignée sur le gold ; 0,287 à 0,307 équivalentes ; ≤ 0,287 la classe votée est meilleure.
- Palier 70 % préparé (référence `PG_T85_f70_l814c` = 0,3159, loss 4,678 / 4,302) : `PG_RDt24_70_l1016c` (1203d017dba206e1) loss 4,674 / 4,546, bascules 0,516 / 71,1 % ; `PG_RDt24_70_l915c` (b62a007211827055) loss 4,475 / 4,305. À n'envoyer que si le test à 60 % est favorable.

## Résultats réels et suite (22/11, 0 h à 1 h 15)
Scores réels des fichiers de cette session (envoyés par l'orchestrateur) : `PG_RDt24_60_l915c` **0,3273** (référence `f60` 0,3009) ; `PG_RDt24_70_l915c` 0,3502 ; `PG_RDt24_70_l1016c` **0,3653**. La classe fausse redistill (hors pli, étalée par ÷ 2,4) est mieux alignée sur le gold que la classe votée par les vues dégradées, et monter la loss paie de nouveau avec elle.

Fichiers déposés ensuite dans `h200:~/eac/t3/t3-methodes/candidates/` (base `PG_RDt24_70_l1016c` : 4,674 / 4,546, bascules atteintes 71,1 %) :

| fichier (sha256) | facteur changé | exactes : loss moy / écart-type | bascules atteintes |
|---|---|---|---|
| `PG_RDt24_70_l1117c` (9294b9014b5922ee) | loss visée 11-17,5 | 4,843 / 4,783 | 71,2 % |
| `PG_RDt24_70_l1218c` (fafd3330b7c4de46) | loss visée 12-18,5 | 5,005 / 4,995 | 71,0 % |
| `PG_RDt24_80_l1016c` (31676678ca5cf3f0) | 80 % d'images | 5,094 / 4,407 | 68,0 % |
| `PG_RDt24_60_l1016c` (6aca20d4f61e7e2a) | 60 % d'images | 4,228 / 4,715 | 73,9 % |
| `PG_RDt24_80_l915c` (ecdea9ac612e5d08) | 80 %, loss 9-15,5 (base `70_l915c`) | 4,881 / 4,152 | 67,8 % |
| `PG_RDm3t24_70_l1016c` (fc9fbd54f77bc6c9) | classe = moyenne de 3 redistillations à 4 blocs | 4,663 / 4,542 | 70,8 % |
| `PG_RDm3t24_80_l1016c` (77c077a1c2e3387d) | idem à 80 % (base `80_l1016c`) | 5,087 / 4,391 | 67,2 % |
| `PG_RD6m2t24_70_l1117c` (f5ec5b05f2a27aef) | classe = 2 redistillations à 6 blocs, loss visée 11-17,5 | 4,574 / 4,569 | 67,2 % |
| `PG_RD6m2t24_70_l1218c` (95b7fbdcfba315f9) | idem, 12-18,5 | 4,710 / 4,759 | 67,1 % |
| `PG_RD6m2t24_70_l1016c` (c50308eba4479ebc) | idem, 10-16,5 | 4,422 / 4,364 | 67,2 % |
| `PG_RDm3r_70_l1016c` (f86ec4dec0820384) | images par rang redistill moyen + classe moyenne (base `RDm3t24_70_l1016c`) | 4,865 / 4,578 | 74,7 % |

Mesures entre redistillations (`oofavg.py`) :
- 4 blocs, 3 graines (plis différents) : précision hors pli 0,633 / 0,638 / 0,636 ; deux graines donnent la même classe prédite sur 81 % des images ; la classe fausse de la moyenne est celle de la graine 0 pour 85,7 % des images.
- 6 blocs, 2 graines : précision hors pli 0,440 / 0,448 (fourchette estimée du gold : 0,40 à 0,55) ; même classe prédite sur 73 % des images ; classe fausse = celle de la moyenne à 4 blocs pour 72 % des images.

Recette d'une cible : `mk_tgt.py NOM --pool 0.4:crop,0.2:mask,0.4:color,0.6:blur,0.4:noise --f 0.70 --llo 10 --lhi 16.5 --rank err --wsrc ext:<...>/targets/<source>.npy --wext 1.0`, avec `<source>` = `redistill_tau2.4` (une graine), `redistill_k4m3_tau2.4` (3 graines) ou `redistill_k6m2_tau2.4` (6 blocs). Sauvegarde : `E:\EAC-lab\sauvegarde-t3\`.

### Fichiers de 1 h 15 à 1 h 35 (demandes de l'orchestrateur et du vérificateur)
Réels reçus entre-temps : `PG_RDt24_80_l915c` 0,3638. Base inchangée : `PG_RDt24_70_l1016c` (0,3653 ; 4,674 / 4,546).

| fichier (sha256) | facteur changé | m / s (exactes) | bascules atteintes |
|---|---|---|---|
| `PG_RDt24n_70_l1016c` (ee177b51b98d95d0) | images non basculées : masse hors classe répartie selon redistill, loss inchangée (`nonflip.py`) | 4,689 / 4,555 | 71,3 % |
| `PG_RDt24r_70_l915c` (8a1c99b0b8c901e7) | images choisies par redistill seul, loss visée 9-15,5 | 4,674 / 4,359 | 74,1 % |
| `PG_RDt24r_70_l1016c` (1b0e4b9bfc6af8c2) | idem, loss visée 10-16,5 | 4,852 / 4,574 | 74,1 % |
| `PG_RD6m4t24_70_l1117c` (6b5b99edf8207086) | classe = 4 redistillations à 6 blocs | 4,561 / 4,559 | 67,8 % |
| `PG_RD8m2t24_70_l1117c` (e600698c245b6015) | classe = 2 redistillations à 8 blocs | 4,451 / 4,499 | 65,5 % |
| `PG_RDt24_80_l1214c` (733d15bad3973b8e) | 80 %, plage de loss 12-14 | 5,061 / 4,312 | 67,5 % |
| `PG_RDt24_80_l1113c` (404033ba2a960e50) | 80 %, plage de loss 11-13 | 4,843 / 4,063 | 67,7 % |

- Profondeur : précision hors pli 0,64 (4 blocs), 0,44-0,45 (6 blocs, 4 graines), 0,41 (8 blocs) ; accord de classe prédite entre deux graines 81 %, 73 %, 71 %. Plus la réinitialisation est profonde, moins la classe fausse est adoptée par le modèle.
- Resserrer la plage de loss ne découple pas la moyenne et l'écart-type : l'écart-type vient de l'écart entre images basculées et non basculées.
- Prédictions du vérificateur (base 0,3653) : `80_l1016c` 0,366 / 0,383 / 0,383 ; `RDm3t24_80_l1016c` 0,367 / 0,384 / 0,384 ; `70_l1117c` 0,360 / 0,367 / 0,372 ; `70_l1218c` 0,345 / 0,365 / 0,370.

### Nouvelle base à 0,3888 et croisements (1 h 50 à 2 h 20)
Réels : `PG_RDt24r_70_l1016c` **0,3888** (images choisies par le rang redistill moyen, + 0,0235 sur 0,3653) ; `PG_RDm3t24_80_l1016c` 0,3713 ; `PG_RDt24_70_l1218c` 0,3682 (l'axe loss plafonne à 70 %). Base : `PG_RDt24r_70_l1016c` (4,852 / 4,574 ; bascules atteintes 74,1 %).

| fichier (sha256) | facteur changé | m / s | bascules atteintes |
|---|---|---|---|
| `PG_RDt24rn_70_l1016c` (37d499648693c7e1) | images non basculées suivant redistill | 4,840 / 4,566 | 73,9 % |
| `PG_RDt24r_80_l1016c` (60b9dcdb15b5580a) | 80 % d'images | 5,243 / 4,360 | 69,2 % |
| `PG_RDt24r_60_l1016c` (9ac9a8fc67a27bcb) | 60 % d'images | 4,398 / 4,751 | 77,3 % |
| `PG_RDt24r_70_l1117c` (265e32b106534ce2) | loss visée 11-17,5 | 5,008 / 4,774 | 74,1 % |
| `PG_RDt24rc_70_l1117c` (feb35054a4654f64) | rang de consensus 4 + 6 + 8 blocs, loss recalée | 4,834 / 4,679 | 72,2 % |
| `PG_RDt24rc_70_l1016c` (a486570b485fc7b8) | idem sans recalage | 4,684 / 4,486 | 72,2 % |
| `PG_RDt24r6_70_l1218c` (e492e31ec3db61d8) | rang 6 blocs seul, loss recalée | 4,771 / 4,706 | 71,3 % |
| `PG_RDt24r6_70_l1016c` (fe32dbfdfc186ac4) | idem sans recalage | 4,535 / 4,376 | 71,5 % |

Les rangs à 4, 6 et 8 blocs sont corrélés à 0,77-0,89 ; le consensus ne déplace que 8 % des images basculées, et les images désignées par les blocs profonds sont moins bien atteintes. Sources : rang `redistill_k4m3.npy`, consensus `redistill_rang_c468.npy`.

### Bruit de graine, partitions, copies (2 h 20 à 3 h 10)
Réels : `PG_RDt24r_70_l1016c` rejoué avec la graine d'entraînement 1 : **0,3942** (bruit de graine ≈ 0,005) ; `r_60` 0,3648 ; `r_80` 0,3657 (le pic de la part est à 70 %) ; `RDt24n` 0,3663 et `RD6m4` 0,3654 (non basculées et 6 blocs : sans effet).

- **Partitions.** Chaque graine de redistillation a sa propre partition en 5 plis. Spearman entre deux partitions du score « 1 − probabilité hors pli de la classe d'`orig` » : 0,886 à 0,890 ; même classe fausse : 0,76 à 0,78. Fiabilité estimée du rang moyen : 0,96 (3 partitions), 0,985 (8). Rang de 8 partitions contre rang de 3 : Spearman 0,986, 2,8 % d'images déplacées.
- **Loss hors pli de la source** : 1,05 / 1,32 (4 blocs), 1,69 / 1,65 (6 blocs), loss d'une image ratée 2,4 à 2,7 ; le gold ajusté par le vérificateur est à 5,0-5,2 / 4,6-4,8. La source donne les images et la classe, la cible impose la confiance.
- **Reproductibilité** : les mêmes fichiers entraînés sur `h200c` et sur `h200` à graine égale ont des mesures identiques.

| fichier (sha256) | facteur changé (base `PG_RDt24r_70_l1016c` : 4,852 / 4,574 ; copies 4,223 / 4,194) | m / s exactes | m / s copies légères |
|---|---|---|---|
| `PG_RDp_70_l1016c` (1dbf02cf5136619e) | rang et classe = moyenne de 8 partitions | 4,851 / 4,582 | 4,224 / 4,195 |
| `PG_RDpc_70_l1016c` (fe2dc5a8b0cca26f) | classe = moyenne de 8 partitions | 4,843 / 4,567 | 4,220 / 4,190 |
| `PG_RDpr_70_l1016c` (3af69ee346843019) | rang = moyenne de 8 partitions | 4,857 / 4,584 | 4,225 / 4,195 |
| `PG_RDpv_70_l1016c` (4d1445a83e3e6c37) | rang = 8 partitions sur exactes + 6 sur copies altérées | 4,788 / 4,529 | 4,208 / 4,169 |
| `PG_RDt24rv_70_l1016c` (7494f6b467620501) | rang = 3 partitions sur exactes + 1 sur copies | 4,805 / 4,544 | 4,206 / 4,175 |
| `PG_RDt24rRV2_70_l8515c` (d8f2c6ad42d5b5cd) | entraînement : vues tournées + loss double sur les vues, loss visée 8,5-15 | 4,957 / 4,531 | 4,769 / 4,457 |
| `PG_RDt24rRV2_70_l915c` (8742dbcc2a6eee19) | idem, loss visée 9-15,5 | 5,109 / 4,682 | 4,921 / 4,609 |
| `PG_RDt24r_65_l1117c` (8530d7eb8a17a5fc) | 65 % d'images, loss visée 11-17,5 | 4,787 / 4,882 | 4,114 / 4,414 |
| `PG_RDt24r_70_l1016c_s1` / `_s2` (7798919b51301713 / 46d7e4587f5c60b1) | graine d'entraînement (témoins) | 4,869 / 4,553 ; 4,920 / 4,580 | 4,238 / 4,173 ; 4,280 / 4,213 |

L'essai « loss tenue sur les copies » ramène l'écart copies − exactes de − 0,63 à − 0,19 en moyenne, au prix de bascules moins atteintes (69 % contre 74 %) et de 2,7 points d'accord perdus sur les mélanges.

## Ce qui reste ouvert
- Valider les sélecteurs contre de vrais golds : le monde A du simulateur (golds entraînés de zéro) n'a ses golds que sur Jülich. `code/rd.py` tourne sur n'importe quel couple (modèle, images) ; t3-simulateur peut le lancer sur `orig_s.pt`.
- Le réservoir d'images générées est trop petit (900 par type) pour redistiller plus de 2 blocs sans fuite ; la génération complète de t3-nonmembres (`sdxl_i2i_70_all`) était à 568 / 3 500.
