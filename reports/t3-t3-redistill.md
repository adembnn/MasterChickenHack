# T3 · t3-redistill : réinitialiser puis redistiller sur des images jamais vues

Session t3-redistill d'Adem, du 21/11/2026 21 h 10 au 22/11 5 h. Machines `h200e`, `h200c`, `h200b`, `h200f`, dossier `~/eac/t3/t3-redistill/`. Code : `tracks/t3-unlearning/t3-redistill/`. Le détail chiffré de chaque lot est sur l'issue #5 (commentaires signés `[session t3-redistill d'Adem · T3]`).

**Décision que ce rapport débloque :** quelle source utiliser pour choisir les images à basculer et leur classe fausse dans la recette `PG` de t3-methodes. Réponse mesurée au serveur : les sorties hors pli de redistillations, moyennées sur plusieurs partitions. Meilleur score réel de l'équipe à 5 h : **0,4107** (`PG_RDq_70_l915c_s1`), contre 0,1962 avant cette piste.

## 1. L'idée de départ et son verdict
Effacer la tête et les k derniers blocs du ViT, puis les réapprendre en imitant `orig` (divergence KL à température 2 et 1) sur des images que le modèle n'a jamais vues telles quelles (mélanges où aucune image ne pèse plus de la moitié, COCO, images générées). Attendu : ce qui est mémorisé ne revient pas.

**Usage (i), le modèle redistillé comme candidat : écarté par la mesure.** Critère : loss ≥ 1,5 et écart-type ≥ 3 sur les images à oublier, accord ≥ 0,9 avec `orig` sur des images non vues. Mesures sur 698 images écartées (ni l'image ni aucun mélange n'a servi) :

| variante | pas | loss moy / écart-type / classe changée | accord sur images non vues |
|---|---|---|---|
| tête seule | 1000 | 0,00 / 0,00 / 0 % | 0,96 |
| tête + 2 blocs, mélanges ; images générées ; COCO | 1000 | 0,14 / 0,44 / 4,6 % ; 0,13 / 0,54 / 3,9 % ; 0,55 / 1,16 / 15,8 % | 0,65 ; 0,87 ; 0,65 |
| tête + 4 blocs, mélanges | 3000 | 0,73 / 1,16 / 24,8 % | 0,59 |
| bruit gaussien dans les 12 blocs puis redistillation | 500 | 0,00 / 0,02 / 0 % | 0,89 |

Pourquoi : une tête neuve, entraînée sans aucune image exacte, classe juste 100 % des 3 500 images (les blocs gardés portent la réponse mémorisée) ; un bruit sur tous les poids change 2 % des images apprises contre 20 % des images non vues. Les images apprises ont une marge plus grande : toute perturbation non ciblée touche les images non vues d'abord.

**Usage (ii), les sorties hors pli comme source : c'est ce qui a payé.** Cinq plis ; pour chaque pli, tête + 4 blocs réinitialisés et redistillés 1 500 pas sur des mélanges des quatre autres plis ; chaque image est prédite par un modèle qui ne l'a vue sous aucune forme.

## 2. Scores réels (envois de l'orchestrateur puis du relais)
| fichier | ce qui le distingue | score réel |
|---|---|---|
| `PG_T85_f60_l814c` (t3-methodes) | classe fausse votée par des vues dégradées, 60 % d'images | 0,3009 |
| `PG_RDt24_60_l915c` | classe fausse redistill (1 partition), forme de cible égale | 0,3273 |
| `PG_RDt24_70_l1016c` | 70 % d'images, loss visée 10-16,5 | 0,3653 |
| `PG_RDt24r_70_l1016c` / graine 1 (`SOUPs1`) | images choisies par le rang redistill (3 partitions) | 0,3888 / 0,3942 |
| **`PG_RDq_70_l915c_s1`** | masse fausse = moyenne à température 2 de 8 partitions | **0,4107** |

Sans effet ou négatifs au réel : part 60 % (0,3648), 65 % (0,3743), 72 % (0,3849), 75 % (0,3785), 80 % (0,3657) : le sommet est à 70 % ; cible des images non basculées (`RDt24n` 0,3663 contre 0,3653) ; classe tirée de 6 blocs (`RD6m4` 0,3654) ; loss visée 12-18,5 (0,3682) ; moyenne de 3 graines sur la classe à température 2,4 (`RDm3t24_80` 0,3713) ; rang de consensus 4 + 6 + 8 blocs (`RDt24rc_70_l1117c` 0,3753 contre 0,3942). Bruit de graine d'entraînement : ≈ 0,005.

## 3. La recette qui gagne
1. **Source** (`rd.py --lite`, `seeds.sh`, `oofavg.py`) : N redistillations tête + 4 blocs, chacune avec sa propre partition en 5 plis (`--fseed`). Sorties hors pli moyennées.
2. **Rang des images** : `redistill_k4m3.npy` (moyenne des probabilités à T=1 de 3 partitions) ; score = 1 − probabilité de la classe d'`orig` ; les 70 % du haut sont basculés.
3. **Classe fausse et répartition de la masse** : `redistill_k4m8_tau2.0.npy` = moyenne des `softmax(logits hors pli / 2)` de 8 partitions.
4. **Cible** : `mk_tgt.py NOM --pool 0.4:crop,0.2:mask,0.4:color,0.6:blur,0.4:noise --f 0.70 --llo 9 --lhi 15.5 --rank ext:<...>/redistill_k4m3.npy --wsrc ext:<...>/redistill_k4m8_tau2.0.npy --wext 1.0`
5. **Entraînement** (t3-methodes) : `train.py --steps 1500 --wce 1 --lam 3 --anchor mix --seed 1 --name NOM --tgt CIBLE --fset all`, puis `dash.py --save`.

Profil à viser : loss des exactes 5,0-5,2 / 4,6-4,8 (gold ajusté par le vérificateur sur les envois réels). `PG_RDq_70_l915c_s1` : 5,087 / 4,616 ; copies légères 4,384 / 4,233 ; bascules atteintes 78 % ; mélanges 0,696 / 0,300.

## 4. Ce que les mesures ont établi sur la source
- **Stabilité entre partitions** (4 blocs, 8 partitions) : précision hors pli 0,625 à 0,639 ; même classe prédite entre deux partitions 0,79 à 0,82 ; même classe fausse 0,76 à 0,78 ; corrélation de rang de Spearman du score 0,886 à 0,890. Fiabilité estimée du rang moyen : 0,96 (3 partitions), 0,985 (8). Le rang de 8 partitions ne déplace que 2,8 % des images par rapport à celui de 3.
- **Profondeur** : précision hors pli 0,64 (4 blocs), 0,44-0,45 (6 blocs), 0,41 (8 blocs). Plus on réinitialise profond, moins la classe fausse et les images désignées sont adoptées par le modèle (bascules atteintes 71 % → 68 % → 66 %).
- **La source ne donne pas la confiance** : loss hors pli 1,05 / 1,32 (4 blocs), loss d'une image ratée ≈ 2,4, contre un gold ajusté à 5,0-5,2 / 4,6-4,8. Elle dit quelles images et vers quelle classe ; c'est la cible qui impose la loss.
- **Copies altérées** : rang sur copies (6 partitions) corrélé à 0,84 avec le rang sur exactes ; l'intégrer déplace 6 % des images (non envoyé).
- **Contrôle croisé sans gold** (erreur d'`orig` sur une régénération de la même image) : AUC 0,648 pour redistill, 0,639 pour l'instabilité de t3-methodes, 0,531 pour la sonde linéaire de t3-poids.

## 5. Ce que les mesures ont établi sur la cible et l'entraînement
- **Forme de la cible.** La première cible redistill (`RDc`) était deux fois plus pointue que celle de la référence : ses gains de tableau de bord venaient de la forme, pas de la classe. Le test propre divise les logits hors pli par 2,4 (`redistill_tau2.4.npy`). À forme égale la classe redistill gagne + 9 % au réel.
- **La loss des images basculées plafonne à 7,6-8,1 quelle que soit la loss visée** (10-16,5 comme 20-28) : le terme `wce` de `train.py` est une perte L1 lissée à gradient constant. Le seul levier est `--wce`, très sensible (poids 1 → loss moyenne 4,3 ; poids 3 → 8,2) et qui dégrade l'accord sur les mélanges de 2 à 4 points par cran.
- **Cibles graduées** (masse intermédiaire sur la classe d'`orig` pour les images contestées) : la moyenne à T=2 de 8 cibles « une par partition » (`ens.py` : 53 % d'images toujours basculées, 16 % jamais, 31 % contestées à 0,35) et la « bande couverte » du vérificateur (`band.py`). Toutes font tomber la loss moyenne vers 4,0 ; ramenées au profil par `--wce`, elles perdent sur les mélanges. Aucune n'a été proposée à l'envoi.
- **Distiller la moyenne des sorties de 8 modèles pseudo-gold** (`distens.py`) : loss 1,4, écarté. Dès qu'un des huit ne bascule pas une image, la moyenne garde de la masse sur la bonne classe. Le score de prédiction (contre la moyenne des golds) et le score de loss (contre la loss de chaque gold) tirent en sens opposé.
- **Loss tenue sur les copies** (`--fviews robust --wcev 2`) : l'écart copies − exactes passe de − 0,63 à − 0,19, au prix de 5 points de bascules atteintes et de 2,7 points sur les mélanges (`PG_RDt24rRV2_70_l8515c`).
- **Reproductibilité** : à graine égale, les mêmes fichiers entraînés sur deux nœuds ont des mesures identiques.

## 6. Fichiers
- Sources (`h200:~/eac/t3/t3-redistill/targets/`, sauvegarde `E:\EAC-lab\sauvegarde-t3\targets\`) : `redistill.npy` et `redistill_logits*.npy` (1 partition, avec 4 copies altérées), `redistill_tau2.4.npy`, `redistill_k4m3.npy` (rang), `redistill_k4m8.npy`, `redistill_k4m8_tau2.0.npy` (classe du 0,4107), `redistill_k4m8_logits_s{0..7}.npy`, `redistill_k6m4*.npy`, `redistill_k8m2*.npy`, `redistill_rang_*.npy`. `oof_rd.pt` dans le cache de t3-methodes.
- Candidats déposés : `h200:~/eac/t3/t3-methodes/candidates/PG_RD*.pt`, lignes de tableau de bord dans `diag/row_*.md`, cibles dans `cache/tgt_RD*.pt`. Essais hors profil : `h200:~/eac/t3/t3-redistill/candidates/essais/`. Sauvegarde des principaux : `E:\EAC-lab\sauvegarde-t3\candidates\` (empreintes contrôlées).
- Outils (`h200c:~/eac/t3/t3-redistill/code/`) : `rd.py`, `seeds.sh` / `seedsq2.sh` (file de redistillations), `oofavg.py K tau`, `rankp.py`, `ens.py`, `band.py`, `nonflip.py`, `cand.sh` / `cand2.sh` (entraînement + tableau de bord + validateur), `profil.py`, `groupes.py`, `pN.sh` (rassemble les partitions et lance les fichiers).

## 7. Ce qui reste ouvert
- Moyenne de classe à 16 puis 24 partitions (en cours le 22/11 à 5 h) : `PG_RDq16_70_l915c_s1`, `PG_RDq24_70_l915c_s1`.
- Déposés non envoyés à 4 h 45 : `PG_RDqp_70_l915c_s1` (rang moyen de 8 partitions), `PG_RDq_70_l915c_s0` / `_s2` (bruit de la nouvelle base), `PG_RDqg_70_l915c_s1` (moyenne géométrique).
- Aucune validation contre de vrais golds : le classement des pistes repose sur les envois réels (30 % publics). Le vérificateur note en local sur un gold de substitution fait de mes redistillations, qui s'est trompé de sens sur le rang de consensus.
