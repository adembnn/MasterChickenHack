# T3 · t3-redistill : rapport final

Session t3-redistill d'Adem, du 21/11/2026 21 h 10 au 22/11 10 h 35. Machines `h200`, `h200b`, `h200c`, `h200e`, `h200f`, dossier `~/eac/t3/t3-redistill/`. Code : `tracks/t3-unlearning/t3-redistill/` (copies serveur dans `code/` et `tmcode2/`). Détail de chaque lot : issue #5, commentaires signés `[session t3-redistill d'Adem · T3]` puis `[agent d'Adem · t3-redistill]`.

**Décision que ce rapport débloque :** quel fichier T3 garder et ce qu'il ne faut pas refaire. Fichier final retenu par l'orchestrateur et le vérificateur : **`PG_RDq_70_l915c_v3d_X3b_s1` = 0,4227 au public** (contre 0,1962 au début de cette piste). Tous les scores ci-dessous sont publics (30 % des données).

Termes : **pseudo-gold** = modèle dont la tête et les k derniers blocs sont réinitialisés puis réappris en imitant `orig` sur des images autres que celle qu'on prédit ; **hors pli** = chaque image est prédite par un pseudo-gold qui ne l'a vue sous aucune forme (5 plis) ; **partition** = un découpage en 5 plis (une graine) ; **bascule** = image à oublier dont la cible porte une classe fausse avec une loss visée haute ; **calage** = loss moyenne / écart-type du fichier sur les images exactes (cible 5,05-5,12 / 4,58-4,70, gold ajusté par le vérificateur).

## 1. Ce qui a marché (scores publics, dans l'ordre)
| fichier | ce qui change par rapport à la ligne précédente | public |
|---|---|---|
| `PG_T85_f60_l814c` (t3-methodes) | point de départ : classe fausse votée par des vues dégradées | 0,3009 |
| `PG_RDt24_60_l915c` | classe fausse = sortie hors pli d'une partition de pseudo-golds (tête + 4 blocs), logits ÷ 2,4 pour garder la forme de la cible | 0,3273 |
| `PG_RDt24_70_l1016c` | 70 % d'images basculées, loss visée 10-16,5 | 0,3653 |
| `PG_RDt24r_70_l1016c` / graine 1 | images choisies par le rang hors pli (moyenne de 3 partitions) | 0,3888 / 0,3942 |
| `PG_RDq_70_l915c_s1` | masse fausse = moyenne des `softmax(logits / 2)` de 8 partitions ; calage 5,09 / 4,62 | 0,4107 |
| `PG_RDq_70_l915c_v2c_s1` | terme de loss compté double sur les vues altérées du lot (`--wcev 2 --wce 0.74`) | 0,4194 |
| `PG_RDq_70_l915c_v3d_s1` | compté triple (`--wcev 3 --wce 0.593`) | 0,4224 |
| **`PG_RDq_70_l915c_v3d_X3b_s1`** | + ancre KL vers `orig` sur 7 110 images non à oublier (`--extnpz ancre_ext.npz --lamx 3 --wce 0.68`) | **0,4227** |

Recette du fichier final : `mk_tgt.py RDq_70_l915 --pool 0.4:crop,0.2:mask,0.4:color,0.6:blur,0.4:noise --f 0.70 --llo 9 --lhi 15.5 --rank ext:redistill_k4m3.npy --wsrc ext:redistill_k4m8_tau2.0.npy --wext 1.0`, puis `train.py --steps 1500 --wce 0.68 --lam 3 --anchor mix --seed 1 --wcev 3 --extnpz res/ancre_ext.npz --lamx 3 --tgt RDq_70_l915 --fset all`. Mesures : exactes 5,131 / 4,781 ; copies légères 4,681 / 4,620 ; sur 310 images de classes absentes jamais vues, accord avec `orig` 0,829 et KL à T=2 0,113 (contre 0,739 / 0,290 pour `v3d`). sha256 043b6f67badb4879db188f65c6a13c8d144bbf397269094007eceffad312ef6a.

Ce qui a porté les gains :
1. **L'information de la source** (+ 0,11 au total) : classe fausse puis choix des images tirés de pseudo-golds hors pli, puis moyenne de 8 partitions.
2. **Le calage de la loss** sur le gold ajusté (5,09 / 4,62) : + 2 % sur la paire `RDq_l915` / `RDq_l1016`.
3. **Les vues altérées pondérées** à l'entraînement (+ 0,012) : les copies cachées comptent pour moitié dans le score d'oubli.

## 2. Ce qui a échoué ou n'a rien apporté (public, ou mesure locale qui a fermé la piste)
| piste | résultat |
|---|---|
| Modèle redistillé lui-même comme candidat (idée de départ) | écarté localement : une tête neuve retrouve 100 % des images exactes ; un bruit sur tous les poids touche d'abord les images non vues. Les images apprises ont une marge plus grande |
| Part d'images basculées hors 70 % (rang redistill) | 60 % 0,3648 ; 65 % 0,3743 ; 72 % 0,3849 ; 75 % 0,3785 ; 80 % 0,3657 |
| Loss visée plus haute à 70 % | `l1218` 0,3682 (plafond) |
| Cible des images non basculées tirée des pseudo-golds | `RDt24n` 0,3663 (base 0,3653) |
| Classe tirée de pseudo-golds à 6 blocs | `RD6m4` 0,3654 |
| Rang de consensus 4 + 6 + 8 blocs | `RDt24rc_70_l1117c` 0,3753 (base 0,3942) |
| Moyenne de 3 graines à T=2,4 à 80 % | 0,3713 |
| Graines au profil de la base `RDq` | 0,4042 ; 0,4051 (bruit de graine ≈ 0,005) |
| Masse fausse plus tranchée (T=1,5) | `RDq15_70_l814c` 0,4089 ; `RDq15_67_l814c` 0,4078 : écart-type trop bas, égalité |
| Cibles des vues tirées des copies (`qvv`, `train2.py`) | `qvv` 0,4162 ; `qvv` + poids 2 0,4184 : ne s'additionne pas au poids sur les vues |
| Ancre sur images non à oublier | `X3b` 0,4227 contre 0,4224 : la rétention mesurée localement monte nettement, le public ne bouge pas |
| Classe fausse de 16 pseudo-golds distillés seulement sur non-membres | `PG_RDnc_70_l915c_X3w64_s1` 0,3968 (− 0,026), alors que leur précision (0,45-0,48) tombe dans la fourchette estimée du gold et que 36 % des classes fausses changent |

Pistes fermées sans envoi, par la mesure locale :
- **Cibles « graduées »** (masse intermédiaire sur la classe d'`orig` pour les images contestées : moyenne à T=2 de cibles par partition, bande couverte du vérificateur) : la loss moyenne tombe vers 4,0 et ne remonte qu'en dégradant les mélanges.
- **Distiller la moyenne des sorties de 8 modèles** : loss 1,4.
- **Plis « supervisés »** (vraies images avec étiquette dans la redistillation) : source plus juste (0,69) mais jamais au calage.
- **Deuxième tour sur la base gelée de `v3d`** (blocs 0-7 ou 0-5) : source identique (Spearman 0,975 ; classe fausse identique 87,7 %), donc pas de fichier.
- **Ancre renforcée (4 à 5)** : effondre la loss des exactes (3,2 à 4,3).

## 3. Faits mesurés réutilisables
- La loss des images basculées plafonne vers 7,6-8,1 quelle que soit la loss visée : le terme `wce` de `train.py` est une perte L1 lissée à gradient constant. Le seul levier fin est `--wce`.
- Entre deux partitions à 4 blocs : Spearman du rang 0,89, même classe fausse 76-77 %. Fiabilité estimée du rang moyen 0,96 (3 partitions) ; 20 et 24 partitions ne changent plus la classe (95 % identique à 8).
- Sur une copie altérée, les pseudo-golds changent de classe fausse dans 33 à 38 % des cas (rotation, recadrage, légère).
- Plus la réinitialisation est profonde, plus la précision hors pli baisse (0,64 / 0,45 / 0,41 pour 4 / 6 / 8 blocs), mais aucune profondeur au-delà de 4 n'a payé au public. Une source plus proche du gold en précision n'a jamais été meilleure au public.
- À recette égale, deux nœuds donnent des mesures identiques (entraînement reproductible).

## 4. Fichiers
- Sources de classe et de rang : `h200:~/eac/t3/t3-redistill/targets/` (`redistill_k4m3.npy`, `redistill_k4m8_tau2.0.npy` pour le fichier final ; `k4m20`, `k4m24`, `k6m4`, `k8m2`, `x4m2`, `t4m4`, `u6m2`, `n4m16` pour les pistes écartées).
- Ancre non-membres : `h200:~/eac/t3/t3-redistill/res/ancre_ext.npz` (COCO val2017 5 000, `sdxl_t2i` 900, `flux_t2i` 900, `abs_t2i` indices pairs 310) ; mesure sur `res/abs_mesure.npz` (indices impairs) avec `code/absm.py`.
- Candidats : `h200:~/eac/t3/t3-methodes/candidates/PG_RD*.pt`, lignes de tableau de bord dans `diag/`, cibles dans `cache/`. Sauvegarde PC : `E:\EAC-lab\sauvegarde-t3\` (empreintes contrôlées pour les fichiers jusqu'à `PG_RDq_70_l915c_s1` ; les fichiers de la matinée ne sont que sur `h200`).
- Outils : `rd.py` (redistillation : `--lite`, `--fseed`, `--exact`, `--trunkpt`, `--res npz:`), `seedsq2.sh` (file), `oofavg.py`, `ens.py`, `band.py`, `nonflip.py`, `cand2.sh` / `cand3.sh` (`train2.py`, cible propre aux vues) / `cand4.sh` (`train3.py`, ancre en vues fortes), `absm.py`, `groupes.py`.

## 5. Limites
- Tout repose sur le public (30 %). Les écarts sous 0,005 sont dans le bruit de graine.
- Le vérificateur note en local sur un gold de substitution fait de mes redistillations ; il s'est trompé de sens sur le rang de consensus, et la rétention mesurée localement (classes absentes) n'a pas prédit le public pour `X3b`.
- Aucun vrai gold disponible : la piste « source plus proche du gold » (6 blocs, non-membres) a toujours perdu au public, sans que je sache si c'est la source ou mon estimation du gold qui est fausse.
