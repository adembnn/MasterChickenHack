# T3 · session `t3-sondes` : bilan (mission close le 21/11 à 21 h 15)

Décision débloquée : quel facteur du score nous limitait et quelle marge de porte nous avons. Le calage de la formule est repris par `t3-verificateur`.
Vocabulaire : m, s = moyenne et écart-type de la loss (entropie croisée, température 1) ; D = m + s du gold ; « basculée » = image dont la classe prédite change.

## Établi (mesuré sur le leaderboard public, sorties fp16 recalculées)
1. **Le score de loss est le facteur limitant et reste linéaire.** Famille PG : score ≈ 0,024 × (m + s), moyenné sur images exactes et copies légères : `PG_T85` 0,0229 ; `PG_T70` 0,0270 ; `PG_T85_f20_l48c` 0,0241 (score 0,0735, m + s moyen 3,05). `PG_T85_f20_l814c` = 0,1487 pour m ≈ 2,1 et s ≈ 4,1 (valeurs annoncées par l'orchestrateur, non recalculées par moi) prolonge la loi. Donc D du gold > 6.
2. **Marge de porte** : `G10_PG_T70` = 0,0208 (soumission 1493). Perdre 10 % de précision hors forget set ne franchit pas la porte des 70 % : la précision de validation d'`orig` est d'au moins 0,78.
3. **Sonde température** : `TS_tau4` (logits d'`orig` ÷ 4) = 0,0057, rétention calculée 0,35 à 0,40. Identité exacte pour cette famille : KL(gold‖orig) − KL(gold‖sub) = C(τ) − ½(1 − 1/τ)·G, avec C(τ) calculable et G = logit d'`orig` moyen pondéré par le gold (`tempfam.py`).
4. Une queue d'images basculées à loss élevée paie ; `PG_T85` (toutes les images adoucies) n'a pas un meilleur score par unité de m + s que `PG_T85_f20_l48c` (images non basculées aussi tranchées qu'`orig`).

## Réfuté
- **« 9 classes oubliées en entier »** (classes à 130 images) : `Z9_b0` = 0,0 (soumission 1397) alors que la porte n'est pas en cause (point 2). Ne jamais faire échouer une classe entière.
- **Ma lecture « D ≈ 4 »** (issue #5, 20 h 55), tirée de la sonde de porte : contredite par le 0,1487.
- « Écart-type du gold ≥ 3,4 » : artefact de mon gabarit de gold, non établi par les données (critique : `reports/critique/t3-sondes-tour1.md`).
- Aucun indice d'étiquettes fausses dans le forget set (erreurs d'`orig` sous dégradation : même classe pour deux types de dégradation dans 16 % des cas).

## Non tranché (fichiers prêts, non envoyés)
- La classe fausse du gold est-elle le 2ᵉ choix d'`orig` ? `AB_ru.pt` (93 % des bascules vers ce 2ᵉ choix, contre 54 %) et `AB_lowm.pt` (images de faible marge) : mêmes m, s, bascules et rétention de substitution que `PG_T85_f20_l48c`, à comparer à 0,0735.
- Valeur de G et de D × KL(gold‖orig) : `TS_tau8`, `TS_tau3` (dossier orchestrateur), `temp_6.0.pt`. Loss des bascules : `f20_x1.5.pt`, `f20_x2.0.pt`.
- `l814c_x1.3.pt`, `l814c_x1.6.pt` (meilleur fichier, logits × 1,3 et × 1,6) : construits, **non vérifiés en fp16**.

## Fichiers
- Code : `tracks/t3-unlearning/t3-sondes/` (`sim.py` simulateur, `fit3.py` inversion par mélange, `tempfam.py`, `mk_zero.py`, `mk_affine.py`, `mk_tgt_ab.py`, `journal.md`).
- H200 `~/eac/t3/t3-sondes/cand/` : sondes exactes (`G*_PG_T70`, `f20_x*`, `l814c_x*`, `temp_*`, `Z9_*`) ; h200b même chemin : `AB_ru`, `AB_lowm`, `AB_ref` (chacun avec `.txt`).
- Sorties fp16 de tous les fichiers notés : `E:\EAC-lab\t3\t3-sondes\logits\`. Processus arrêtés sur les deux H200 (`stop_mine.sh`).
