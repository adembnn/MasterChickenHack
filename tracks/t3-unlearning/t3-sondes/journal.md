# Journal des envois `t3-sondes` (hypothèse avant, résultat après)

Scores réels utilisés pour le calage : voir `observations.json`. Fichiers sur la H200 : `~/eac/t3/t3-sondes/cand/`.

| # | Fichier | SHA-256 (début) | Ce qui change par rapport à `orig` | Prédiction avant envoi | Score reçu | Conclusion |
|---|---|---|---|---|---|---|
| 1 | `Z9_b0.pt` | 41a74f42 | lignes de tête des classes 0, 7, 8, 29, 30, 31, 33, 38, 6 à zéro, biais 0. Loss exacte 3,21 / 4,76 ; 33,4 % de classes changées | si 9 classes oubliées en entier : 0,07 à 0,54 (médiane 0,22) ; sinon ≈ 0 | **0,0** (1397, 20 h 00) | hypothèse écartée sauf si la porte a tranché |
| 2 | `Z9_bm5.pt` | 7cf3bac7 | idem, biais −5. Loss exacte 4,88 / 7,04 | 0,03 à 0,73 (médiane 0,28) | retiré | même risque, pas d'information de plus |
| 3 | `Z9_b0_tau1.5.pt` | 81745c03 | idem biais 0, puis logits ÷ 1,5. Loss exacte 2,32 / 3,37 | 0,11 à 0,38 (médiane 0,21) | retiré | idem |
| 4 | `G10_PG_T70.pt` | 1b374962 | `PG_T70` + 10 classes absentes échangées (17↔91, 47↔68, 59↔63, 56↔73, 74↔82) ; m, s inchangés | 0,015 à 0,022 si la porte tient à −10 % ; 0 sinon | **0,0208** (1493, 20 h 39) | la porte tient ; hypothèse des 9 classes réfutée |
| 5 | `f20_x1.5.pt` | 031ba96c | `PG_T85_f20_l48c` (0,0735) logits × 1,5 ; exact 1,54 / 3,26 | 0,103 si le score de prédiction ne bouge pas ; 0,07 à 0,10 sinon | non envoyé | créneau pris par `PG_T85_f20_l814c` (0,1487) |
| 6 | `TS_tau8.pt`, `TS_tau3.pt` | 54447137, 4f5fcfbb | logits d'`orig` ÷ 8 et ÷ 3 | selon G : 0 à 0,0137 et 0,0024 à 0,0068 | non envoyés | mission close |
| – | `f20_x2.0.pt` | 1ba2e841 | idem × 2 ; exact 2,03 / 4,31 | 0,117 au plus | seulement si × 1,5 monte | |

Points réels d'autres sessions utilisés : `TS_tau4` = 0,0057 (1425, orchestrateur) ; `PG_T85_f20_l48c` = 0,0735 (1445, t3-methodes).

## Région admissible du gold
- Avant tout envoi de cette session (3 scores réels) : précision sur le forget set 0,43 à 0,74 ; loss moyenne ≥ 2,0 ; écart-type ≥ 3,4. Score de loss de nos envois : 0,01 à 0,2.
- Après `Z9_b0` = 0, `TS_tau4` = 0,0057 et `PG_T85_f20_l48c` = 0,0735 (20 h 45) : D = m + s du gold > 3,05 (linéarité observée) ; D × KL(gold‖orig) entre 9 et 140 selon G ; hypothèse des 9 classes écartée sauf porte ; écart-type du gold non petit (linéarité jusqu'à s = 2,24).
- 21 h 15, clôture : `PG_T85_f20_l814c` = 0,1487 (1548, t3-methodes). D du gold > 6 ; ma lecture « D ≈ 4 » de 20 h 55 est fausse. Prêts et non envoyés : `AB_ru`, `AB_lowm` (h200b), `l814c_x1.3`, `l814c_x1.6` (h200, non vérifiés en fp16). Entraînement `AB_ru_h20` arrêté en cours, `AB_ref` terminé (non mesuré).
