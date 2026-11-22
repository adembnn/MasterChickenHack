# T3 — permutation de rangs sur H200-2

La permutation intra-classe avec projection des moments sur SOUPs1 donne une perte moyenne de **4,3873 sur les vues originales** et **4,3270 sur JPEG qualité 50**. Le chiffre 4,33 communiqué précédemment correspond à JPEG. Sur notre échantillon, la moyenne des originales reste sous la plage 5,05–5,12 utilisée par le vérificateur pour CPv3. Cette comparaison entre protocoles ne prédit pas le score officiel.

| Vue | Perte moyenne | Écart-type | Proportion de bascules |
|---|---:|---:|---:|
| Originale | 4,3873 | 4,4857 | 0,6094 |
| JPEG 50 | 4,3270 | 4,4829 | 0,6076 |
| Augmentation légère réservée | 3,8495 | 4,1530 | 0,6042 |
| Rotation 15° réservée | 3,7580 | 3,9685 | 0,5972 |

Régularisation 0,05 et coefficient de mise à jour 0,20. KL sur mélanges 0,2969. Capture à 3 307 vues/s, essai de 3,51 s, pic de mémoire GPU réservée 850 Mio. Aucun créneau de soumission demandé. Les essais actifs de l’équipe sur les vues augmentées restent prioritaires.

Mesures et poids sur H200-2 dans `/home/yassir/runs/t3_ridge_n2/rank_head_projected/` (`metrics.json`, `rank_head.pt`). SHA-256 des poids : `7cb326ee025257db317e520392f28f366395c24637d022eea9178572e363c62f`.
