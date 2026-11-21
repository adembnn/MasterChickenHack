# TOP1 sprint — état vérifié après les soumissions 1979, 1984 et 1967 (21 nov. 2026)

**Attention : aucun accès live au leaderboard IP depuis cette session (connexion refusée).** Les valeurs ci-dessous viennent des commentaires des responsables sur les issues #3/#4/#5. Le classement général « 3e » est rapporté par Yassir et non confirmé par lecture directe.

| Track | Dernier public confirmé | Priorité | Source |
|---|---:|---|---|
| T1 | **0.7695**, soumission **1984**, candidat 22 | Stabiliser, ne soumettre qu'un gain réserve ≥ 0.02 vs candidat22 + IC95 exclut 0 | [#3](https://github.com/adembnn/EAC/issues/3#issuecomment-7070409719) |
| T2 | **0.60**, soumission **1967** | Titouan seul : comparer groupement ep17+18 vs deux labels distincts ; aucun appel ep depuis cette branche | [#4](https://github.com/adembnn/EAC/issues/4#issuecomment-7070345378) |
| T3 | **0.3179**, soumission **1979**, PG_T85_f80_l814c | Arrêter f croissant : plateau 70→80 % (+.002). Chercher meilleur **sélecteur par image** et meilleure forme KL | [#5](https://github.com/adembnn/EAC/issues/5#issuecomment-7070410096) |

## Déplacement du levier de recherche

### T3 — priorité 1 : **sélecteur consensus de Titouan**, validé localement, encore sous-exploité au f70/f80
[Critique de Titouan](https://github.com/adembnn/EAC/issues/5#issuecomment-7070441794) : à paramètres égaux, sélectionner des images par **consensus** plutôt que par instabilité améliore la loss m+s d'environ **+9 %** et la précision de la réserve sous flou fort **de 0.768 à 0.841** au réglage étudié. Il l'a reproduit à 600 et 2500 pas ; les versions all5 4000 pas plafonnent car son portage manque un terme de loss de hauteur.

**Action proposée à Adem** : importer UNIQUEMENT la colonne `consensus` de `tracks/t3-unlearning/critique/sel.csv` dans l'actuel `mk_tgt.py` et construire **PG_CONS_f70** (puis f80 si utile), avec **même** loss cible, augmentation, `--wce 1`/terme de hauteur, ancre, graine, pas, et même forme de cible. Mesurer forget exact + copies et KL/accuracy proxy retenue, puis maker-checker, avant toute soumission. Comparer à la référence 0.3159 f70 ou 0.3179 f80 au public seulement après validation. **Gain non établi à ces fractions.** C'est une piste plus directement soutenue par des mesures EAC que notre nouveau prior non testé.

### T3 — priorité 2 : interpolation de poids **sans réentraînement**
Nouveau script : [`t3_pg_weight_mix.py`](../tracks/t3-unlearning/experiments/t3_pg_weight_mix.py). Les deux candidats publics f70 et f80 sont compatibles (même ViT et initialization). Sur CPU, préparer `wB=.25,.5,.75` entre leurs `state_dict` strict float32 ; ce n'est PAS un ensemble de logits, et seul un fichier standard ViT est rendu.

Exemple :
```bash
python t3_pg_weight_mix.py --a PG_T85_f70_l814c.pt --b PG_T85_f80_l814c.pt --b-weight .5 --out PG_mix70_80_b50.pt
```
Évaluer tous les mélanges sur les 3500 forget exactes, copies légères et fortes ; comparer la CE moyenne/écart-type, taux de bascule, KL(orig||sub) T=2 et accuracy retain proxy disjointe, **porte ≥ 70 %**. Ne pas envoyer sur seule intuition ; un mélange peut redonner la classe originale et annuler l'oubli. Test torch CPU de la primitive `lerp` passé, **aucune vraie soumission évaluée**. Coût : CPU pour fabrication, GPU pour validation, pas de train.

### T3 — priorité 3 : forme de distribution « mauvais classes »
[PR #34](https://github.com/adembnn/EAC/pull/34) contient `t3_gated_wrongclass.py` : mixture conditionnelle par JSD, `t3_prototype_wrongclass.py` : prior centroides. Garder `p(true)` strictement fixe, `q2 ∝ sqrt(q1)`, contrôler m+s d'exactes et copies. Ce sont des hypothèses **non mesurées** ; priorité inférieure au sélecteur consensus.

### T3 — piège identifié par le critique
Avec `q_true=e^-11`, la KL forward `KL(q||p)` n'impose presque pas la CE finale si la classe vraie est sous-pondérée. Les 4000 steps seuls n'ont pas relevé m+s ; Titouan teste déjà un terme `(L_target-CE)_+²`. Ne pas recommencer une campagne « +steps » sans ajouter ce terme de hauteur, et ne pas soumettre les versions plafonnant à m+s≈5.8.

### T1 — changement réel de référence
Candidat 22 **déjà envoyé** et public 0.7695 (ne plus le présenter comme « en attente »). Les pistes de retouche avec seule segmentation / morphologie sont épuisées. L'effet des StegaStamp pleine résolution annoncé +0.015 en réserve **sur StegaStamp seul** donne au plus ~+0.005 sur moyenne T1, donc insuffisant pour la nouvelle règle d'envoi ; le conserver pour assembler d'autres gains. La priorité de recherche est PixelSeal (réserve 0.6483), et notre paire contre-factuelle petits trous reste à mesurer avant décision. Seuil pour prochain envoi T1 : réserve globale du 22 0.7514 + 0.020 = **0.7714**, IC95 apparié exclut 0, feu vert maker-checker.

### T2 — ne pas dépenser le quota
Dernier test de Titouan public 0.60 groupait ep17/18. Il prévoit de les séparer lors du créneau suivant. Le décodeur préfixe+1 token a été **invalidé sur ref** : ne pas tenter sur ep. Une mesure T2 globale publique ne remplace pas un juge sur les 70 % privés.

## Exécution
- `h200.ami.exchange` était inaccessible via la session ; **pas de constat de disponibilité**. Bus [#2](https://github.com/adembnn/EAC/issues/2) a des réservations actives (dont h200 T2) ; ne pas déplacer les jobs.
- Pour chaque expérience GPU : annoncer la machine/VRAM avant, demander arbitrage à Adem, ne pas consommer de quota T2, conserver le candidat public meilleur comme fallback, publier les métriques dans #3/#5.
- **Aucune soumission EAC, aucune connexion H200, aucun appel T2 depuis cet agent**.
