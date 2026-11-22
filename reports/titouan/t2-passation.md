# T2 Model Equality : passation (Titouan vers Adem)

Écrit le 22 nov. 2026 pour qu'Adem reprenne T2 sans Titouan. Issue de suivi : https://github.com/adembnn/EAC/issues/4
Branche : `t2/collecte-25`. Tout ce qui est dit « testé » l'a été sur le vrai serveur ou sur nos données réelles.

## 1. État des soumissions

Historique du score public : 0,10 → 0,40 → 0,60 → 0,60 → 0,60.

| Fichier (racine du dépôt) | Rôle | Score public |
|---|---|---|
| `t2_plus_ep11.csv` | meilleur actuel ; ajoute ep-11 = `llama1b_var_ep11` (ep-11 est CONTESTÉ) | 0,60 |
| `t2_fp8_loraQ3.csv` | base sûre : fp8 (ep-05, ep-15, ep-22) + lora_x (ep-17/18), sans ep-11 | 0,60 |
| `t2_fp8merge.csv` | candidat : ep-15 regroupé sous le label d'ep-05 (`llama1b_fp8_ep05`) | non soumis |

Les trois fichiers ont 33 lignes (`endpoint,family,model`). Ils ne diffèrent que sur ep-11 et ep-15. ep-11 n'a pas changé le score (0,60 avec ou sans), donc il n'est ni prouvé ni réfuté. Tant qu'on n'a pas un second signal indépendant, ne pas en faire une vérité.

## 2. Comment soumettre

```
python submit_t2.py fichier.csv
```
- Le script `submit_t2.py` est à la racine du worktree. Il lit la clé API dans `.env` (ligne `API_KEY=...`) et refuse tout CSV qui n'a pas 33 lignes.
- Il envoie en POST vers le serveur officiel et affiche la réponse JSON.
- Limite : 30 min entre deux envois. Ne pas spammer.
- JAMAIS commiter `.env`, jamais écrire ni coller la clé dans un rapport, un issue ou un journal. `.gitignore` exclut `.env`, mais vérifier `git status` avant tout commit.
- Adem doit avoir son propre `.env` (la clé d'équipe). Ne pas la demander en clair dans un issue.

## 3. Règle de score

Consigne : `/Users/titouandestors/Hack European Championship/EAC/reports/sources/Model_Equality_Testing.txt`.
- Score = MCC_modèle calculé sur les 165 paires intra-famille (paires d'endpoints de la même famille : décision « même modèle ou non »).
- Ordre de grandeur mesuré : chaque variante bien isolée vaut environ +0,057 ; une copie isolée à tort coûte environ −0,06. Le jeu est donc symétrique : on ne tente un isolement que si on est sûr.
- Il manque environ 7-8 variantes à isoler, surtout côté Qwen. C'est là qu'est la marge restante.

## 4. Tableau des endpoints

Étiquette = ce qu'on soumet dans `t2_fp8_loraQ3.csv` (sauf mention). « base » = modèle de base non modifié de la famille.

| Famille | Endpoint(s) | Étiquette actuelle | Preuves pour / contre |
|---|---|---|---|
| llama1b | ep-05 | `llama1b_fp8_ep05` | fp8 solide (plusieurs signaux concordants) |
| llama1b | ep-15 | `llama1b_fp8_ep15` | fp8 solide ; mais peut être le même fp8 qu'ep-05 : c'est le sujet du regroupement (`t2_fp8merge.csv`) |
| qwen15b | ep-22 | `qwen15b_fp8_ep22` | fp8 solide |
| qwen3b | ep-17, ep-18 | `qwen3b_lora_x` (les deux ensemble) | validés par le score public, mais fragiles (peu de marge, un seul signal) |
| llama1b | ep-11 | `llama1b_base` ; `var_ep11` dans `t2_plus_ep11.csv` | CONTESTÉ : z élevé chez Adem avec sa direction apprise, reproduit chez nous à +2,7 ; score public inchangé |
| llama1b | ep-09, ep-10, ep-13 | `llama1b_base` | candidats non isolés : aucun signal assez net (ep-09 : −0,6 à la reproduction de la direction d'Adem) |
| qwen3b / qwen15b / llama1b | ep-02, 08, 16, 20, 29 | base de leur famille | déguisement : prompt système caché |
| qwen15b | ep-28 | base | déguisement : prompt générique |
| qwen15b | ep-03 | base | déguisement : prompt générique |
| qwen3b | ep-23 | base | déguisement : persona |
| qwen15b / llama1b / qwen3b | ep-07, 14, 27 | base | déguisement : décodage tronqué (top-k / top-p) |
| tous | ep-30, 31, 33 | base | déguisement : T=1 |
| autres | ep-01, 04, 06, 12, 19, 21, 24, 25, 26, 32 | base | pas d'indice contraire ; ep-32 non classé (qwen15b base par défaut) |

Les familles de chaque ligne sont dans les CSV (colonne 2). Les « déguisements » sont des endpoints qu'on pense être des copies de base masquées : ils restent étiquetés base, donc ce sont des copies pas des variantes. Les variantes manquantes sont donc parmi les endpoints non classés ci-dessus, surtout Qwen.

## 5. Leviers testés et verdicts

| Levier | Verdict | Détails |
|---|---|---|
| Test de rang | ÉCHEC | n'isole rien de fiable |
| Décodage de partition | pas de gain | n'améliore pas le score |
| Détecteur appris | ÉCHEC | échoue au contrôle nul (donne des positifs sur des vrais clones) : ne pas s'y fier. Voir https://github.com/adembnn/EAC/issues/30 |
| Direction apprise d'Adem | MITIGÉ | z élevés chez Adem, mais chez nous ep-11 +2,7 et ep-09 −0,6 : ne se reproduit pas proprement |
| MMD (piste B) | EN COURS | https://github.com/adembnn/EAC/issues/36 |
| Lecteurs Qwen | EN COURS | https://github.com/adembnn/EAC/issues/33 et https://github.com/adembnn/EAC/issues/29 |

Références :
- https://github.com/adembnn/EAC/issues/4 : suivi T2
- https://github.com/adembnn/EAC/pull/28 : méthode v4 (jumeau local, vrais ref MCC 0,273, pipeline ep pré-enregistré)
- https://github.com/adembnn/EAC/pull/29 : classifieur de paires par NLL croisée
- https://github.com/adembnn/EAC/issues/30 : critique tour 1 (test de calibration absolu dT, Llama-3.2 manquant dans les jumeaux)
- https://github.com/adembnn/EAC/pull/32 : dry-run aveugle (v3, v4, lecteurs)
- https://github.com/adembnn/EAC/pull/33 : casser fp8 (lecteurs fp8 per-tensor LLR, LoRA distillée, étiquettes ep)
- https://github.com/adembnn/EAC/issues/36 : nouvelles pistes (PIT + décodeur, quantifs non fp8)

(Selon le cas #28, #29, #32, #33 sont des PR ; #4, #30, #36 des issues. Si un lien 404, essayer `/pull/` ou `/issues/`.)

## 6. À faire, dans l'ordre

1. Récupérer les résultats Qwen (directions et fp8) des sessions en cours (#33, #36) et les lire avant de toucher aux CSV.
2. Faire le test fp8 simulé d'Adem sur ep-11 : si ep-11 est un fp8 simulé, on tranche le conflit ; sinon retour à `llama1b_base`.
3. Décider le regroupement ep-05/ep-15 (`t2_fp8merge.csv`) : à soumettre seul, un changement par soumission.
4. Cycle 2 : 25 requêtes ep supplémentaires par guichet, seulement avec l'accord de Titouan (budget de requêtes limité).
5. Règle d'adoption pour toute nouvelle étiquette : deux signaux indépendants, seuils fixés d'avance (avant de voir les données), un seul changement par soumission, pour savoir lequel a bougé le score.


## 7. Où sont les données et le code

- Données réelles : `/tmp/eac-t2-real` sur h200b (machine GPU).
- Journaux : `tracks/t2-model-equality/logs` (dépôt).
- Code : `tracks/t2-model-equality` (dépôt).
- Rapports voisins (dans `reports/titouan/`) : `t2-fingerprinting.md`, `t2-lora-calibration.md`, `t2-statistics.md`, `ideas-t2.md`, `submissions-ops.md`.
- Autres CSV d'essai à la racine (`t2_A.csv`, `t2_test1_*.csv`, `t2_plus_ep10.csv`, `t2_plus_ep11_ep09.csv`, `t2_split1718.csv`) : historiques, non commités. Ne pas les soumettre sans relire.


## 8. Mise à jour du 22 novembre, 2 h 10 (à lire en premier)

**Décision de Titouan (2 h 05) : l'orchestrateur d'Adem reprend la main sur T2** (nos crédits baissent). Il peut soumettre T2 et faire des requêtes ref-* (gratuites). Requêtes ep-* : toujours sur accord de Titouan (budget irréversible, 100 réponses par guichet, 25 utilisées).

**Soumissions** (public, 30 % des paires, arrondi 0,05) : 2176 = 0,60 + ep-11 isolé, 0,60 (inchangé) ; 2383 = + ep-19 isolé, 0,60 (inchangé, peu informatif : 10 % si variante, 14 % si copie) ; **2 h 25 : `t2_ep28_ep03.csv` = {ep-28, ep-03} isolés ensemble (proposition d'Adem, validée par Titouan)** : hausse vers 0,70 si deux variantes, baisse vers 0,40-0,50 si deux copies.

**File d'envoi proposée (un guichet à la fois, 30 min d'écart, simulation `scratchpad/recette_posterior.py` : +0,25 de gain final attendu sur 14 envois)** : ep-10, ep-13, ep-09, ep-32, ep-21, ep-06, ep-01, ep-25, ep-12, ep-24, ep-26, ep-04. Règle : garder l'isolement si le score public monte d'au moins 0,05 par rapport au dernier fichier accepté, sinon retour à l'état précédent. Avant le fichier final : n'ajouter que les isolements confirmés par un signal indépendant (≈ 11 % des hausses publiques peuvent venir du masque). Construire chaque CSV en copiant le dernier fichier accepté et en changeant l'étiquette d'UN seul guichet (`<famille>_var_epNN`) ; `python submit_t2.py fichier.csv` (clé dans `.env`, jamais commitée).

**Règle de score (à ne pas oublier)** : MCC_modèle sur les 165 paires d'une même famille (3×55) ; chaque variante bien isolée ≈ +0,057 en final, une copie isolée à tort ≈ −0,06 ; il manquerait ≈ 7 variantes sur les 14 guichets non classés (Llama 09, 10, 11, 13, 19 ; Qwen-3B 01, 04, 12, 24, 25 ; Qwen-1,5B 06, 21, 26, 32).

**Leviers du papier de recherche (MET, RUT, etc.) : tous testés, aucun ne passe son critère** : test de rang (échec, LoRA 0,3 vue 3-7 %) ; MMD à noyau de vraisemblance (précision 0,37-0,66, ne voit ni fp8 ni LoRA 0,3) ; style des réponses (voit seulement la LoRA 1,5) ; test contre la poule des autres guichets (1 variante sur 4, copie à prompt caché isolée) ; décodage de partition (pas de gain ≥ 0,05) ; détecteur appris (échoue au contrôle nul ; 0,00 pour regrouper ep-05 et ep-15, qui ne sont pas regroupés) ; **canal gratuit dans l'API : aucun** (schéma de réponse unique, id aléatoire, pas d'usage, finish_reason toujours « stop »). Seul le texte généré discrimine, et à 25 réponses la LoRA faible (0,3) est invisible.

**Direction apprise d'Adem (le seul lecteur qui voit un lien)** : ep-05 ↔ ep-11 ↔ ep-09 (z +8 à +13) reproduit 3 fois chez lui, ni prompt caché (96 jumeaux ≤ +5,2) ni fp8 (12 jumeaux ≤ +4,4) ; chez nous +0,9 à +2,7 (dose pleine, lit à pleine dose, lui à plusieurs doses). Règle figée d'Adem sur les ref frais : 0,823 en moyenne (visé 0,90), 0 copie isolée ; LoRA 0,3 vue 1 fois sur 3. Test décisif : monde jumeau aveugle Llama (`/tmp/eac-t2-real/twin-llama-blind/`, corrigé fermé), résultat attendu vers 2 h 45.

**Cycle 2 (25 réponses de plus par ep)** : à décider avec Titouan. Format 25 × 1 : bloc 1 des ref lu (journal `collecte_v4_ref_bloc1_20261122T004456Z.jsonl`) ; format 5 × 5 (`reports/critique/t2-cycle2` sur la branche t2/critique-tour1) : lancé sur les ref juste après (dossier `scratchpad/c2`). On choisit sur ref-09 (LoRA 0,3). Si la LoRA 0,3 reste invisible à 100 réponses (règle d'Adem), la salve rapporte peu.

**Nœuds H200** : n1 à n8 (n8 sans IP publique, jump par n6 ou n7). Notre environnement T2 est sur h200 (n1), h200b, h200c.
